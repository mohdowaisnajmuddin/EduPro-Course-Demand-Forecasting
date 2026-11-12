
from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / (
    "demo" if os.getenv("EDUPRO_DEMO_MODE") == "1" else "raw"
)


# ---------------------------------------------------------
# 1. LOAD AND VALIDATE SOURCE DATA
# ---------------------------------------------------------

def require_columns(df, required_columns, filename):
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )


def load_source_data():
    courses = pd.read_csv(DATA / "courses.csv")
    teachers = pd.read_csv(DATA / "teachers.csv")

    transactions = pd.read_csv(
        DATA / "transactions.csv",
        parse_dates=["TransactionDate"]
    )

    require_columns(
        courses,
        [
            "CourseID",
            "CourseName",
            "CourseCategory",
            "CourseType",
            "CourseLevel",
            "CoursePrice",
            "CourseDuration",
            "CourseRating"
        ],
        "courses.csv"
    )

    require_columns(
        teachers,
        [
            "TeacherID",
            "TeacherName",
            "Expertise",
            "YearsOfExperience",
            "TeacherRating"
        ],
        "teachers.csv"
    )

    require_columns(
        transactions,
        [
            "TransactionID",
            "UserID",
            "CourseID",
            "TransactionDate",
            "Amount",
            "PaymentMethod",
            "TeacherID"
        ],
        "transactions.csv"
    )

    if courses["CourseID"].duplicated().any():
        raise ValueError("Duplicate CourseID values found.")

    if transactions["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID values found.")

    if transactions["TransactionDate"].isna().any():
        raise ValueError("Missing or invalid transaction dates found.")

    if transactions["Amount"].isna().any():
        raise ValueError("Missing transaction amounts found.")

    if (transactions["Amount"] < 0).any():
        raise ValueError("Negative transaction amounts found.")

    unknown_courses = (
        set(transactions["CourseID"])
        - set(courses["CourseID"])
    )

    if unknown_courses:
        raise ValueError(
            f"Unknown CourseID values: {sorted(unknown_courses)}"
        )

    return courses, teachers, transactions


# ---------------------------------------------------------
# 2. COURSE FEATURE ENGINEERING
# ---------------------------------------------------------

def add_course_features(df):
    df = df.copy()

    df["PriceBand"] = pd.cut(
        df["CoursePrice"],
        bins=[-0.01, 0, 150, 350, np.inf],
        labels=["Free", "Low", "Medium", "High"]
    ).astype(str)

    df["DurationBucket"] = pd.cut(
        df["CourseDuration"],
        bins=[-np.inf, 10, 30, np.inf],
        labels=["Short", "Medium", "Long"]
    ).astype(str)

    df["RatingTier"] = pd.cut(
        df["CourseRating"],
        bins=[-np.inf, 2, 3.5, 4.5, np.inf],
        labels=["Low", "Moderate", "Good", "Excellent"]
    ).astype(str)

    return df


# ---------------------------------------------------------
# 3. CALENDAR FEATURES
# ---------------------------------------------------------

def add_calendar_features(df):
    df = df.copy()

    df["MonthNumber"] = df["Month"].dt.month
    df["Quarter"] = df["Month"].dt.quarter

    df["MonthSin"] = np.sin(
        2 * np.pi * df["MonthNumber"] / 12
    )

    df["MonthCos"] = np.cos(
        2 * np.pi * df["MonthNumber"] / 12
    )

    first_month = df["Month"].min()

    df["TimeIndex"] = (
        (df["Month"].dt.year - first_month.year) * 12
        + df["Month"].dt.month
        - first_month.month
    )

    return df


# ---------------------------------------------------------
# 4. HISTORICAL ENROLLMENT FEATURES
# ---------------------------------------------------------

def add_historical_features(panel, group_column):
    """
    Create historical enrollment and revenue features.

    All lag and rolling features use previous months only.
    Missing early history remains NaN for model imputation.
    """
    panel = panel.sort_values(
        [group_column, "Month"]
    ).copy()

    grouped_enrollments = panel.groupby(
        group_column, sort=False
    )["Enrollments"]

    panel["EnrollLag1"] = grouped_enrollments.shift(1)
    panel["EnrollLag2"] = grouped_enrollments.shift(2)

    panel["EnrollRolling3"] = grouped_enrollments.transform(
        lambda s: s.shift(1).rolling(
            window=3, min_periods=1
        ).mean()
    )

    grouped_revenue = panel.groupby(
        group_column, sort=False
    )["Revenue"]

    panel["RevenueLag1"] = grouped_revenue.shift(1)
    panel["RevenueLag2"] = grouped_revenue.shift(2)

    panel["RevenueRolling3"] = grouped_revenue.transform(
        lambda s: s.shift(1).rolling(
            window=3, min_periods=1
        ).mean()
    )

    panel["RevenuePerEnrollment"] = np.where(
        panel["Enrollments"] > 0,
        panel["Revenue"] / panel["Enrollments"],
        0.0
    )

    grouped_rpe = panel.groupby(
        group_column, sort=False
    )["RevenuePerEnrollment"]

    panel["RevenuePerEnrollmentLag1"] = grouped_rpe.shift(1)
    panel["RevenuePerEnrollmentLag2"] = grouped_rpe.shift(2)

    panel["RevenuePerEnrollmentRolling3"] = (
        grouped_rpe.transform(
            lambda s: s.shift(1).rolling(
                window=3, min_periods=1
            ).mean()
        )
    )

    return panel



# ---------------------------------------------------------
# 5. COURSE-MONTH DATASET
# ---------------------------------------------------------


def add_teacher_features(panel, teachers, transactions):
    """
    Add course-month teacher features using only
    teacher assignments observed before each month.

    Each distinct teacher counts once per course.
    """

    teacher_columns = [
        "TeacherID",
        "Expertise",
        "YearsOfExperience",
        "TeacherRating"
    ]

    teacher_info = teachers[teacher_columns].drop_duplicates(
        subset=["TeacherID"]
    )

    assignments = transactions[
        ["CourseID", "TeacherID", "TransactionDate"]
    ].dropna(subset=["CourseID", "TeacherID"]).copy()

    assignments = assignments.merge(
        teacher_info,
        on="TeacherID",
        how="left",
        validate="many_to_one"
    )

    assignments["FirstObserved"] = assignments[
        "TransactionDate"
    ].dt.to_period("M").dt.to_timestamp()

    assignments = (
        assignments.groupby(
            ["CourseID", "TeacherID"],
            as_index=False
        )
        .agg(
            FirstObserved=("FirstObserved", "min"),
            Expertise=("Expertise", "first"),
            YearsOfExperience=("YearsOfExperience", "first"),
            TeacherRating=("TeacherRating", "first")
        )
    )

    course_months = panel[
        ["CourseID", "Month", "CourseCategory"]
    ].copy()

    joined = course_months.merge(
        assignments,
        on="CourseID",
        how="left"
    )

    joined = joined[
        joined["FirstObserved"] < joined["Month"]
    ].copy()

    joined["ExpertiseMatch"] = (
        joined["Expertise"].astype(str).str.strip().str.lower()
        ==
        joined["CourseCategory"].astype(str).str.strip().str.lower()
    ).astype(float)

    summary = joined.groupby(
        ["CourseID", "Month"],
        as_index=False
    ).agg(
        AvgTeacherExperience=("YearsOfExperience", "mean"),
        AvgTeacherRating=("TeacherRating", "mean"),
        ExpertiseMatchScore=("ExpertiseMatch", "mean"),
        TeacherCount=("TeacherID", "nunique")
    )

    panel = panel.merge(
        summary,
        on=["CourseID", "Month"],
        how="left",
        validate="one_to_one"
    )

    panel["TeacherCount"] = panel["TeacherCount"].fillna(0)

    panel["ExperienceBucket"] = pd.cut(
        panel["AvgTeacherExperience"],
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=[
            "Beginner",
            "Intermediate",
            "Experienced",
            "Expert"
        ]
    ).astype("object")

    return panel


def build_monthly_panel(courses, transactions, teachers=None):
    """
    Build one row per course per month.

    Enrollments include both free and paid transactions.
    Revenue is the sum of transaction amounts.
    """

    courses = add_course_features(courses)

    tx = transactions.copy()

    tx["Month"] = (
        tx["TransactionDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = tx.groupby(
        ["CourseID", "Month"]
    ).agg(
        Enrollments=("TransactionID", "count"),
        Revenue=("Amount", "sum"),
        PaidEnrollments=(
            "Amount",
            lambda values: (values > 0).sum()
        ),
        FreeEnrollments=(
            "Amount",
            lambda values: (values == 0).sum()
        )
    ).reset_index()

    months = pd.date_range(
        start=tx["Month"].min(),
        end=tx["Month"].max(),
        freq="MS"
    )

    full_index = pd.MultiIndex.from_product(
        [
            courses["CourseID"].unique(),
            months
        ],
        names=["CourseID", "Month"]
    ).to_frame(index=False)

    panel = (
        full_index
        .merge(
            monthly,
            on=["CourseID", "Month"],
            how="left"
        )
        .merge(
            courses,
            on="CourseID",
            how="left",
            validate="many_to_one"
        )
    )

    count_columns = [
        "Enrollments",
        "PaidEnrollments",
        "FreeEnrollments"
    ]

    panel[count_columns] = (
        panel[count_columns].fillna(0)
    )

    panel["Revenue"] = panel["Revenue"].fillna(0)

    panel = add_calendar_features(panel)

    panel = add_historical_features(
        panel,
        group_column="CourseID"
    )

    if teachers is not None:
     panel = add_teacher_features(
        panel,
        teachers,
        transactions
    )

    return panel.sort_values(
        ["CourseID", "Month"]
    ).reset_index(drop=True)




# ---------------------------------------------------------
# 6. CATEGORY-MONTH DATASET
# ---------------------------------------------------------

def build_category_monthly_panel(courses, transactions):
    """
    Build one row per course category per month.

    This dataset supports category-level demand and revenue
    analysis. Revenue is aggregated from actual transactions.
    """

    course_categories = courses[
        ["CourseID", "CourseCategory"]
    ].drop_duplicates()

    tx = transactions.merge(
        course_categories,
        on="CourseID",
        how="left",
        validate="many_to_one"
    )

    tx["Month"] = (
        tx["TransactionDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = tx.groupby(
        ["CourseCategory", "Month"]
    ).agg(
        Enrollments=("TransactionID", "count"),
        Revenue=("Amount", "sum"),
        PaidEnrollments=(
            "Amount",
            lambda values: (values > 0).sum()
        ),
        FreeEnrollments=(
            "Amount",
            lambda values: (values == 0).sum()
        )
    ).reset_index()

    categories = courses[
        "CourseCategory"
    ].dropna().unique()

    months = pd.date_range(
        start=tx["Month"].min(),
        end=tx["Month"].max(),
        freq="MS"
    )

    full_index = pd.MultiIndex.from_product(
        [categories, months],
        names=["CourseCategory", "Month"]
    ).to_frame(index=False)

    panel = full_index.merge(
        monthly,
        on=["CourseCategory", "Month"],
        how="left"
    )

    count_columns = [
        "Enrollments",
        "PaidEnrollments",
        "FreeEnrollments"
    ]

    panel[count_columns] = (
        panel[count_columns].fillna(0)
    )

    panel["Revenue"] = panel["Revenue"].fillna(0)

    panel = add_calendar_features(panel)

    panel = add_historical_features(
        panel,
        group_column="CourseCategory"
    )

    return panel.sort_values(
        ["CourseCategory", "Month"]
    ).reset_index(drop=True)


# ---------------------------------------------------------
# 7. MODEL FEATURE DEFINITIONS
# ---------------------------------------------------------


NUMERIC_FEATURES = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "AvgTeacherExperience",
    "AvgTeacherRating",
    "ExpertiseMatchScore",
    "TeacherCount",
    "MonthNumber",
    "Quarter",
    "MonthSin",
    "MonthCos",
    "TimeIndex",
    "EnrollLag1",
    "EnrollLag2",
    "EnrollRolling3",
    "RevenueLag1",
    "RevenueLag2",
    "RevenueRolling3",
    "RevenuePerEnrollmentLag1",
    "RevenuePerEnrollmentLag2",
    "RevenuePerEnrollmentRolling3"
]

CATEGORICAL_FEATURES = [
    "CourseID",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "PriceBand",
    "DurationBucket",
    "RatingTier",
    "ExperienceBucket"
]


FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

# Enrollment is the ML prediction target.
# Revenue is calculated from predicted enrollments
# and course prices in the forecasting stage.
TARGETS = ["Enrollments"]
