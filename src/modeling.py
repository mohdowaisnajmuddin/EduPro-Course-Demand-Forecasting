
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .data_processing import (
    load_source_data,
    build_monthly_panel,
    build_category_monthly_panel,
    add_course_features,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# 1. PREPROCESSING AND MODEL PIPELINE
# ---------------------------------------------------------

def make_pipeline(estimator):

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", prep),
        ("model", estimator)
    ])


# ---------------------------------------------------------
# 2. MODEL CANDIDATES
# ---------------------------------------------------------

def get_candidate_models():

    return {
        "Linear Regression": lambda: LinearRegression(),

        "Ridge Regression": lambda: Ridge(
            alpha=1.0
        ),

        "Lasso Regression": lambda: Lasso(
            alpha=0.01,
            max_iter=10000
        ),

        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            loss="huber",
            random_state=42
        )
    }


# ---------------------------------------------------------
# 3. EVALUATION METRICS
# ---------------------------------------------------------

def metric_row(split, model_name, actual, predicted, target="Enrollments"):

    predicted = np.clip(
        np.asarray(predicted),
        0,
        None
    )

    return {
        "Split": split,
        "Target": target,
        "Model": model_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


# ---------------------------------------------------------
# 4. BASELINE MODELS
# ---------------------------------------------------------

def get_baseline_predictions(train, test):
    """
    Generate four baseline predictions using training history only.
    """

    train = train.copy()
    test = test.copy()

    train["Month"] = pd.to_datetime(train["Month"])
    test["Month"] = pd.to_datetime(test["Month"])

    global_mean = train["Enrollments"].mean()

    # Baseline 1: overall training mean
    global_predictions = np.full(len(test), global_mean)

    # Baseline 2: each course's training mean
    course_means = train.groupby(
        "CourseID"
    )["Enrollments"].mean()

    course_predictions = (
        test["CourseID"]
        .map(course_means)
        .fillna(global_mean)
        .to_numpy()
    )

    # Baseline 3 and 4: previous actual observations
    history = pd.concat(
        [train, test],
        ignore_index=True
    ).sort_values(["CourseID", "Month"])

    grouped = history.groupby("CourseID")["Enrollments"]

    history["PreviousMonth"] = grouped.shift(1)

    history["Previous3MonthAverage"] = (
        grouped.transform(
            lambda values: values.shift(1).rolling(
                window=3,
                min_periods=3
            ).mean()
        )
    )

    time_baselines = history[
        history["Month"].isin(test["Month"].unique())
    ].copy()

    time_baselines = time_baselines.set_index(
        ["CourseID", "Month"]
    )

    test_keys = pd.MultiIndex.from_frame(
        test[["CourseID", "Month"]]
    )

    previous_month = (
        time_baselines["PreviousMonth"]
        .reindex(test_keys)
        .fillna(global_mean)
        .to_numpy()
    )

    previous_3_month = (
        time_baselines["Previous3MonthAverage"]
        .reindex(test_keys)
        .fillna(global_mean)
        .to_numpy()
    )

    return {
        "Global Mean Baseline": global_predictions,
        "Course Mean Baseline": course_predictions,
        "Previous Month Baseline": previous_month,
        "Previous 3-Month Average": previous_3_month
    }



# ---------------------------------------------------------
# 5. COURSE-LEVEL RECURSIVE FORECAST
# ---------------------------------------------------------

def recursive_forecast(
    panel,
    model,
    horizon=3,
    revenue_model=None
):
    """
    Forecast enrollments and revenue recursively.

    Revenue methods:
    1. Direct prediction using the Ridge revenue model.
    2. Predicted enrollments multiplied by course price.

    Future historical features use prior predictions,
    not future actual outcomes.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    sorted_panel = panel.sort_values(
        ["CourseID", "Month"]
    )

    static = (
        sorted_panel.groupby("CourseID")
        .tail(1)
        .set_index("CourseID")
    )

    enrollment_history = {}
    revenue_history = {}

    for course_id, group in sorted_panel.groupby("CourseID"):
        enrollment_history[course_id] = (
            group["Enrollments"].tail(3).astype(float).tolist()
        )
        revenue_history[course_id] = (
            group["Revenue"].tail(3).astype(float).tolist()
        )

    first_month = panel["Month"].min()
    last_month = panel["Month"].max()

    future_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    outputs = []

    for month in future_months:
        rows = []

        for course_id, base in static.iterrows():
            enroll = enrollment_history[course_id]
            revenue = revenue_history[course_id]

            row = base.to_dict()

            row.update({
                "CourseID": course_id,
                "Month": month,
                "MonthNumber": month.month,
                "Quarter": month.quarter,
                "MonthSin": np.sin(
                    2 * np.pi * month.month / 12
                ),
                "MonthCos": np.cos(
                    2 * np.pi * month.month / 12
                ),
                "TimeIndex": (
                    (month.year - first_month.year) * 12
                    + month.month - first_month.month
                ),
                "EnrollLag1": enroll[-1],
                "EnrollLag2": enroll[-2],
                "EnrollRolling3": float(
                    np.mean(enroll[-3:])
                ),
                "RevenueLag1": revenue[-1],
                "RevenueLag2": revenue[-2],
                "RevenueRolling3": float(
                    np.mean(revenue[-3:])
                )
            })

            rpe_history = [
                r / e if e > 0 else 0.0
                for e, r in zip(enroll, revenue)
            ]

            row["RevenuePerEnrollmentLag1"] = (
                rpe_history[-1]
            )
            row["RevenuePerEnrollmentLag2"] = (
                rpe_history[-2]
            )
            row["RevenuePerEnrollmentRolling3"] = (
                float(np.mean(rpe_history[-3:]))
            )

            rows.append((course_id, row))

        x = pd.DataFrame(
            [item[1] for item in rows]
        )[FEATURES]

        enrollment_predictions = np.clip(
            model.predict(x), 0, None
        )

        if revenue_model is not None:
            ridge_predictions = np.clip(
                revenue_model.predict(x), 0, None
            )
        else:
            ridge_predictions = [np.nan] * len(rows)

        for (course_id, row), enrollment_pred, ridge_pred in zip(
            rows,
            enrollment_predictions,
            ridge_predictions
        ):
            predicted_enrollment = max(
                0, int(round(float(enrollment_pred)))
            )

            course_price = float(row["CoursePrice"])

            price_revenue = (
                predicted_enrollment * course_price
            )

            ridge_revenue = float(ridge_pred)
            hybrid_revenue = (
    0.0 if course_price <= 0 else ridge_revenue
)

            enrollment_history[course_id].append(
                float(predicted_enrollment)
            )

            # Use direct revenue forecast for recursive
            # revenue history when available.
            history_revenue = (
                ridge_revenue
                if revenue_model is not None
                else price_revenue
            )

            revenue_history[course_id].append(
                history_revenue
            )

            outputs.append({
    "Month": month,
    "CourseID": course_id,
    "PredictedEnrollments": predicted_enrollment,
    "CoursePrice": course_price,
    "PredictedRevenue": price_revenue,
    "PredictedRevenuePriceBased": price_revenue,
    "PredictedRevenueRidge": ridge_revenue,
    "PredictedRevenueHybrid": hybrid_revenue,
})
    result = pd.DataFrame(outputs)

    course_details = panel[
        ["CourseID", "CourseName", "CourseCategory"]
    ].drop_duplicates()

    return result.merge(
        course_details,
        on="CourseID",
        how="left",
        validate="many_to_one"
    )


# ---------------------------------------------------------
# 6. CATEGORY-LEVEL FORECAST
# ---------------------------------------------------------


def build_category_forecast(course_forecast):
    """
    Aggregate course forecasts by month and category,
    retaining both revenue prediction methods.
    """

    category_forecast = (
        course_forecast
        .groupby(
            ["Month", "CourseCategory"],
            as_index=False
        )
        .agg(
            PredictedEnrollments=(
                "PredictedEnrollments",
                "sum"
            ),
            PredictedRevenue=(
                "PredictedRevenue",
                "sum"
            ),
            PredictedRevenuePriceBased=(
                "PredictedRevenuePriceBased",
                "sum"
            ),
            PredictedRevenueRidge=(
    "PredictedRevenueRidge",
    "sum"
),
PredictedRevenueHybrid=(
    "PredictedRevenueHybrid",
    "sum"
)
        )
    )

    return category_forecast



# ---------------------------------------------------------
# 7. COMPLETE TRAINING AND EVALUATION
# ---------------------------------------------------------

def train_project():

    courses, teachers, transactions = load_source_data()

    panel = build_monthly_panel(
    courses,
    transactions,
    teachers
)

    category_panel = build_category_monthly_panel(
        courses,
        transactions
    )

    unique_months = sorted(
        panel["Month"].unique()
    )

    if len(unique_months) < 12:
        raise ValueError(
            "At least 12 months are required for the "
            "current train/validation/test design."
        )

    # First 8 months: training
    # Next 2 months: validation
    # Final 2 months: untouched test

    train_months = unique_months[:8]
    validation_months = unique_months[8:10]
    test_months = unique_months[10:]

    train = panel[
        panel["Month"].isin(train_months)
    ].copy()

    validation = panel[
        panel["Month"].isin(validation_months)
    ].copy()

    train_validation = panel[
        panel["Month"].isin(
            train_months + validation_months
        )
    ].copy()

    test = panel[
        panel["Month"].isin(test_months)
    ].copy()

    print("\n=== CHRONOLOGICAL SPLIT ===")

    print(
        "Training:",
        train["Month"].min(),
        "to",
        train["Month"].max()
    )

    print(
        "Validation:",
        validation["Month"].min(),
        "to",
        validation["Month"].max()
    )

    print(
        "Testing:",
        test["Month"].min(),
        "to",
        test["Month"].max()
    )

    candidate_factories = get_candidate_models()

    metrics = []
    selected = {}

    # -----------------------------------------------------
    # MODEL SELECTION USING VALIDATION DATA ONLY
    # -----------------------------------------------------

    print("\n=== MODEL SELECTION ===")

    for name, factory in candidate_factories.items():

        pipe = make_pipeline(factory())

        pipe.fit(
            train[FEATURES],
            train["Enrollments"]
        )

        predictions = pipe.predict(
            validation[FEATURES]
        )

        row = metric_row(
            "Validation",
            name,
            validation["Enrollments"],
            predictions
        )

        metrics.append(row)

        print(
            f"{name}: "
            f"Validation RMSE = {row['RMSE']:.4f}"
        )

            # Compare baseline methods on the same validation rows.
    validation_baselines = get_baseline_predictions(
        train,
        validation
    )

    for name, predictions in validation_baselines.items():
        row = metric_row(
            "Validation",
            name,
            validation["Enrollments"],
            predictions
        )
        metrics.append(row)

    # Select the best machine-learning model for forecasting.
    validation_ml_metrics = [
        row for row in metrics
        if row["Split"] == "Validation"
        and row["Model"] in candidate_factories
    ]

    best_model = min(
        validation_ml_metrics,
        key=lambda row: row["RMSE"]
    )

    selected["Enrollments"] = best_model["Model"]

    # -----------------------------------------------------
    # REVENUE MODEL SELECTION
    # -----------------------------------------------------

    print("\n=== REVENUE MODEL SELECTION ===")

    revenue_validation_metrics = []

    for name, factory in candidate_factories.items():

        revenue_pipe = make_pipeline(factory())

        revenue_pipe.fit(
            train[FEATURES],
            train["Revenue"]
        )

        revenue_predictions = revenue_pipe.predict(
            validation[FEATURES]
        )

        row = metric_row(
            "Validation",
            name,
            validation["Revenue"],
            revenue_predictions,
            target="Revenue"
        )

        metrics.append(row)
        revenue_validation_metrics.append(row)

        print(
            f"{name}: "
            f"Revenue Validation RMSE = {row['RMSE']:.4f}"
        )

    best_revenue_model = min(
        revenue_validation_metrics,
        key=lambda row: row["RMSE"]
    )

    selected["Revenue"] = best_revenue_model["Model"]

    print(
        "Best Revenue machine-learning model:",
        selected["Revenue"]
    )

    # Also report the best method overall, including baselines.
    validation_all_metrics = [
        row for row in metrics
        if row["Split"] == "Validation"
        and row["Target"] == "Enrollments"
    ]


    best_overall = min(
        validation_all_metrics,
        key=lambda row: row["RMSE"]
    )

    selected["BestValidationMethod"] = best_overall["Model"]

    print(
        "\nBest machine-learning model:",
        selected["Enrollments"]
    )

    print(
        "Best validation method overall:",
        selected["BestValidationMethod"]
    )



    # -----------------------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------------------

    print("\n=== FINAL TEST EVALUATION ===")

    final_factory = candidate_factories[
        selected["Enrollments"]
    ]

    test_model = make_pipeline(
        final_factory()
    )

    test_model.fit(
        train_validation[FEATURES],
        train_validation["Enrollments"]
    )

    test_predictions = test_model.predict(
        test[FEATURES]
    )

    test_result = metric_row(
        "Test",
        selected["Enrollments"],
        test["Enrollments"],
        test_predictions
    )

    metrics.append(test_result)

    # -----------------------------------------------------
    # REVENUE TEST EVALUATION
    # -----------------------------------------------------

    revenue_test_factory = candidate_factories[
        selected["Revenue"]
    ]

    revenue_test_model = make_pipeline(
        revenue_test_factory()
    )

    revenue_test_model.fit(
        train_validation[FEATURES],
        train_validation["Revenue"]
    )

    revenue_test_predictions = revenue_test_model.predict(
        test[FEATURES]
    )

    revenue_test_result = metric_row(
        "Test",
        selected["Revenue"],
        test["Revenue"],
        revenue_test_predictions,
        target="Revenue"
    )

    metrics.append(revenue_test_result)



    # -----------------------------------------------------
    # REFIT SELECTED MODEL ON ALL AVAILABLE HISTORY
    # -----------------------------------------------------

    print("\n=== FINAL MODEL FIT ===")

    final_model = make_pipeline(
        final_factory()
    )

    final_model.fit(
        panel[FEATURES],
        panel["Enrollments"]
    )

    revenue_final_model = make_pipeline(
        candidate_factories[selected["Revenue"]]()
    )

    revenue_final_model.fit(
        panel[FEATURES],
        panel["Revenue"]
    )

    final_models = {
        "Enrollments": final_model,
        "Revenue": revenue_final_model
    }



    # -----------------------------------------------------
    # FUTURE FORECAST
    # -----------------------------------------------------

    forecast = recursive_forecast(
    panel,
    final_model,
    horizon=3,
    revenue_model=revenue_final_model
)

    category_forecast = build_category_forecast(
        forecast
    )

    metrics_df = pd.DataFrame(metrics)

    print("\n=== TEST METRICS ===")

    print(
        metrics_df[
            metrics_df["Split"] == "Test"
        ].to_string(index=False)
    )

    return {
        "courses": courses,
        "teachers": teachers,
        "transactions": transactions,
        "panel": panel,
        "category_panel": category_panel,
        "metrics": metrics_df,
        "selected": selected,
        "models": final_models,
        "forecast": forecast,
        "category_forecast": category_forecast,
        "test_result": test_result
    }


# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ---------------------------------------------------------

def feature_importance(bundle, target="Enrollments"):
    """
    Extract feature importance from a trained sklearn Pipeline.

    Accepts either:
    1. A trained Pipeline directly.
    2. A dictionary containing models and a target.
    """

    # Support both pipeline and dictionary input
    if isinstance(bundle, dict):
        model = bundle["models"][target]
    else:
        model = bundle

    names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))

    else:
        return pd.DataFrame(
            columns=["Feature", "Importance"]
        )

    return pd.DataFrame({
        "Feature": names,
        "Importance": values
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------
# 9. SCENARIO ANALYSIS
# ---------------------------------------------------------


def make_scenario_row(
    bundle,
    course_id,
    price,
    duration,
    level,
    exp,
    rating
):
    panel = bundle["panel"]

    matching = panel[
        panel["CourseID"] == course_id
    ].sort_values("Month")

    if matching.empty:
        raise ValueError(
            f"Course ID not found: {course_id}"
        )

    row = matching.iloc[-1].copy()

    next_month = (
        panel["Month"].max()
        + pd.offsets.MonthBegin(1)
    )

    first_month = panel["Month"].min()

    # Apply scenario inputs.
    row["CoursePrice"] = float(price)
    row["CourseDuration"] = float(duration)
    row["CourseLevel"] = level
    row["AvgTeacherExperience"] = float(exp)
    row["AvgTeacherRating"] = float(rating)

    row["ExperienceBucket"] = pd.cut(
        [float(exp)],
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=[
            "Beginner",
            "Intermediate",
            "Experienced",
            "Expert"
        ]
    )[0]

    # Update the calendar features.
    row["Month"] = next_month
    row["MonthNumber"] = next_month.month
    row["Quarter"] = next_month.quarter

    row["MonthSin"] = np.sin(
        2 * np.pi * next_month.month / 12
    )

    row["MonthCos"] = np.cos(
        2 * np.pi * next_month.month / 12
    )

    row["TimeIndex"] = (
        (next_month.year - first_month.year) * 12
        + next_month.month
        - first_month.month
    )

    # Refresh historical features for the next month.
    enroll = matching["Enrollments"].tail(3).astype(float).tolist()
    revenue = matching["Revenue"].tail(3).astype(float).tolist()

    row["EnrollLag1"] = enroll[-1]
    row["EnrollLag2"] = enroll[-2]
    row["EnrollRolling3"] = float(np.mean(enroll))

    row["RevenueLag1"] = revenue[-1]
    row["RevenueLag2"] = revenue[-2]
    row["RevenueRolling3"] = float(np.mean(revenue))

    rpe = [
        r / e if e > 0 else 0.0
        for e, r in zip(enroll, revenue)
    ]

    row["RevenuePerEnrollmentLag1"] = rpe[-1]
    row["RevenuePerEnrollmentLag2"] = rpe[-2]
    row["RevenuePerEnrollmentRolling3"] = float(np.mean(rpe))

    # Recalculate derived course features.
    course_features = add_course_features(
        pd.DataFrame([row])
    ).iloc[0]

    row["PriceBand"] = course_features["PriceBand"]
    row["DurationBucket"] = course_features["DurationBucket"]
    row["RatingTier"] = course_features["RatingTier"]

    return pd.DataFrame(
        [row[FEATURES]],
        columns=FEATURES
    )


def predict_scenario(bundle, scenario):
    enrollment_model = bundle["models"]["Enrollments"]
    revenue_model = bundle["models"]["Revenue"]

    predicted_enrollments = max(
        0.0,
        float(enrollment_model.predict(scenario)[0])
    )

    predicted_enrollments_rounded = max(
        0,
        int(round(predicted_enrollments))
    )

    price = float(scenario["CoursePrice"].iloc[0])

    price_based_revenue = (
        predicted_enrollments_rounded * price
    )

    ridge_revenue = max(
        0.0,
        float(revenue_model.predict(scenario)[0])
    )
    hybrid_revenue = (
    0.0
    if price <= 0
    else ridge_revenue
)

    return {
    "Enrollments": predicted_enrollments,
    "Revenue": price_based_revenue,
    "RevenuePriceBased": price_based_revenue,
    "RevenueRidge": ridge_revenue,
    "RevenueHybrid": hybrid_revenue,
}
