
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data_processing import (
    load_source_data,
    build_monthly_panel,
    FEATURES,
)
from src.modeling import make_pipeline, get_candidate_models


def calculate_metrics(name, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "Method": name,
        "Rows": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2
            else np.nan
        ),
    }


def main():
    print("Loading EduPro data...")

    courses, teachers, transactions = load_source_data()

    panel = build_monthly_panel(
        courses,
        transactions,
        teachers,
    )

    months = sorted(panel["Month"].unique())

    if len(months) != 12:
        raise ValueError(
            f"Expected 12 months, found {len(months)}."
        )

    train_validation = panel[
        panel["Month"].isin(months[:10])
    ].copy()

    test = panel[
        panel["Month"].isin(months[10:])
    ].copy()

    print("\nTraining rows:", len(train_validation))
    print("Testing rows:", len(test))
    print("Test months:", test["Month"].unique())

    factories = get_candidate_models()

    # Model names were selected using validation data.
    enrollment_model = make_pipeline(
        factories["Lasso Regression"]()
    )

    revenue_model = make_pipeline(
        factories["Ridge Regression"]()
    )

    enrollment_model.fit(
        train_validation[FEATURES],
        train_validation["Enrollments"],
    )

    revenue_model.fit(
        train_validation[FEATURES],
        train_validation["Revenue"],
    )

    # Use the same holdout features for both methods.
    enrollment_predictions = np.maximum(
        enrollment_model.predict(test[FEATURES]),
        0,
    )

    # The dashboard rounds predicted enrollments.
    enrollment_predictions = np.rint(
        enrollment_predictions
    )

    price_based = (
        enrollment_predictions
        * test["CoursePrice"].to_numpy(dtype=float)
    )

    ridge_predictions = np.maximum(
        revenue_model.predict(test[FEATURES]),
        0,
    )

    baseline_mean = float(
        train_validation["Revenue"].mean()
    )

    baseline_predictions = np.full(
        len(test),
        baseline_mean,
    )

    actual = test["Revenue"].to_numpy(dtype=float)

    is_free = (
    test["CoursePrice"].eq(0)
    | test["CourseType"].astype(str).str.lower().eq("free")
).to_numpy()

    hybrid_predictions = np.where(
    is_free,
    0.0,
    ridge_predictions
)

    methods = {
    "Ridge Regression": ridge_predictions,
    "Price × Predicted Enrollments": price_based,
    "Hybrid (Free=0, Paid=Ridge)": hybrid_predictions,
    "Training Mean Baseline": baseline_predictions,
}


    results = pd.DataFrame([
        calculate_metrics(name, actual, predicted)
        for name, predicted in methods.items()
    ]).sort_values("RMSE")

    print("\n=== REVENUE METHOD COMPARISON ===")
    print(results.round(4).to_string(index=False))

    # Inspect free-course and paid-course performance.
    details = test[
        [
            "Month",
            "CourseID",
            "CourseCategory",
            "CourseType",
            "CoursePrice",
            "Enrollments",
            "Revenue",
        ]
    ].copy()

    details["PredictedEnrollments"] = (
        enrollment_predictions
    )
    details["RevenuePriceBased"] = price_based
    details["RevenueRidge"] = ridge_predictions
    details["RevenueHybrid"] = hybrid_predictions
    details["RevenueBaseline"] = baseline_predictions

    details["IsFree"] = (
        details["CoursePrice"].eq(0)
        | details["CourseType"]
        .astype(str)
        .str.lower()
        .eq("free")
    )

    print("\n=== FREE-COURSE CHECK ===")

    free_test = details[details["IsFree"]]

    print("Free-course test rows:", len(free_test))
    print(
        "Actual free-course revenue:",
        round(free_test["Revenue"].sum(), 2)
    )
    print(
        "Free-course rows with positive actual revenue:",
        int((free_test["Revenue"] > 0).sum())
    )
    print(
        "Ridge predicted free-course revenue:",
        round(free_test["RevenueRidge"].sum(), 2)
    )


    segmented_results = []

    for segment, mask in [
        ("Free", details["IsFree"]),
        ("Paid", ~details["IsFree"]),
    ]:
        subset = details.loc[mask]

        if subset.empty:
            continue

        for name, column in [
            ("Ridge Regression", "RevenueRidge"),
            (
                "Price × Predicted Enrollments",
                "RevenuePriceBased",
            ),
            (
                "Hybrid (Free=0, Paid=Ridge)",
                "RevenueHybrid",
            ),
            (
                "Training Mean Baseline",
                "RevenueBaseline",
            ),
        ]:
            row = calculate_metrics(
                name,
                subset["Revenue"],
                subset[column],
            )

            row["Segment"] = segment
            segmented_results.append(row)

    segmented = pd.DataFrame(segmented_results)

    print("\n=== FREE VS PAID PERFORMANCE ===")
    print(
        segmented.round(4).to_string(index=False)
    )

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    results.to_csv(
        output_dir / "revenue_method_comparison.csv",
        index=False,
    )

    segmented.to_csv(
        output_dir / "revenue_method_segments.csv",
        index=False,
    )

    details.to_csv(
        output_dir / "revenue_test_predictions.csv",
        index=False,
    )

    print("\nSaved reports:")
    print("reports/revenue_method_comparison.csv")
    print("reports/revenue_method_segments.csv")
    print("reports/revenue_test_predictions.csv")


if __name__ == "__main__":
    main()
