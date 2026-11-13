
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EduPro | Course Demand & Revenue Forecasting",
    page_icon="📚",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dashboard"


# =========================================================
# 2. LOAD AND VALIDATE DEPLOYMENT DATA
# =========================================================

@st.cache_data
def load_data():

    required_files = {
        "history": "monthly_history.csv",
        "category_history": "category_annual_history.csv",
        "metrics": "model_metrics.csv",
        "forecast": "category_forecast.csv",
        "enrollment_importance": "feature_importance_enrollments.csv",
        "revenue_importance": "feature_importance_revenue.csv",
    }

    loaded = {}

    for key, filename in required_files.items():

        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required deployment file missing: {filename}"
            )

        loaded[key] = pd.read_csv(path)

    # Parse date columns.
    for key in ["history", "forecast"]:
        loaded[key]["Month"] = pd.to_datetime(
            loaded[key]["Month"],
            errors="coerce"
        )

        if loaded[key]["Month"].isna().any():
            raise ValueError(
                f"Invalid dates found in {required_files[key]}"
            )

    # Validate required columns.
    required_columns = {
        "history": [
            "Month", "Enrollments", "Revenue"
        ],
        "category_history": [
    "CourseCategory", "Enrollments", "Revenue"
],
        "forecast": [
            "Month",
            "CourseCategory",
            "PredictedEnrollments",
            "PredictedRevenuePriceBased",
            "PredictedRevenueRidge",
            "PredictedRevenueHybrid",
        ],
        "metrics": [
            "Split", "Target", "Model", "MAE", "RMSE", "R2"
        ],
        "enrollment_importance": [
            "Feature", "Importance"
        ],
        "revenue_importance": [
            "Feature", "Importance"
        ],
    }

    for key, columns in required_columns.items():

        missing = set(columns) - set(loaded[key].columns)

        if missing:
            raise ValueError(
                f"{required_files[key]} is missing columns: "
                f"{sorted(missing)}"
            )

    return loaded


st.title("📚 EduPro | Course Demand & Revenue Forecasting")

st.caption(
    "Unified Mentor Internship Project | "
    "Historical Analytics, Forecasting and Planning"
)

try:
    data = load_data()

except Exception as exc:

    st.error(f"Unable to load dashboard data: {exc}")

    st.info(
        "Check the data/deployment folder and verify "
        "that all six required CSV files are available."
    )

    st.stop()


history = data["history"]
category_history = data["category_history"]
metrics = data["metrics"]
forecast = data["forecast"]

enrollment_importance = data["enrollment_importance"]
revenue_importance = data["revenue_importance"]


st.info(
    "This public dashboard uses aggregated historical data "
    "and precomputed forecasts from the internship analysis. "
    "It does not expose individual transaction records. "
    "Scenario predictions are illustrative calculations, "
    "not outputs from the original trained ML models."
)


# =========================================================
# 3. GLOBAL NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Demand Forecast",
        "Revenue Forecast",
        "Category Comparison",
        "Model Evaluation",
        "Feature Importance",
        "Scenario Predictor",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Historical period: January–December 2025"
)

st.sidebar.caption(
    "Forecast period: January–March 2026"
)


# =========================================================
# 4. REVENUE METHOD DEFINITIONS
# =========================================================

revenue_methods = {
    "Price-based": "PredictedRevenuePriceBased",
    "Ridge Regression": "PredictedRevenueRidge",
    "Hybrid": "PredictedRevenueHybrid",
}


# =========================================================
# 5. OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Historical Performance Overview")

    total_enrollments = history["Enrollments"].sum()
    total_revenue = history["Revenue"].sum()

    total_courses_categories = (
        category_history["CourseCategory"].nunique()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Historical Enrollments",
        f"{total_enrollments:,.0f}"
    )

    col2.metric(
        "Historical Revenue",
        f"{total_revenue:,.2f}"
    )

    col3.metric(
        "Course Categories",
        f"{total_courses_categories:,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Monthly Enrollments")

        monthly = history.sort_values("Month")

        fig = px.line(
    monthly,
    x="Month",
    y="Enrollments",
    markers=True,
    title="Monthly Historical Enrollments",
    labels={
        "Month": "Month",
        "Enrollments": "Total Enrollments",
    },
)

    fig.update_traces(
    line_color="#2563EB",
    line_width=3,
    marker_size=9,
)

    fig.update_layout(
    height=400,
    showlegend=False,
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="overview_enrollments_chart",
)

    with right:

        st.subheader("Monthly Revenue")

        fig = px.line(
    monthly,
    x="Month",
    y="Enrollments",
    markers=True,
    title="Monthly Historical Enrollments",
    labels={
        "Month": "Month",
        "Enrollments": "Total Enrollments",
    },
)

    fig.update_traces(
    line_color="#2563EB",
    line_width=3,
    marker_size=9,
)

    fig.update_layout(
    height=400,
    showlegend=False,
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="overview_revenue_chart",
)

    st.subheader("Historical Monthly Summary")

    st.dataframe(
        monthly,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Historical revenue is calculated using transaction "
        "Amount aggregated by month. Individual transactions "
        "are not displayed."
    )


# =========================================================
# 6. DEMAND FORECAST
# =========================================================

elif page == "Demand Forecast":

    st.header("Course Demand Forecast")

    categories = sorted(
        forecast["CourseCategory"].dropna().unique()
    )

    selected_categories = st.multiselect(
        "Select course categories",
        categories,
        default=categories,
    )

    filtered = forecast[
        forecast["CourseCategory"].isin(
            selected_categories
        )
    ].copy()

    if filtered.empty:

        st.warning(
            "Select at least one course category."
        )

    else:

        monthly = (
            filtered.groupby("Month")[
                "PredictedEnrollments"
            ]
            .sum()
            .reset_index()
        )

        st.metric(
            "Predicted Enrollments (3 Months)",
            f"{monthly['PredictedEnrollments'].sum():,.0f}"
        )

        st.subheader("Monthly Demand Forecast")

        monthly_chart = monthly.copy()

        monthly_chart["ForecastMonth"] = (
            monthly_chart["Month"].dt.strftime("%b %Y")
        )

        fig_monthly_demand = px.bar(
            monthly_chart,
            x="ForecastMonth",
            y="PredictedEnrollments",
            color="ForecastMonth",
            color_discrete_sequence=[
                "#2563EB",
                "#8B5CF6",
                "#06B6D4",
            ],
            text_auto=",.0f",
            title="Monthly Predicted Enrollments — Jan to Mar 2026",
            labels={
                "ForecastMonth": "Forecast Month",
                "PredictedEnrollments": "Predicted Enrollments",
            },
        )

        fig_monthly_demand.update_layout(
            showlegend=False,
            height=430,
            bargap=0.4,
            yaxis=dict(rangemode="tozero"),
        )

        fig_monthly_demand.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        st.plotly_chart(
            fig_monthly_demand,
            use_container_width=True,
            key="monthly_demand_forecast_chart",
        )

        st.subheader("Category-Level Predictions")

        category_totals = (
            filtered.groupby(
                "CourseCategory",
                as_index=False
            )["PredictedEnrollments"]
            .sum()
            .sort_values(
                "PredictedEnrollments",
                ascending=False
            )
        )

        fig = px.bar(
    category_totals,
    x="CourseCategory",
    y="PredictedEnrollments",
    color="CourseCategory",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="Predicted Enrollments by Course Category",
    labels={
        "CourseCategory": "Course Category",
        "PredictedEnrollments": "Predicted Enrollments",
    },
    text_auto=".0f",
)

        fig.update_layout(
    showlegend=False,
    xaxis_tickangle=-35,
    height=500,
)

        fig.update_traces(
    textposition="outside",
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="demand_category_chart",
)

    st.dataframe(
            filtered.sort_values(
                ["Month", "PredictedEnrollments"],
                ascending=[True, False]
            ),
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
            "Download Demand Forecast CSV",
            data=filtered.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="edupro_demand_forecast.csv",
            mime="text/csv",
        )


# =========================================================
# 7. REVENUE FORECAST
# =========================================================

elif page == "Revenue Forecast":

    st.header("Revenue Forecast Comparison")

    selected_method = st.selectbox(
        "Select Revenue Forecasting Method",
        list(revenue_methods.keys()),
    )

    selected_column = revenue_methods[
        selected_method
    ]

    total_forecast = forecast[
        selected_column
    ].sum()

    st.metric(
        f"Total Forecast Revenue — {selected_method}",
        f"{total_forecast:,.2f}"
    )

    monthly = (
        forecast.groupby("Month", as_index=False)
        [selected_column]
        .sum()
    )

    st.subheader("Monthly Revenue Forecast")

    monthly_revenue_chart = monthly.copy()

    monthly_revenue_chart["ForecastMonth"] = (
    monthly_revenue_chart["Month"].dt.strftime("%b %Y")
)

    fig_monthly_revenue = px.bar(
    monthly_revenue_chart,
    x="ForecastMonth",
    y=selected_column,
    color="ForecastMonth",
    color_discrete_sequence=[
        "#10B981",
        "#3B82F6",
        "#F59E0B",
    ],
    text_auto=",.0f",
    title=f"Monthly Revenue Forecast — {selected_method}",
    labels={
        "ForecastMonth": "Forecast Month",
        selected_column: "Predicted Revenue",
    },
)

    fig_monthly_revenue.update_layout(
    showlegend=False,
    height=430,
    bargap=0.4,
    yaxis=dict(rangemode="tozero"),
)

    fig_monthly_revenue.update_traces(
    textposition="outside",
    cliponaxis=False,
    hovertemplate=(
        "Month: %{x}"
        "<br>Predicted Revenue: %{y:,.2f}"
        "<extra></extra>"
    ),
)

    st.plotly_chart(
    fig_monthly_revenue,
    use_container_width=True,
    key="monthly_revenue_forecast_chart",
)

    category_totals = (
        forecast.groupby(
            "CourseCategory",
            as_index=False
        )[selected_column]
        .sum()
        .sort_values(
            selected_column,
            ascending=False
        )
    )

    st.subheader("Revenue by Category")

    fig = px.bar(
    category_totals,
    x="CourseCategory",
    y=selected_column,
    color="CourseCategory",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title=f"Revenue by Category — {selected_method}",
    labels={
        "CourseCategory": "Course Category",
        selected_column: "Predicted Revenue",
    },
    text_auto=",.0f",
)

    fig.update_layout(
    showlegend=False,
    xaxis_tickangle=-35,
    height=500,
    margin=dict(t=60, b=100),
)

    fig.update_traces(
    textposition="outside",
    cliponaxis=False,
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="revenue_category_chart",
)

    st.subheader("Revenue Forecast Comparison")

    comparison = (
        forecast.groupby("Month", as_index=False)
        .agg(
            PriceBasedRevenue=(
                "PredictedRevenuePriceBased", "sum"
            ),
            RidgeRevenue=(
                "PredictedRevenueRidge", "sum"
            ),
            HybridRevenue=(
                "PredictedRevenueHybrid", "sum"
            ),
        )
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The three revenue methods are alternative estimates. "
        "They must not be added together."
    )


# =========================================================
# 8. CATEGORY COMPARISON
# =========================================================

elif page == "Category Comparison":

    st.header("Course Category Performance")

    st.caption(
        "Comparison of historical performance "
        "with January–March 2026 forecasts."
    )

    selected_metric = st.selectbox(
        "Select comparison metric",
        ["Enrollments", "Revenue"],
    )

    historical = (
        category_history.groupby(
            "CourseCategory",
            as_index=False
        )[selected_metric]
        .sum()
    )

    if selected_metric == "Enrollments":

        forecast_category = (
            forecast.groupby(
                "CourseCategory",
                as_index=False
            )["PredictedEnrollments"]
            .sum()
            .rename(
                columns={
                    "PredictedEnrollments": "Forecast"
                }
            )
        )

        historical = historical.rename(
            columns={
                "Enrollments": "Historical"
            }
        )

    else:

        selected_method = st.radio(
            "Forecast revenue method",
            list(revenue_methods.keys()),
            horizontal=True,
        )

        selected_column = revenue_methods[
            selected_method
        ]

        forecast_category = (
            forecast.groupby(
                "CourseCategory",
                as_index=False
            )[selected_column]
            .sum()
            .rename(
                columns={
                    selected_column: "Forecast"
                }
            )
        )

        historical = historical.rename(
            columns={
                "Revenue": "Historical"
            }
        )

    combined = historical.merge(
        forecast_category,
        on="CourseCategory",
        how="inner",
        validate="one_to_one"
    )

    combined = combined.sort_values(
        "Historical",
        ascending=False
    )

    st.subheader("Historical vs Forecast")

    st.info(
    "Historical figures cover January–December 2025 (12 months). "
    "Forecast figures cover January–March 2026 (3 months). "
    "The periods have different durations, so their totals "
    "should not be interpreted as a like-for-like comparison."
)

    fig = px.bar(
    combined,
    x="CourseCategory",
    y=["Historical", "Forecast"],
    barmode="group",
    color_discrete_map={
        "Historical": "#F59E0B",
        "Forecast": "#2563EB",
    },
    title=(
    f"Historical 2025 (12 Months) vs "
    f"Forecast Jan–Mar 2026 (3 Months) — {selected_metric}"
),

    labels={
        "CourseCategory": "Course Category",
        "value": selected_metric,
        "variable": "Period",
    },
)

    fig.update_layout(
    xaxis_tickangle=-35,
    height=520,
    legend_title_text="Period",
    margin=dict(t=60, b=100),
)

    fig.update_traces(
    hovertemplate="%{x}<br>Value: %{y:,.2f}<extra>%{fullData.name}</extra>"
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="category_comparison_chart",
)

    st.dataframe(
        combined,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Category Comparison",
        data=combined.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="edupro_category_comparison.csv",
        mime="text/csv",
    )


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.header("Chronological Model Evaluation")

    st.caption(
        "Original internship model evaluation results."
    )

    st.subheader("Validation and Test Metrics")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Final Test Results")

    test = metrics[
        metrics["Split"].str.lower() == "test"
    ].copy()

    st.dataframe(
        test,
        use_container_width=True,
        hide_index=True
    )

    enrollment_test = test[
        test["Target"] == "Enrollments"
    ]

    revenue_test = test[
        test["Target"] == "Revenue"
    ]

    if not enrollment_test.empty:

        enrollment_r2 = float(
            enrollment_test.iloc[0]["R2"]
        )

        if enrollment_r2 < 0:

            st.warning(
                f"The enrollment model has negative test R² "
                f"({enrollment_r2:.4f}), indicating limited "
                "predictive performance on the held-out test data."
            )

    if not revenue_test.empty:

        revenue_r2 = float(
            revenue_test.iloc[0]["R2"]
        )

        st.info(
            f"The revenue model achieved test R² of "
            f"{revenue_r2:.4f} on the held-out test period."
        )

    st.caption(
        "The test period covers November and December 2025. "
        "The dataset contains only one year of historical data, "
        "so results should be interpreted cautiously."
    )

    st.download_button(
        "Download Model Evaluation CSV",
        data=metrics.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="edupro_model_evaluation.csv",
        mime="text/csv",
    )


# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.header("Feature Importance Explorer")

    target = st.selectbox(
        "Select prediction target",
        ["Enrollments", "Revenue"],
    )

    importance = (
        enrollment_importance.copy()
        if target == "Enrollments"
        else revenue_importance.copy()
    )

    # Sort by importance.
    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    # Remove preprocessing prefixes for display.
    importance["DisplayFeature"] = (
        importance["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    # CourseID-specific features are not included
    # in these public deployment reports.
    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=min(39, len(importance)),
        value=min(15, len(importance)),
    )

    displayed = importance.head(top_n)

    st.subheader(
        f"Top {top_n} Features — {target}"
    )

    chart_data = (
        displayed.sort_values("Importance")
        .set_index("DisplayFeature")["Importance"]
    )

    feature_color = (
    "#8B5CF6"
    if target == "Enrollments"
    else "#16A34A"
)

    fig = px.bar(
    displayed.sort_values("Importance"),
    x="Importance",
    y="DisplayFeature",
    orientation="h",
    title=f"Top {top_n} Important Features — {target}",
    labels={
        "Importance": "Feature Importance Score",
        "DisplayFeature": "Feature",
    },
    text="Importance",
)

    fig.update_traces(
    marker_color=feature_color,
    texttemplate="%{text:,.2f}",
    textposition="outside",
    cliponaxis=False,
)

    fig.update_layout(
    height=max(450, top_n * 32),
    showlegend=False,
    margin=dict(l=170, r=80, t=60, b=50),
    yaxis_title=None,
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="feature_importance_chart",
)

    st.dataframe(
        displayed[
            ["DisplayFeature", "Importance"]
        ].rename(
            columns={
                "DisplayFeature": "Feature"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These scores come from the original trained models. "
        "CourseID-specific features have been excluded from "
        "the public reports. Feature importance measures "
        "model reliance, not causal influence."
    )

    st.download_button(
        "Download Public Feature Importance",
        data=importance[
            ["DisplayFeature", "Importance"]
        ].to_csv(index=False).encode("utf-8"),
        file_name=(
            f"edupro_feature_importance_"
            f"{target.lower()}.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# 11. SCENARIO PREDICTOR
# =========================================================

elif page == "Scenario Predictor":

    st.header("Course Planning Scenario Simulator")

    st.warning(
        "This is an assumption-based planning simulator. "
        "It does not execute the original trained course-level "
        "machine-learning models."
    )

    st.write(
        "Select a category and explore how changes in "
        "expected demand and revenue per enrollment affect "
        "the category-level forecast."
    )

    categories = sorted(
        forecast["CourseCategory"].dropna().unique()
    )

    selected_category = st.selectbox(
        "Select course category",
        categories,
    )

    selected_method = st.selectbox(
        "Baseline revenue method",
        list(revenue_methods.keys()),
        key="scenario_revenue_method",
    )

    selected_column = revenue_methods[
        selected_method
    ]

    baseline = forecast[
        forecast["CourseCategory"] == selected_category
    ]

    baseline_enrollments = float(
        baseline["PredictedEnrollments"].sum()
    )

    baseline_revenue = float(
        baseline[selected_column].sum()
    )

    st.subheader("Baseline Forecast")

    col1, col2 = st.columns(2)

    col1.metric(
        "Baseline Enrollments",
        f"{baseline_enrollments:,.0f}"
    )

    col2.metric(
        "Baseline Revenue",
        f"{baseline_revenue:,.2f}"
    )

    st.divider()

    st.subheader("Adjust Scenario Assumptions")

    demand_change = st.slider(
        "Expected demand change (%)",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
    )

    revenue_change = st.slider(
        "Revenue per enrollment change (%)",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
    )

    demand_factor = 1 + demand_change / 100
    revenue_factor = 1 + revenue_change / 100

    scenario_enrollments = (
        baseline_enrollments * demand_factor
    )

    scenario_revenue = (
        baseline_revenue
        * demand_factor
        * revenue_factor
    )

    st.divider()

    st.subheader("Scenario Results")

    col1, col2 = st.columns(2)

    col1.metric(
        "Scenario Enrollments",
        f"{scenario_enrollments:,.0f}",
        delta=f"{demand_change:+d}%"
    )

    col2.metric(
        "Scenario Revenue",
        f"{scenario_revenue:,.2f}",
        delta=f"{revenue_change:+d}% revenue per enrollment"
    )

    comparison = pd.DataFrame({
        "Metric": [
            "Enrollments",
            "Revenue",
        ],
        "Baseline": [
            baseline_enrollments,
            baseline_revenue,
        ],
        "Scenario": [
            scenario_enrollments,
            scenario_revenue,
        ],
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )
    st.subheader("Baseline vs Scenario Comparison")

    fig = px.bar(
    comparison.melt(
        id_vars="Metric",
        value_vars=["Baseline", "Scenario"],
        var_name="Forecast Type",
        value_name="Value",
    ),
    x="Forecast Type",
    y="Value",
    color="Forecast Type",
    facet_col="Metric",
    color_discrete_map={
        "Baseline": "#2563EB",
        "Scenario": "#F59E0B",
    },
    barmode="group",
    text_auto=",.0f",
    title="Impact of Scenario Adjustments",
)

    fig.update_yaxes(matches=None, showticklabels=True)

    fig.update_traces(
    textposition="outside",
    cliponaxis=False,
)

    fig.update_layout(
    height=450,
    showlegend=True,
    margin=dict(t=80, b=50),
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="scenario_comparison_chart",
)

    st.info(
        "Calculation methodology: scenario enrollments equal "
        "baseline enrollments multiplied by the demand adjustment. "
        "Scenario revenue equals baseline revenue multiplied "
        "by both the demand adjustment and revenue-per-enrollment "
        "adjustment. These are illustrative assumptions, "
        "not statistically validated predictions."
    )

    st.download_button(
        "Download Scenario Results",
        data=comparison.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="edupro_scenario_results.csv",
        mime="text/csv",
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "EduPro | Predictive Modeling for Course Demand "
    "and Revenue Forecasting | Unified Mentor Internship"
)

st.caption(
    "Historical period: January–December 2025 | "
    "Forecast period: January–March 2026 | "
    "Forecasts are estimates, not guaranteed outcomes."
)
