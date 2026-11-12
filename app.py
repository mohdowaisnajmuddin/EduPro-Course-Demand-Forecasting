
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_processing import load_source_data
from src.modeling import (
    train_project,
    feature_importance,
    make_scenario_row,
    predict_scenario,
)

st.set_page_config(
    page_title="EduPro Forecasting",
    page_icon="📚",
    layout="wide",
)

# Consistent chart styling
PALETTE = px.colors.qualitative.Bold
CHART_TEMPLATE = "plotly_dark"


def style_chart(fig, *, x_angle=0, height=460):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=35, r=25, t=70, b=85),
        legend_title_text="Category",
        xaxis_title_standoff=12,
        yaxis_title_standoff=12,
    )
    fig.update_xaxes(tickangle=x_angle, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


@st.cache_resource(
    show_spinner="Training and evaluating the forecasting models..."
)
def get_bundle():
    return train_project()


@st.cache_data
def get_sources():
    return load_source_data()


st.title("📚 EduPro | Course Demand & Revenue Forecasting")

st.caption(
    "Historical analytics, machine-learning forecasts "
    "and course planning scenarios"
)

try:
    bundle = get_bundle()
    courses, teachers, transactions = get_sources()

except Exception as exc:
    st.error(f"Could not load or train the project: {exc}")
    st.info(
        "Check that the CSV files are present in data/raw "
        "and install requirements.txt."
    )
    st.stop()


panel = bundle["panel"]
forecast = bundle["forecast"]
metrics = bundle["metrics"]

historical_revenue = transactions["Amount"].sum()
total_enrollments = len(transactions)

# Dashboard summary metrics
forecast_revenue_total = forecast[
    "PredictedRevenuePriceBased"
].sum()

forecast_ridge_total = forecast[
    "PredictedRevenueRidge"
].sum()
forecast_enrollment_total = forecast["PredictedEnrollments"].sum()

c1, c2, c3 = st.columns(3)

c1.metric(
    "Historical enrollments",
    f"{total_enrollments:,}"
)
c2.metric(
    "Historical revenue",
    f"{historical_revenue:,.2f}"
)
c3.metric(
    "Total courses",
    f"{courses['CourseID'].nunique():,}"
)

c4, c5, c6 = st.columns(3)

c4.metric(
    "Course categories",
    f"{courses['CourseCategory'].nunique():,}"
)
c5.metric(
    "Forecast enrollments (3 months)",
    f"{forecast_enrollment_total:,.0f}"
)
c6.metric(
    "Price-based revenue (3 months)",
    f"{forecast_revenue_total:,.2f}"
)

st.caption(
    f"Ridge revenue forecast (3 months): "
    f"{forecast_ridge_total:,.2f}"
)
st.caption(
    "Forecast figures refer to January–March 2026, "
    "based on the dataset ending in December 2025."
)


# Navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Demand forecast",
        "Revenue forecast",
        "Category comparison",
        "Model evaluation",
        "Feature importance",
        "Scenario predictor",
    ],
)

# --------------------------------------------------
# GLOBAL DASHBOARD FILTERS
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Dashboard filters")

category_options = sorted(
    courses["CourseCategory"].dropna().unique().tolist()
)

type_options = sorted(
    courses["CourseType"].dropna().unique().tolist()
)

level_options = sorted(
    courses["CourseLevel"].dropna().unique().tolist()
)

selected_categories = st.sidebar.multiselect(
    "Course categories",
    category_options,
    default=category_options,
)

selected_types = st.sidebar.multiselect(
    "Course types",
    type_options,
    default=type_options,
)

selected_levels = st.sidebar.multiselect(
    "Course levels",
    level_options,
    default=level_options,
)

filtered_courses = courses[
    courses["CourseCategory"].isin(selected_categories)
    & courses["CourseType"].isin(selected_types)
    & courses["CourseLevel"].isin(selected_levels)
].copy()

filtered_course_ids = filtered_courses["CourseID"].unique()

filtered_panel = panel[
    panel["CourseID"].isin(filtered_course_ids)
].copy()

filtered_forecast = forecast[
    forecast["CourseID"].isin(filtered_course_ids)
].copy()

filtered_transactions = transactions[
    transactions["CourseID"].isin(filtered_course_ids)
].copy()


# --------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Historical performance")

    monthly = transactions.assign(
        Month=transactions.TransactionDate.dt.to_period("M").dt.to_timestamp()
    ).groupby("Month").agg(
        Enrollments=("TransactionID", "count"),
        Revenue=("Amount", "sum"),
    ).reset_index()

    left, right = st.columns(2)

    with left:
        fig = px.line(
            monthly,
            x="Month",
            y="Enrollments",
            markers=True,
            title="Monthly enrollments",
            color_discrete_sequence=[PALETTE[0]],
        )

        st.plotly_chart(
            style_chart(fig, height=400),
            use_container_width=True,
        )

    with right:
        fig = px.line(
            monthly,
            x="Month",
            y="Revenue",
            markers=True,
            title="Monthly revenue",
            color_discrete_sequence=[PALETTE[1]],
        )

        st.plotly_chart(
            style_chart(fig, height=400),
            use_container_width=True,
        )

    course_totals = panel.groupby(
        ["CourseID", "CourseName"]
    ).agg(
        Enrollments=("Enrollments", "sum"),
        Revenue=("Revenue", "sum"),
    ).reset_index()

    a, b = st.columns(2)

    with a:
        top_enrollment = (
            course_totals
            .nlargest(10, "Enrollments")
            .sort_values("Enrollments")
        )

        fig = px.bar(
            top_enrollment,
            x="CourseName",
            y="Enrollments",
            color="Enrollments",
            color_continuous_scale="Tealgrn",
            title="Top 10 courses by enrollment",
        )

        st.plotly_chart(
            style_chart(fig, x_angle=35, height=520),
            use_container_width=True,
        )

    with b:
        top_revenue = (
            course_totals
            .nlargest(10, "Revenue")
            .sort_values("Revenue")
        )

        fig = px.bar(
            top_revenue,
            x="CourseName",
            y="Revenue",
            color="Revenue",
            color_continuous_scale="Sunset",
            title="Top 10 courses by revenue",
        )

        st.plotly_chart(
            style_chart(fig, x_angle=35, height=520),
            use_container_width=True,
        )

    st.caption(
        "Revenue uses the Amount field from transactions. "
        "Free-course enrollments remain included in demand."
    )


# --------------------------------------------------
# 2. DEMAND FORECAST
# --------------------------------------------------

elif page == "Demand forecast":

    st.subheader(
        "First three months after the dataset cutoff: "
        "predicted course demand"
    )

    monthly_demand = filtered_forecast.groupby(
    "Month", as_index=False
)["PredictedEnrollments"].sum()
    fig = px.line(
        monthly_demand,
        x="Month",
        y="PredictedEnrollments",
        markers=True,
        title="Forecast total enrollments",
        color_discrete_sequence=[PALETTE[0]],
    )

    st.plotly_chart(
        style_chart(fig, height=420),
        use_container_width=True,
    )

    st.dataframe(
        forecast.sort_values(
            ["Month", "PredictedEnrollments"],
            ascending=[True, False],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
    label="Download demand forecast CSV",
    data=filtered_forecast.to_csv(index=False).encode("utf-8"),
    file_name="edupro_demand_forecast.csv",
    mime="text/csv",
)

    top = (
    filtered_forecast.groupby(
        ["CourseName", "CourseCategory"],
        as_index=False,
    )["PredictedEnrollments"]
    .sum()
    .nlargest(10, "PredictedEnrollments")
    .sort_values("PredictedEnrollments")
)

    fig = px.bar(
        top,
        x="CourseName",
        y="PredictedEnrollments",
        color="CourseCategory",
        barmode="group",
        color_discrete_sequence=PALETTE,
        title="Top forecast-demand courses (three-month total)",
    )

    st.plotly_chart(
        style_chart(fig, x_angle=35, height=560),
        use_container_width=True,
    )


# --------------------------------------------------
# 3. REVENUE FORECAST
# --------------------------------------------------


elif page == "Revenue forecast":

    st.subheader("Revenue forecast: January–March 2026")

    st.caption(
        "Compare direct Ridge Regression revenue predictions "
        "with predicted enrollments multiplied by course price."
    )


    monthly_forecast = (
        filtered_forecast.groupby("Month", as_index=False)
        .agg(
            PriceBasedRevenue=("PredictedRevenuePriceBased", "sum"),
            RidgeRevenue=("PredictedRevenueRidge", "sum"),
            HybridRevenue=("PredictedRevenueHybrid", "sum"),
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Price-based revenue (3 months)",
        f"{monthly_forecast['PriceBasedRevenue'].sum():,.2f}",
    )
    c2.metric(
        "Ridge revenue (3 months)",
        f"{monthly_forecast['RidgeRevenue'].sum():,.2f}",
    )
    c3.metric(
        "Hybrid revenue (3 months)",
        f"{monthly_forecast['HybridRevenue'].sum():,.2f}",
    )

    comparison = monthly_forecast.melt(
        id_vars="Month",
        value_vars=[
            "PriceBasedRevenue",
            "RidgeRevenue",
            "HybridRevenue",
        ],
        var_name="Method",
        value_name="Revenue",
    )

    comparison["Method"] = comparison["Method"].replace({
        "PriceBasedRevenue": "Price × enrollments",
        "RidgeRevenue": "Ridge Regression",
        "HybridRevenue": "Hybrid (Free=0, Paid=Ridge)",
    })



    fig = px.bar(
        comparison,
        x="Month",
        y="Revenue",
        color="Method",
        barmode="group",
        color_discrete_sequence=PALETTE,
        title="Monthly revenue forecast comparison"
    )

    st.plotly_chart(
        style_chart(fig, height=420),
        use_container_width=True
    )


    revenue_method = st.radio(
        "Select revenue method for course ranking",
        [
            "Price-based",
            "Ridge Regression",
            "Hybrid",
        ],
        horizontal=True,
        key="revenue_method_selector",
    )

    revenue_columns = {
        "Price-based": "PredictedRevenuePriceBased",
        "Ridge Regression": "PredictedRevenueRidge",
        "Hybrid": "PredictedRevenueHybrid",
    }

    revenue_column = revenue_columns[revenue_method]



    course_rev = (
        filtered_forecast.groupby(
            ["CourseName", "CourseCategory"],
            as_index=False
        )[revenue_column]
        .sum()
        .nlargest(15, revenue_column)
        .sort_values(revenue_column)
    )

    fig = px.bar(
        course_rev,
        x="CourseName",
        y=revenue_column,
        color="CourseCategory",
        color_discrete_sequence=PALETTE,
        title=f"Top 15 courses: {revenue_method} revenue"
    )

    st.plotly_chart(
        style_chart(fig, x_angle=35, height=620),
        use_container_width=True
    )

    st.dataframe(
        monthly_forecast,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download both revenue forecasts",
        data=filtered_forecast.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="edupro_both_revenue_forecasts.csv",
        mime="text/csv"
    )



# --------------------------------------------------
# 4. CATEGORY COMPARISON
# --------------------------------------------------

elif page == "Category comparison":

    st.subheader("Course category performance")

    st.info(
    "Revenue forecasts are available using two methods: "
    "Ridge Regression and predicted enrollments multiplied "
    "by course price. These are alternative estimates, "
    "not amounts to add together."
)
    historical = filtered_panel.groupby(
    "CourseCategory",
    as_index=False,
).agg(
    Enrollments=("Enrollments", "sum"),
    Revenue=("Revenue", "sum"),
)

    predicted = filtered_forecast.groupby(
    "CourseCategory",
    as_index=False
).agg(
    PredictedEnrollments=("PredictedEnrollments", "sum"),
    PredictedRevenue=("PredictedRevenue", "sum"),
    PredictedRevenuePriceBased=(
        "PredictedRevenuePriceBased", "sum"
    ),
    PredictedRevenueRidge=(
        "PredictedRevenueRidge", "sum"
    )
)

    combined = historical.merge(
    predicted,
    on="CourseCategory",
    how="outer",
).fillna(0)



    metric = st.selectbox(
        "Compare using",
        ["Revenue", "Enrollments"],
    )

    fig = px.bar(
        combined.sort_values(metric, ascending=False),
        x="CourseCategory",
        y=metric,
        color="CourseCategory",
        color_discrete_sequence=PALETTE,
        title=f"Historical {metric.lower()} by category",
    )

    st.plotly_chart(
        style_chart(fig, x_angle=35, height=520),
        use_container_width=True,
    )

    if metric == "Revenue":
        revenue_choice = st.radio(
            "Forecast revenue method",
            ["Price-based", "Ridge Regression"],
            horizontal=True,
            key="category_revenue_method"
        )

        pred_col = (
            "PredictedRevenuePriceBased"
            if revenue_choice == "Price-based"
            else "PredictedRevenueRidge"
        )
    else:
        pred_col = "PredictedEnrollments"



    fig = px.bar(
        combined.sort_values(pred_col, ascending=False),
        x="CourseCategory",
        y=pred_col,
        color="CourseCategory",
        color_discrete_sequence=PALETTE,
        title=f"Forecast {metric.lower()} by category",
    )

    st.plotly_chart(
        style_chart(fig, x_angle=35, height=520),
        use_container_width=True,
    )

    st.dataframe(
        combined.sort_values(metric, ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 5. MODEL EVALUATION
# --------------------------------------------------

elif page == "Model evaluation":

    st.subheader("Chronological model evaluation")

    st.write(
        "The dataset is split chronologically: January–August 2025 "
        "for training, September–October for validation, and "
        "November–December for final testing."
    )

    split = st.radio(
        "Choose evaluation period",
        ["Validation", "Test"],
        horizontal=True,
    )

    split_metrics = metrics[
        metrics["Split"].astype(str).str.lower() == split.lower()
    ].copy()

    if split_metrics.empty:
        st.warning(f"No {split.lower()} metrics are available.")
    else:
        st.dataframe(
            split_metrics.round(3),
            use_container_width=True,
            hide_index=True,
        )

        for target in ["Enrollments", "Revenue"]:
            target_metrics = split_metrics[
                split_metrics["Target"] == target
            ]

            if target_metrics.empty:
                st.info(f"No {target.lower()} metrics for this split.")
                continue

            for metric_name in ["RMSE", "MAE"]:
                fig = px.bar(
                    target_metrics.sort_values(metric_name),
                    x="Model",
                    y=metric_name,
                    color="Model",
                    color_discrete_sequence=PALETTE,
                    title=(
                        f"{target}: {metric_name} comparison "
                        f"on {split.lower()} data "
                        "(lower is better)"
                    ),
                )

                st.plotly_chart(
                    style_chart(fig, x_angle=30, height=480),
                    use_container_width=True,
                )

    st.info(
        "Enrollment validation results: the Global Mean Baseline "
        "performed better than the machine-learning candidates. "
        "Lasso was the best enrollment machine-learning candidate "
        "by validation RMSE."
    )

    st.info(
        "Revenue validation results: Ridge Regression was the best "
        "revenue machine-learning candidate by validation RMSE."
    )

    st.caption(
        "The final test period contains only November and December "
        "2025. Interpret the results cautiously because the dataset "
        "contains only one year of history."
    )

# --------------------------------------------------
# 6. FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature importance":

    target = st.selectbox(
        "Prediction target",
        ["Enrollments", "Revenue"],
    )

    importance = feature_importance(
        bundle,
        target,
    ).head(20)

    if importance.empty:

        st.warning(
            "Feature importance is not available "
            "for the selected estimator."
        )

    else:

        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Turbo",
            title=f"Important features for {target} prediction",
        )

        st.plotly_chart(
            style_chart(fig, height=760),
            use_container_width=True,
        )

        st.dataframe(
            importance,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# 7. SCENARIO PREDICTOR
# --------------------------------------------------

elif page == "Scenario predictor":

    st.subheader("Test a course-planning scenario")

    st.write(
        "Choose an existing course, then adjust its price, "
        "duration and instructor profile. The scenario predicts "
        "the first month after the supplied dataset cutoff "
        "(December 2025), not the current calendar month."
    )

    course_options = courses.drop_duplicates("CourseID").copy()

    course_options["DisplayName"] = (
        course_options["CourseName"].astype(str)
        + " ("
        + course_options["CourseID"].astype(str)
        + ")"
    )

    selected_course_id = st.selectbox(
        "Select Course",
        course_options["CourseID"].tolist(),
        format_func=lambda course_id: course_options.loc[
            course_options["CourseID"] == course_id,
            "DisplayName",
        ].iloc[0],
        key="scenario_course_selector",
    )

    course = course_options[
        course_options["CourseID"] == selected_course_id
    ].iloc[0]



    teacher_rows = transactions[
            transactions.CourseID == course.CourseID
        ].merge(
            teachers,
            on="TeacherID",
            how="left",
        )

    exp_default = teacher_rows["YearsOfExperience"].mean()
    rating_default = teacher_rows["TeacherRating"].mean()

    exp_default = (
        0.0 if pd.isna(exp_default) else float(exp_default)
    )

    rating_default = (
        0.0 if pd.isna(rating_default) else float(rating_default)
    )

    with st.form("scenario_form"):

            price = st.number_input(
                "Course price",
                min_value=0.0,
                value=float(course.CoursePrice),
                step=5.0,
            )

            duration = st.number_input(
                "Course duration",
                min_value=0.1,
                value=float(course.CourseDuration),
                step=1.0,
            )

            level = st.selectbox(
                "Course level",
                ["Beginner", "Intermediate", "Advanced"],
                index=[
                    "Beginner",
                    "Intermediate",
                    "Advanced",
                ].index(course.CourseLevel),
            )

            exp = st.number_input(
                "Instructor experience (years)",
                min_value=0.0,
                value=exp_default,
                step=1.0,
            )

            rating = st.slider(
                "Instructor rating",
                min_value=0.0,
                max_value=5.0,
                value=min(5.0, max(0.0, rating_default)),
                step=0.1,
            )

            submitted = st.form_submit_button(
                "Predict first month after dataset cutoff"
            )

    if submitted:
        scenario = make_scenario_row(
            bundle,
            course["CourseID"],
            price,
            duration,
            level,
            exp,
            rating,
        )

        prediction = predict_scenario(bundle, scenario)


        x, y, z, h = st.columns(4)

        x.metric(
            "Predicted enrollments",
            f"{prediction['Enrollments']:.1f}",
        )

        y.metric(
            "Price-based revenue",
            f"{prediction['RevenuePriceBased']:,.2f}",
        )

        z.metric(
            "Ridge revenue",
            f"{prediction['RevenueRidge']:,.2f}",
        )

        h.metric(
            "Hybrid revenue",
            f"{prediction['RevenueHybrid']:,.2f}",
        )
