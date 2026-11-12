# Executive Summary: EduPro Course Demand and Revenue Forecasting

## Purpose

Provide EduPro decision-makers with forward-looking estimates of course enrollments and revenue to support course roadmap, pricing and instructor-capacity discussions.

## Source data

The modelling package contains 60 courses, 60 teachers and 10,000 transactions dated January–December 2025. Recorded transaction revenue totals 911,323.47 in the dataset's unspecified currency. Approximately 64.03% of transactions have a zero amount. These transactions are retained as enrollments but contribute no revenue.

## What the solution delivers

- Monthly course-level enrollment and revenue analysis.
- Course, instructor-summary, calendar and lagged-performance features.
- Five machine-learning candidates: Linear Regression, Ridge Regression, Lasso Regression, Random Forest and Gradient Boosting.
- Baseline comparisons for enrollment forecasting.
- MAE, RMSE and R² model evaluation.
- Three-month recursive enrollment forecasts by course and category.
- Three revenue forecasting methods: price-based, Ridge Regression, and hybrid.
- A Streamlit dashboard for historical trends, forecasts, category comparisons, feature importance and planning scenarios.

## Modelling and evaluation

The data is divided chronologically into eight training months (January–August 2025), two validation months (September–October 2025) and two test months (November–December 2025).

Lasso Regression was selected as the best machine-learning model for enrollment forecasting. Ridge Regression was selected for the revenue modelling task.

The global-mean baseline performed best among the enrollment validation methods. This indicates that the enrollment machine-learning model has not demonstrated a reliable advantage over a simple baseline.

On the final test period:
- Enrollment Lasso Regression: MAE 3.104, RMSE 4.046, R² -0.211.
- Revenue Ridge Regression: MAE 250.586, RMSE 577.369, R² 0.9316.

## Forecast results

The project generates forecasts for January–March 2026, based on the supplied transaction history ending on 30 December 2025.

The latest forecast contains:
- 180 course-month records (60 courses across three months).
- 3,151 predicted enrollments.
- 286,685.77 price-based revenue.
- 259,441.54 Ridge revenue.
- 237,115.24 hybrid revenue.

Price-based revenue is predicted enrollments multiplied by course price. Ridge revenue uses the trained revenue model. Hybrid revenue uses Ridge predictions for paid courses and zero revenue for free courses.

## How to use the forecasts

Review demand and revenue together. High-enrollment free courses should not be labelled commercial failures solely because their recorded revenue is zero. Use category and course forecasts as planning signals, then validate major pricing or launch decisions with controlled experiments and business-owner review.

## Caveats

The source contains only one year of history and does not include marketing spend, discounts, refunds, course launch dates or availability. Enrollment test performance is weak, with a negative R². Forecasts are exploratory, not guaranteed outcomes. Scenario predictions are not causal estimates of price elasticity.

The January-March 2026 forecast is based on the historical dataset cutoff, not on current operational data as of November 2026.

## Recommended next actions

1. Extend the history to at least 24–36 months.
2. Add discounts, marketing campaigns, refunds, course availability and launch dates.
3. Establish repeated rolling-origin backtests before operational use.
4. Monitor actual versus predicted demand and revenue each month.
5. Use scenario outputs as decision support, with human review.

## Exploratory hybrid revenue evaluation

The November-December 2025 test comparison produced:

- Hybrid: MAE 235.250, RMSE 576.332, R² 0.9319.
- Ridge Regression: MAE 250.586, RMSE 577.369, R² 0.9316.

The hybrid method assigns zero predicted revenue to free courses and
uses Ridge predictions for paid courses. Historical checks confirmed
456 free-course monthly records with zero revenue.

Because the hybrid rule was adopted after inspecting test behavior,
its apparent improvement is exploratory and requires independent
future validation.
