# EduPro: Predictive Modeling for Course Demand and Revenue Forecasting

A reproducible Data Science project using EduPro course, teacher, and transaction data. It builds a monthly course-level dataset, compares regression models and baseline methods, generates a three-month forecast, and provides an interactive Streamlit dashboard.

## Project Contents

```text
EduPro_Course_Demand_Forecasting/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── courses.csv
│       ├── teachers.csv
│       ├── transactions.csv
├── src/
│   ├── data_processing.py
│   └── modeling.py
└── reports/
    ├── source_data_profile.json
    ├── historical_monthly_summary.csv
    ├── model_evaluation.csv
    ├── forecast_next_3_months.csv
    ├── executive_summary.md
    └── research_paper.md
```


## Run Locally

1. Install Python 3.10 or newer.
2. Extract the project ZIP.
3. Open a terminal in the project folder.
4. Create and activate a virtual environment (recommended).
5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Start the dashboard:

```bash
streamlit run app.py
```

The dashboard trains and evaluates the models when the application runs. The first run may take a little longer.

## Data Availability

The datasets used in this project were provided by Unified Mentor
for an internship assignment.

The original CSV datasets are not included in the public repository
because permission for redistribution has not been established.

To run the project locally, authorized users must place these files
inside `data/raw/`:

- `courses.csv`
- `teachers.csv`
- `transactions.csv`

Certain course-level forecast and prediction outputs are also
excluded from the public repository.

The repository documents the forecasting methodology and provides
the application source code. Access to the original datasets must
be obtained through an authorized source.

## Dataset and Targets

The supplied data contains:

- 60 courses.
- 60 teachers.
- 10,000 transactions.
- Transaction dates from January 1 to December 30, 2025.
- 12 course categories.

The modeling targets are:

- **Enrollment demand:** Number of transaction rows per course and month, including zero-amount and free transactions.
- **Historical revenue:** Sum of transaction `Amount` per course and month.
- **Category revenue:** Sum of course-level revenue within each category.

The historical dataset contains 10,000 transaction rows and total transaction revenue of 911,323.47 in the source currency. Approximately 64.03% of transactions have a zero amount. These are retained as enrollments but contribute zero revenue.

## Data Processing and Feature Engineering

The pipeline:

- Aggregates transactions into a complete course-by-month panel.
- Creates a 60-course by 12-month dataset, containing 720 course-month records.
- Joins course metadata and teacher information.
- Aggregates teacher experience and ratings using distinct course-teacher assignments rather than transaction frequency.
- Calculates an expertise-match score by comparing teacher expertise with course category.
- Creates calendar, seasonal, lag, and rolling-three-month features.
- Adds course price, duration, rating, and teacher-related features.
- Prepares numerical and categorical features for model training.

## Modeling Method

The project uses a chronological split to preserve the time order of the data:

| Dataset | Period | Months |
|---|---|---:|
| Training | January–August 2025 | 8 |
| Validation | September–October 2025 | 2 |
| Testing | November–December 2025 | 2 |

Five machine-learning algorithms are compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting

For enrollment forecasting, the machine-learning models are also compared against four baselines:

- Global Mean Baseline
- Course Mean Baseline
- Previous Month Baseline
- Previous 3-Month Average

Model selection is based on validation RMSE, not test performance.

The best enrollment machine-learning candidate was Lasso Regression. However, the Global Mean Baseline performed better overall on enrollment validation data.

Ridge Regression was selected as the best revenue machine-learning candidate based on validation RMSE.

The selected models are evaluated on the final two-month test period after fitting on the combined training and validation data.

## Model Evaluation Results

| Target | Selected model | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| Enrollments | Lasso Regression | 3.104 | 4.046 | -0.211 |
| Revenue | Ridge Regression | 250.586 | 577.369 | 0.9316 |

Interpretation:

- The enrollment model's negative test R² indicates that its predictions did not outperform a simple test-set mean benchmark on that metric. The Global Mean Baseline also performed better during validation.
- The revenue model achieved a promising test R² of approximately 0.9316.
- The test period contains only two months, so these results should be interpreted cautiously.

## Three-Month Forecast

The forecast covers January-March 2026, based on historical
transaction data ending in December 2025.

It contains 180 course-month records: 60 courses across 3 months.

| Forecast metric | Total |
|---|---:|
| Predicted enrollments | 3,151 |
| Price-based revenue | 286,685.77 |
| Ridge revenue | 259,441.54 |
| Hybrid revenue | 237,115.24 |

### Revenue Forecasting Methods

The project provides three revenue estimates:

1. **Price-based:** Predicted enrollments multiplied by course price.
2. **Ridge:** Direct predictions from the trained revenue model.
3. **Hybrid:** Ridge predictions for paid courses and zero
   predicted revenue for free courses.

### Exploratory Revenue Comparison

The November-December 2025 test comparison produced:

| Method | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Hybrid | 235.250 | 576.332 | 0.9319 |
| Ridge | 250.586 | 577.369 | 0.9316 |
| Price-based | 280.817 | 684.235 | 0.9040 |
| Training Mean Baseline | 1734.196 | 2208.638 | -0.0003 |

The hybrid method enforces zero predicted revenue for free
courses. Historical checks found 456 free-course monthly
records with zero actual revenue.

**Important limitation:** The hybrid rule was adopted after
inspecting test behavior. Its improvement is exploratory
and requires independent future validation.

These forecasts reflect the December 2025 dataset cutoff.
They are not current operational forecasts as of November 2026.

Forecast outputs are saved in `reports/`, including:

- `forecast_next_3_months.csv`
- `historical_monthly_summary.csv`
- `model_evaluation.csv`
- `revenue_method_comparison.csv`

## Dashboard Features

The Streamlit dashboard includes:

1. Overview
2. Demand Forecast
3. Revenue Forecast
4. Category Comparison
5. Model Evaluation
6. Feature Importance
7. Scenario Predictor

The forecast pages provide CSV download options.

## Important Interpretation Notes and Limitations

- The source contains only one year of historical data. Forecasts are exploratory and should not be treated as reliable long-range projections.
- The test period covers only November and December 2025.
- A course-month with no transactions is represented as zero enrollment and zero revenue.
- Zero-amount transactions count as enrollments but do not contribute to revenue.
- Revenue is estimated using price-based, Ridge, and hybrid methods. Actual revenue may differ because of discounts, refunds, and promotions.
- Teacher information is aggregated using distinct course-teacher assignments.
- The expertise-match feature uses an exact category-to-expertise comparison after case normalization.
- Scenario predictions are analytical estimates. Changing price or teacher inputs does not prove that those changes will cause the displayed outcome.
- The project is a planning and forecasting tool, not a causal pricing experiment.

## Suggested Next Steps for a Stronger Production Model

For future improvements:

- Collect at least 24–36 months of consistent historical data.
- Include course launch dates and course availability.
- Add marketing spend, discount, coupon, and refund information.
- Maintain explicit instructor-to-course assignment history.
- Evaluate models with rolling-origin backtesting across multiple forecast windows.
- Compare forecasts against simple baselines before using them for business decisions.
- Monitor forecast errors as new monthly data becomes available.
