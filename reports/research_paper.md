# Research Paper: Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

## Abstract

This project develops a course-level forecasting workflow for EduPro, an online learning platform. It uses historical transactions and course/instructor metadata to estimate monthly enrollment demand and revenue. Transaction records are aggregated into a course-month panel. Course attributes, instructor summaries, calendar variables and lagged performance are used as predictors.

Five machine-learning algorithms are evaluated: Linear Regression, Ridge Regression, Lasso Regression, Random Forest and Gradient Boosting. Enrollment models are also compared with simple forecasting baselines. A chronological training, validation and testing strategy is used to reduce temporal leakage. The resulting Streamlit application presents historical performance, category comparisons, forecasts, model evaluation, feature importance and course-planning scenarios.

## 1. Introduction and problem statement

EduPro's planning decisions include selecting future courses, adjusting course pricing and allocating instructor resources. Historical dashboards describe past activity but do not directly provide forward-looking estimates.

The objective is to build predictive indicators for:

1. Monthly course enrollment count.
2. Monthly course revenue.
3. Category-level demand and revenue derived by aggregating course-level results.

## 2. Dataset description

The supplied source exports contain:

- **Courses:** CourseID, CourseName, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration and CourseRating.
- **Teachers:** TeacherID, expertise, years of experience and teacher rating. Teacher names and demographic fields are not used.
- **Transactions:** TransactionID, UserID, CourseID, TransactionDate, Amount, PaymentMethod and TeacherID.
- **Users:** User identifiers and demographic/contact fields. These fields are excluded from modelling because they are unnecessary for the stated targets and include personal information.

The extracted modelling data contains 60 courses, 60 teachers and 10,000 transaction records covering January–December 2025. The recorded transaction amount is used as revenue. Zero-amount transactions remain valid enrollments and contribute to demand, but not revenue.

Total recorded transaction revenue is 911,323.47 in the dataset's unspecified currency. Approximately 64.03% of transactions have a zero amount.

## 3. Data preparation

Transactions are parsed into dates and numeric amounts. Data quality checks include missing values, duplicate transaction identifiers, invalid amounts and referential integrity for course and teacher IDs.

Transactions are aggregated by CourseID and calendar month:

- **Enrollment Count:** Count of transaction records.
- **Course Revenue:** Sum of transaction Amount.

A complete course-by-month panel is created so that months with no transactions are represented explicitly as zero demand and zero revenue. Course metadata is joined using CourseID.

Teacher-related features are calculated using distinct CourseID–TeacherID assignments rather than weighting teachers by the number of transactions. These features include average teacher experience, average teacher rating, teacher count and an expertise-to-course-category match score.

The users dataset is excluded from model training because personal contact and demographic information is not required for the forecasting objectives.

## 4. Feature engineering

### 4.1 Course features

Course category, course type, course level, listed price, duration and rating are included. Additional categorical features are created for price bands, duration buckets and rating tiers.

### 4.2 Instructor features

Instructor summaries include:

- Average teacher experience.
- Average teacher rating.
- Number of distinct teachers associated with a course.
- Expertise match score based on teacher expertise and course category.

### 4.3 Historical performance features

Historical features include:

- Previous-month enrollment and revenue.
- Two-month lagged enrollment and revenue.
- Trailing three-month enrollment and revenue averages.
- Lagged revenue per enrollment.

Lagged and rolling features are shifted so that the current month's target is not used to predict itself. Missing initial lag values are filled with zero.

### 4.4 Calendar features

Calendar predictors include month number, quarter, sine/cosine month encoding and a sequential time index.

### 4.5 Preprocessing

Categorical features are one-hot encoded. Numeric features are imputed with the median and standardized within a scikit-learn pipeline.

## 5. Modelling methodology

Two separate regression tasks are implemented:

1. Monthly enrollment prediction.
2. Monthly revenue modelling.

The five candidate machine-learning algorithms are:

1. Linear Regression.
2. Ridge Regression.
3. Lasso Regression.
4. Random Forest Regressor.
5. Gradient Boosting Regressor.

Enrollment predictions are also compared with simple baselines:

- Global Mean Baseline.
- Course Mean Baseline.
- Previous Month Baseline.
- Previous 3-Month Average.

### 5.1 Chronological data split

The 12-month dataset is divided chronologically:

| Period | Months | Purpose |
|---|---|---|
| Training | January–August 2025 | Model fitting |
| Validation | September–October 2025 | Model selection |
| Testing | November–December 2025 | Final evaluation |

Models are selected using validation RMSE rather than test performance. The selected models are subsequently evaluated on the held-out test period.

### 5.2 Model selection

Lasso Regression achieved the best validation RMSE among the machine-learning candidates for enrollment prediction. Ridge Regression achieved the best validation RMSE among the revenue machine-learning candidates.

However, the global-mean baseline performed better than the enrollment machine-learning candidates on validation. Therefore, the selected enrollment machine-learning model has not demonstrated an advantage over the simple baseline.

### 5.3 Forecast generation

The enrollment forecasting process generates predictions recursively for the next three months. Predicted values are used as lag inputs for subsequent forecast months.

The supplied transaction history ends on 30 December 2025, so the generated forecast period is January–March 2026.

The forecast includes three revenue methods: price-based (predicted enrollments multiplied by course price), Ridge Regression, and hybrid revenue. The hybrid method uses Ridge predictions for paid courses and zero revenue for free courses.

## 6. Evaluation

The project reports three evaluation metrics:

- **MAE:** Average absolute prediction error.
- **RMSE:** Square root of mean squared error, which penalizes larger errors.
- **R²:** Proportion of target variance explained relative to a constant-mean baseline.

### 6.1 Final test results

The final test evaluation on November–December 2025 produced:

| Target | Selected model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Enrollments | Lasso Regression | 3.104 | 4.046 | -0.211 |
| Revenue | Ridge Regression | 250.586 | 577.369 | 0.9316 |

The enrollment model's negative R² indicates that it did not outperform a constant-mean prediction on the test period. Enrollment forecasts should therefore be treated cautiously.

Ridge Regression achieved test R-squared of 0.9316. An exploratory hybrid comparison achieved MAE 235.250, RMSE 576.332, and R-squared of 0.9319. The hybrid rule was assessed after inspecting test behavior, so its improvement requires independent validation.

Only two months are available in the final test period. The results are therefore not sufficient to establish stable long-term forecasting performance.

### 6.2 Exploratory revenue-method comparison

The November-December 2025 test period contains 120
course-month records.

| Method | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Hybrid | 235.250 | 576.332 | 0.9319 |
| Ridge Regression | 250.586 | 577.369 | 0.9316 |
| Price-based | 280.817 | 684.235 | 0.9040 |
| Training-mean baseline | 1734.196 | 2208.638 | -0.0003 |

The hybrid method uses Ridge predictions for paid courses
and zero revenue for free courses.

The test period includes 76 free-course and 44 paid-course
records. Historical validation found 456 free-course monthly
records with zero recorded revenue.

Because the hybrid rule was adopted after examining test
behavior, its improvement is exploratory and requires
independent future validation.

### 6.3 Three-month forecast results

The January-March 2026 forecast contains 180 course-month
records across 60 courses.

- Predicted enrollments: 3,151.
- Price-based revenue: 286,685.77.
- Ridge revenue: 259,441.54.
- Hybrid revenue: 237,115.24.

These forecasts are based on the December 2025 dataset cutoff,
not on current operational data.

## 7. Streamlit application

The application provides:

- Historical enrollment and revenue trends.
- Top courses by enrollment and revenue.
- Three-month course demand forecasts.
- Three-month revenue estimates.
- Historical and predicted category comparisons.
- Model evaluation metrics.
- Feature importance for each target.
- A scenario predictor allowing changes to course price, duration, level and instructor experience/rating for an existing course.

The application is intended to make the modelling outputs accessible for exploration and planning.

## 8. Business interpretation and recommendations

Enrollment demand and revenue should be reviewed together rather than ranking courses on revenue alone.

Free courses can have high demand but no direct transaction revenue. They may still serve acquisition or engagement goals, which are not measured by this dataset.

Categories with high forecast revenue may merit further capacity and instructor planning, while low-revenue/high-demand categories may warrant pricing or monetization analysis.

Pricing scenarios should be treated as model sensitivity exercises, not causal evidence of price elasticity.

## 9. Limitations

- Only one year of transaction history is available.
- The final test period contains only two months.
- Enrollment test performance is weak, with a negative R².
- No marketing spend, promotions, refunds, course availability, launch dates or competitor information is provided.
- Instructor-course relationships are inferred from distinct transaction-recorded CourseID–TeacherID assignments rather than a separate authoritative assignment history.
- User demographics and contact information are excluded.
- Recursive forecasts depend on prior predictions and may accumulate error.
- The forecast file contains separate price-based, Ridge, and hybrid revenue estimates. The hybrid method enforces zero predicted revenue for free courses.
- The data does not establish causal price elasticity.

## 10. Conclusion

The project converts EduPro's historical course transactions into course-level monthly demand and revenue estimates and makes the results accessible through an interactive dashboard.

It provides a reproducible starting point for data-informed planning. The revenue model demonstrates promising performance on the available test period, while enrollment prediction remains a key area for improvement.

Reliable operational deployment requires more historical data, repeated rolling-origin backtesting, additional business drivers and ongoing monitoring of actual versus predicted outcomes.

## 11. Forecast date interpretation

The supplied transactions end on 30 December 2025. Accordingly, the three forecast months generated by this project are January–March 2026, measured from the dataset cutoff.

These are historical-cutoff forecasts, not current-date forecasts as of November 2026. Transaction history should be refreshed before presenting forecasts as current operational guidance.
