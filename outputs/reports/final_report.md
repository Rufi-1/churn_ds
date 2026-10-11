# Customer Churn - Business Insights Report
_Generated 2026-10-11T20:46:40 | model: **Logistic Regression** | dataset version `9f931ab0c3f2`_

## 1. Executive summary
- The model flags **755 of 1,500 customers (50.3%)** as likely to churn at a probability threshold of 0.36.
- **Revenue at risk:** $53,612 per month, **$643,347 per year** (55.8% of monthly revenue is held by flagged customers).
- Probability-weighted expected monthly loss across all customers: $43,171.
- On the untouched test set the model reaches ROC-AUC 0.825, recall 0.825, precision 0.462, F1 0.592.
- Illustrative campaign economics (assumes 30% of true churners are saved and $40 per offer): cost $30,200, retained revenue about $92,225/year, net benefit about $62,025.

## 2. The five business questions

### Which customers are most likely to churn?
The priority list (ranked by a score combining churn probability, revenue, early tenure and number of risk factors) is in `outputs/tables/priority_customers_top500.csv`. Top 5:

| customerID | churn_probability | risk_tier | MonthlyCharges | tenure | priority_score | primary_action |
| --- | --- | --- | --- | --- | --- | --- |
| 1400-MMYXY | 0.966 | Critical | 105.900 | 3 | 92.770 | Offer a 12-month contract with a first-year discount or price lock |
| 4587-VVTOX | 0.951 | Critical | 105.300 | 6 | 91.430 | Offer a 12-month contract with a first-year discount or price lock |
| 5178-LMXOP | 0.965 | Critical | 95.100 | 1 | 89.630 | Offer a 12-month contract with a first-year discount or price lock |
| 6023-YEBUP | 0.945 | Critical | 100.950 | 3 | 89.410 | Offer a 12-month contract with a first-year discount or price lock |
| 8884-ADFVN | 0.948 | Critical | 101.950 | 7 | 89.150 | Offer a 12-month contract with a first-year discount or price lock |

### Which factors drive churn?
Business drivers ranked by permutation importance on the test set:

| driver | importance_mean | mean_abs_effect_pp |
| --- | --- | --- |
| Tenure / customer age | 0.0962 | 13.2384 |
| Contract type | 0.0201 | 8.4151 |
| Monthly price level | 0.0106 | 2.4266 |
| Internet service type | 0.0102 | 3.7634 |
| Payment & billing | 0.0075 | 4.5019 |

- **Tenure / customer age** - highest risk: 0-6 months (+28.4 pp); lowest risk: 49+ months (-13.3 pp).
- **Contract type** - highest risk: Month-to-month (+10.8 pp); lowest risk: Two year (-7.1 pp).
- **Monthly price level** - highest risk: Premium (+4.5 pp); lowest risk: Medium (-2.8 pp).

- Overall churn rate is 26.5% (398 of 1,500 customers); the classes are imbalanced, so accuracy alone is misleading.
- Contract type: 'Month-to-month' customers churn at 42.6% versus 2.5% for 'Two year'.
- Internet service: 'Fiber optic' customers churn at 41.5% versus 7.9% for 'No'.
- Payment method: 'Electronic check' customers churn at 44.4% versus 16.4% for 'Credit card (automatic)'.
- Tenure: '0-6 months' customers churn at 56.0% versus 7.4% for '49+ months'.
- Tech support: 'No' customers churn at 40.9% versus 15.7% for 'Yes'.

_Drivers describe associations learned by the model, not proven causes._

### How much monthly and annual revenue is at risk?
$53,612 per month / $643,347 per year among flagged customers; $28,713 per month sits with the 367 'Critical' customers (probability >= 70%).

| Contract | customers | flagged | monthly_revenue_at_risk | annual_revenue_at_risk |
| --- | --- | --- | --- | --- |
| Month-to-month | 828 | 703 | 48,973 | 587,674 |
| One year | 311 | 51 | 4,530 | 54,366 |
| Two year | 361 | 1 | 109 | 1,307 |

### Which customer segments need urgent retention action?
| segment | customers | flagged_pct | avg_churn_probability | monthly_revenue_at_risk | urgency | top_recommended_action |
| --- | --- | --- | --- | --- | --- | --- |
| Month-to-month / 0-6 months | 305 | 0.99 | 0.74 | 17,236.90 | Urgent | Offer a 12-month contract with a first-year discount or price lock |
| Month-to-month / 13-24 months | 175 | 0.86 | 0.62 | 11,510.35 | Urgent | Offer a loyalty upgrade to a 1- or 2-year contract with a free add-on |
| Month-to-month / 25-48 months | 163 | 0.72 | 0.52 | 9,974.15 | Urgent | Offer a loyalty upgrade to a 1- or 2-year contract with a free add-on |
| Month-to-month / 7-12 months | 117 | 0.80 | 0.60 | 6,627.00 | Urgent | Offer a 12-month contract with a first-year discount or price lock |
| Month-to-month / 49+ months | 68 | 0.59 | 0.43 | 3,624.45 | Urgent | Offer a loyalty upgrade to a 1- or 2-year contract with a free add-on |
| One year / 49+ months | 134 | 0.13 | 0.18 | 1,661.90 | Monitor | Offer a one-off bill credit for switching to automatic bank/credit-card payment |
| One year / 25-48 months | 108 | 0.17 | 0.20 | 1,652.95 | Monitor | Offer a free 3-month Tech Support + Online Security trial |
| One year / 13-24 months | 33 | 0.21 | 0.23 | 621.00 | Monitor | Offer a free 3-month Tech Support + Online Security trial |

### What retention actions should be prioritised?
| recommended_action | risk_factor | impact | flagged_customers_affected | monthly_revenue_covered |
| --- | --- | --- | --- | --- |
| Offer a loyalty upgrade to a 1- or 2-year contract with a free add-on | Month-to-month contract, established customer | 3 | 308 | 25,109 |
| Offer a 12-month contract with a first-year discount or price lock | Month-to-month contract, first year | 3 | 395 | 23,864 |
| Schedule a 30-day onboarding call and first-bill walkthrough | Onboarding window (very new customer) | 3 | 306 | 17,536 |
| Offer a free 3-month Tech Support + Online Security trial | Internet without security or tech support | 2 | 476 | 35,433 |
| Offer a one-off bill credit for switching to automatic bank/credit-card payment | Pays by electronic check | 2 | 405 | 30,790 |
| Run a service-quality check and a plan/price review call | High-priced fibre plan | 2 | 319 | 29,891 |
| Proactive bill review: explain the increase or offer a price lock | Bill above historical average | 2 | 51 | 3,969 |
| Offer assisted support: priority line and a simplified plan | Senior customer | 1 | 183 | 14,532 |
| Bundle security/backup add-ons at a discount to deepen engagement | Internet-only, few add-ons | 1 | 148 | 9,120 |

Actions are rule-based (contract, tenure, price, payment method, services, risk tier) and each customer gets one primary action plus any secondary ones; see `customer_recommendations.csv`.

## 3. Statistical evidence (association, not causation)
- Month-to-month vs Two-year contract: churn 42.6% vs 2.5% (difference +40.1 pts, 95% CI +36.4 to +43.9; risk ratio 17.10; adjusted p=<0.001).
- Fiber optic vs DSL internet: churn 41.5% vs 19.7% (difference +21.9 pts, 95% CI +16.8 to +27.0; risk ratio 2.11; adjusted p=<0.001).
- Electronic check vs other payment methods: churn 44.4% vs 17.4% (difference +27.0 pts, 95% CI +22.1 to +32.0; risk ratio 2.56; adjusted p=<0.001).
- Senior citizens vs non-seniors: churn 43.7% vs 23.3% (difference +20.4 pts, 95% CI +13.7 to +27.1; risk ratio 1.88; adjusted p=<0.001).
- No tech support vs tech support (internet customers): churn 40.9% vs 15.7% (difference +25.1 pts, 95% CI +20.2 to +30.1; risk ratio 2.60; adjusted p=<0.001).
- First-year (<=12 months) vs longer-tenure customers: churn 48.1% vs 16.6% (difference +31.5 pts, 95% CI +26.4 to +36.5; risk ratio 2.89; adjusted p=<0.001).

## 4. Assumptions and limitations
- Scores for training customers are out-of-fold predictions and for test customers are hold-out predictions, so no customer is scored by a model that saw their own label.
- Revenue figures use MonthlyCharges as revenue; margins, discounts and customer lifetime value are not modelled.
- Save rate and offer cost are planning assumptions in `config.yaml`; validate them with an A/B retention test.
- The IBM Telco data is a single snapshot of 7,043 customers; model drift should be monitored and the model retrained regularly. `gender` is excluded from the model as a fairness precaution.


---

## Appendix: Data Quality Report
_Generated 2026-10-11T20:45:14_

## Overview
- Raw rows x columns: **1,500 x 21**; clean rows: **1,500**
- Duplicate rows (raw): **0**; duplicate customer IDs (raw): **0**
- Missing cells raw (incl. blank strings): **3**; after cleaning: **0**
- Invalid category values found in raw data: **0** distinct values
- Overall churn rate: **26.53%**

## Missing values (raw)
| column | missing | missing_pct | blank_strings |
| --- | --- | --- | --- |
| TotalCharges | 3 | 0.20 | 3 |

## Invalid categories (raw)
_(no rows)_

## Data types and cardinality
| column | raw_dtype | clean_dtype | unique_values_raw | unique_values_clean |
| --- | --- | --- | --- | --- |
| customerID | str | object | 1,500 | 1,500 |
| gender | str | object | 2 | 2 |
| SeniorCitizen | int64 | object | 2 | 2 |
| Partner | str | object | 2 | 2 |
| Dependents | str | object | 2 | 2 |
| tenure | int64 | int64 | 73 | 73 |
| PhoneService | str | object | 2 | 2 |
| MultipleLines | str | object | 3 | 3 |
| InternetService | str | object | 3 | 3 |
| OnlineSecurity | str | object | 3 | 3 |
| OnlineBackup | str | object | 3 | 3 |
| DeviceProtection | str | object | 3 | 3 |
| TechSupport | str | object | 3 | 3 |
| StreamingTV | str | object | 3 | 3 |
| StreamingMovies | str | object | 3 | 3 |
| Contract | str | object | 3 | 3 |
| PaperlessBilling | str | object | 2 | 2 |
| PaymentMethod | str | object | 4 | 4 |
| MonthlyCharges | float64 | float64 | 844 | 844 |
| TotalCharges | str | float64 | 1,463 | 1,463 |
| Churn | str | object | 2 | 2 |

## Numeric summary (clean)
| column | count | mean | std | min | 25% | median | 75% | max | skew | zeros | iqr_outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenure | 1,500 | 31.93 | 24.46 | 0.00 | 8.00 | 28.00 | 55.00 | 72.00 | 0.25 | 3 | 0 |
| MonthlyCharges | 1,500 | 64.10 | 29.80 | 18.25 | 35.40 | 69.90 | 89.40 | 116.80 | -0.21 | 0 | 0 |
| TotalCharges | 1,500 | 2,194.04 | 2,198.31 | 0.00 | 370.01 | 1,391.95 | 3,491.84 | 8,496.70 | 1.03 | 3 | 11 |

## Cleaning log
| step | column | issue | rows_affected | action |
| --- | --- | --- | --- | --- |
| whitespace | TotalCharges | leading/trailing/duplicate whitespace | 3 | trimmed and collapsed spaces |
| missing | TotalCharges | blank string | 3 | converted to NaN |
| categories | SeniorCitizen | numeric 0/1 coding | 1,500 | mapped 0/1 to No/Yes |
| missing | TotalCharges | missing for tenure = 0 (not yet billed) | 3 | set to 0.0 (logical rule) |
| outliers | TotalCharges | values outside Tukey IQR fences | 11 | kept (genuine values); winsorised on training data in the model pipeline |
| summary | - | 1500 raw rows -> 1500 clean rows | 0 | rows removed in total |


---

## Appendix: Exploratory Data Analysis Summary

## Key business insights

- Overall churn rate is 26.5% (398 of 1,500 customers); the classes are imbalanced, so accuracy alone is misleading.
- Contract type: 'Month-to-month' customers churn at 42.6% versus 2.5% for 'Two year'.
- Internet service: 'Fiber optic' customers churn at 41.5% versus 7.9% for 'No'.
- Payment method: 'Electronic check' customers churn at 44.4% versus 16.4% for 'Credit card (automatic)'.
- Tenure: '0-6 months' customers churn at 56.0% versus 7.4% for '49+ months'.
- Tech support: 'No' customers churn at 40.9% versus 15.7% for 'Yes'.
- Online security: 'No' customers churn at 41.6% versus 14.6% for 'Yes'.
- Senior citizen status: 'Yes' customers churn at 43.7% versus 23.3% for 'No'.
- Paperless billing: 'Yes' customers churn at 32.8% versus 17.3% for 'No'.
- Customer value segment: 'High' customers churn at 38.5% versus 11.9% for 'Low'.
- Gender shows little difference in churn (Female 28.3%, Male 24.9%).
- Churned customers have a median tenure of 8 months vs 38 for retained, and median monthly charges of $76.58 vs $64.58.
- High correlation (|r| >= 0.8) between predictors: Tenure (months) & Total charges (r=0.81); Monthly charges & Average monthly spend (r=1.00); Monthly charges & Number of services (r=0.85); Average monthly spend & Number of services (r=0.84); Number of services & Protection add-ons (count) (r=0.85). These are redundant for linear models; regularisation / tree models are used.
- Multicollinearity (VIF > 5): Monthly charges (129.7), Average monthly spend (126.2), Number of services (27.6), Total charges (9.6), Protection add-ons (count) (8.8), Tenure (months) (5.6).

## Strength of association with churn (Cramer's V)
| label | cramers_v |
| --- | --- |
| Contract type | 0.411 |
| Tenure band | 0.403 |
| Service profile | 0.357 |
| Online security | 0.343 |
| Tech support | 0.330 |
| Internet service type | 0.313 |
| Online backup | 0.290 |
| Payment method | 0.290 |
| Device protection | 0.288 |
| Streaming TV | 0.232 |
| Streaming movies | 0.229 |
| Customer value segment | 0.225 |
| Paperless billing | 0.173 |
| Has partner | 0.173 |
| Senior citizen | 0.169 |
| Has dependents | 0.160 |
| Multiple lines | 0.041 |
| Gender | 0.038 |
| Phone service | 0.014 |

## Multicollinearity (VIF)
| feature | vif | concern |
| --- | --- | --- |
| MonthlyCharges | 129.71 | severe (>10) |
| avg_monthly_spend | 126.16 | severe (>10) |
| total_services | 27.61 | severe (>10) |
| TotalCharges | 9.65 | moderate (>5) |
| protection_count | 8.84 | moderate (>5) |
| tenure | 5.60 | moderate (>5) |
| streaming_count | 3.89 | ok |

## Charts
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_01_target_distribution.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_02_numeric_distributions.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_03_categorical_counts.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_04_churn_by_Contract.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_05_churn_by_tenure_group.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_06_churn_by_InternetService.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_07_churn_by_PaymentMethod.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_08_churn_by_SeniorCitizen.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_09_churn_by_gender.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_10_churn_by_TechSupport.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_11_churn_by_OnlineSecurity.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_12_numeric_by_churn_boxplots.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_13_numeric_by_churn_density.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_14_tenure_vs_monthly_scatter.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_15_correlation_heatmap.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_16_vif.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_17_cramers_v.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_18_multivariate_heatmaps.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_19_churn_by_services.png`
- `C:\Users\USER\Desktop\Customer_Churn_Prediction\churn_ds\outputs\figures\eda_20_churn_by_tenure_month.png`


---

## Appendix: Statistical Insights

Significance level alpha = 0.05. p-values are adjusted across tests with the Benjamini-Hochberg procedure to limit false discoveries.

> These tests show statistical **association**, not causation. Customers who differ on one attribute usually differ on many others (e.g. month-to-month customers are also newer), so a significant result is a lead to investigate, not proof that changing the attribute would change churn. Only a controlled retention experiment (A/B test) can show cause.

## 1. Chi-square tests: churn vs categorical variables (sorted by effect size)
| label | chi2 | dof | p_value | p_adjusted | cramers_v | effect_size |
| --- | --- | --- | --- | --- | --- | --- |
| Contract type | 252.8214 | 2 | 0.0000 | 0.0000 | 0.4105 | moderate |
| Online security | 176.9798 | 2 | 0.0000 | 0.0000 | 0.3435 | moderate |
| Tech support | 163.2663 | 2 | 0.0000 | 0.0000 | 0.3299 | moderate |
| Internet service type | 146.9219 | 2 | 0.0000 | 0.0000 | 0.3130 | moderate |
| Online backup | 126.5090 | 2 | 0.0000 | 0.0000 | 0.2904 | small |
| Payment method | 126.4550 | 3 | 0.0000 | 0.0000 | 0.2904 | small |
| Device protection | 124.3881 | 2 | 0.0000 | 0.0000 | 0.2880 | small |
| Streaming TV | 80.5939 | 2 | 0.0000 | 0.0000 | 0.2318 | small |
| Streaming movies | 78.4597 | 2 | 0.0000 | 0.0000 | 0.2287 | small |
| Paperless billing | 44.2142 | 1 | 0.0000 | 0.0000 | 0.1732 | small |
| Has partner | 43.8795 | 1 | 0.0000 | 0.0000 | 0.1725 | small |
| Senior citizen | 41.7133 | 1 | 0.0000 | 0.0000 | 0.1688 | small |
| Has dependents | 37.5512 | 1 | 0.0000 | 0.0000 | 0.1599 | small |
| Multiple lines | 2.4653 | 2 | 0.2915 | 0.3110 | 0.0405 | negligible |
| Gender | 1.9952 | 1 | 0.1578 | 0.1803 | 0.0380 | negligible |
| Phone service | 0.2068 | 1 | 0.6493 | 0.6493 | 0.0144 | negligible |

- Churn differs significantly across contract type groups (moderate association, Cramer's V=0.41): 'Month-to-month' churns at 42.6% vs 2.5% for 'Two year'.
- Churn differs significantly across online security groups (moderate association, Cramer's V=0.34): 'No' churns at 41.6% vs 7.9% for 'No internet service'.
- Churn differs significantly across tech support groups (moderate association, Cramer's V=0.33): 'No' churns at 40.9% vs 7.9% for 'No internet service'.
- Churn differs significantly across internet service type groups (moderate association, Cramer's V=0.31): 'Fiber optic' churns at 41.5% vs 7.9% for 'No'.
- Churn differs significantly across online backup groups (small association, Cramer's V=0.29): 'No' churns at 39.6% vs 7.9% for 'No internet service'.
- Churn differs significantly across payment method groups (small association, Cramer's V=0.29): 'Electronic check' churns at 44.4% vs 16.4% for 'Credit card (automatic)'.
- Churn differs significantly across device protection groups (small association, Cramer's V=0.29): 'No' churns at 39.7% vs 7.9% for 'No internet service'.
- Churn differs significantly across streaming tv groups (small association, Cramer's V=0.23): 'No' churns at 34.4% vs 7.9% for 'No internet service'.
- Churn differs significantly across streaming movies groups (small association, Cramer's V=0.23): 'No' churns at 33.8% vs 7.9% for 'No internet service'.
- Churn differs significantly across paperless billing groups (small association, Cramer's V=0.17): 'Yes' churns at 32.8% vs 17.3% for 'No'.
- Churn differs significantly across has partner groups (small association, Cramer's V=0.17): 'No' churns at 33.9% vs 18.6% for 'Yes'.
- Churn differs significantly across senior citizen groups (small association, Cramer's V=0.17): 'Yes' churns at 43.7% vs 23.3% for 'No'.
- Churn differs significantly across has dependents groups (small association, Cramer's V=0.16): 'No' churns at 31.2% vs 15.9% for 'Yes'.
- No statistically significant difference in churn across multiple lines groups (adjusted p=0.311).
- No statistically significant difference in churn across gender groups (adjusted p=0.180).
- No statistically significant difference in churn across phone service groups (adjusted p=0.649).

## 2. Numeric variables: churned vs retained customers
Mann-Whitney U is used when either group is non-normal (Shapiro-Wilk, p<0.05); otherwise Welch's t-test.

| label | test | statistic | p_adjusted | effect_size_name | effect_size | effect_magnitude | median_churned | median_retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tenure (months) | Mann-Whitney U | 103,815.0000 | 0.0000 | rank-biserial r | 0.5266 | large | 8.0000 | 38.0000 |
| Monthly charges | Mann-Whitney U | 269,095.0000 | 0.0000 | rank-biserial r | -0.2271 | small | 76.5750 | 64.5750 |
| Total charges | Mann-Whitney U | 143,524.0000 | 0.0000 | rank-biserial r | 0.3455 | medium | 549.0750 | 1,659.1000 |
| Average monthly spend | Mann-Whitney U | 269,141.5000 | 0.0000 | rank-biserial r | -0.2273 | small | 77.4428 | 64.0017 |
| Number of services | Mann-Whitney U | 213,760.0000 | 0.4505 | rank-biserial r | 0.0253 | negligible | 4.0000 | 4.0000 |

- Churned customers have lower tenure (months) (median 8.0 vs 38.0); Mann-Whitney U p=<0.001, large effect (rank-biserial r=0.53).
- Churned customers have higher monthly charges (median 76.6 vs 64.6); Mann-Whitney U p=<0.001, small effect (rank-biserial r=-0.23).
- Churned customers have lower total charges (median 549.1 vs 1,659.1); Mann-Whitney U p=<0.001, medium effect (rank-biserial r=0.35).
- Churned customers have higher average monthly spend (median 77.4 vs 64.0); Mann-Whitney U p=<0.001, small effect (rank-biserial r=-0.23).
- No significant difference in number of services between churned and retained customers (adjusted p=0.451).

## 3. Business-relevant group comparisons (two-proportion z-tests)
| comparison | rate_a | rate_b | difference | risk_ratio | p_adjusted |
| --- | --- | --- | --- | --- | --- |
| Month-to-month vs Two-year contract | 0.4263 | 0.0249 | 0.4014 | 17.1005 | 0.0000 |
| Fiber optic vs DSL internet | 0.4154 | 0.1965 | 0.2189 | 2.1136 | 0.0000 |
| Electronic check vs other payment methods | 0.4440 | 0.1736 | 0.2704 | 2.5582 | 0.0000 |
| Senior citizens vs non-seniors | 0.4370 | 0.2330 | 0.2040 | 1.8757 | 0.0000 |
| No tech support vs tech support (internet customers) | 0.4085 | 0.1571 | 0.2514 | 2.5998 | 0.0000 |
| First-year (<=12 months) vs longer-tenure customers | 0.4809 | 0.1663 | 0.3146 | 2.8912 | 0.0000 |
| Paperless vs paper billing | 0.3285 | 0.1727 | 0.1558 | 1.9020 | 0.0000 |
| Male vs female customers | 0.2490 | 0.2826 | -0.0336 | 0.8813 | 0.1413 |

- Month-to-month vs Two-year contract: churn 42.6% vs 2.5% (difference +40.1 pts, 95% CI +36.4 to +43.9; risk ratio 17.10; adjusted p=<0.001).
- Fiber optic vs DSL internet: churn 41.5% vs 19.7% (difference +21.9 pts, 95% CI +16.8 to +27.0; risk ratio 2.11; adjusted p=<0.001).
- Electronic check vs other payment methods: churn 44.4% vs 17.4% (difference +27.0 pts, 95% CI +22.1 to +32.0; risk ratio 2.56; adjusted p=<0.001).
- Senior citizens vs non-seniors: churn 43.7% vs 23.3% (difference +20.4 pts, 95% CI +13.7 to +27.1; risk ratio 1.88; adjusted p=<0.001).
- No tech support vs tech support (internet customers): churn 40.9% vs 15.7% (difference +25.1 pts, 95% CI +20.2 to +30.1; risk ratio 2.60; adjusted p=<0.001).
- First-year (<=12 months) vs longer-tenure customers: churn 48.1% vs 16.6% (difference +31.5 pts, 95% CI +26.4 to +36.5; risk ratio 2.89; adjusted p=<0.001).
- Paperless vs paper billing: churn 32.8% vs 17.3% (difference +15.6 pts, 95% CI +11.3 to +19.9; risk ratio 1.90; adjusted p=<0.001).
- Male vs female customers: churn 24.9% vs 28.3% (difference -3.4 pts, 95% CI -7.8 to +1.1; risk ratio 0.88; adjusted p=0.141).

## 4. Correlation with churn (numeric variables)
| variable | label | pearson_r | pearson_p | spearman_rho | spearman_p |
| --- | --- | --- | --- | --- | --- |
| tenure | Tenure (months) | -0.3870 | 0.0000 | -0.4029 | 0.0000 |
| TotalCharges | Total charges | -0.2232 | 0.0000 | -0.2642 | 0.0000 |
| MonthlyCharges | Monthly charges | 0.1816 | 0.0000 | 0.1736 | 0.0000 |
| avg_monthly_spend | Average monthly spend | 0.1807 | 0.0000 | 0.1738 | 0.0000 |
| total_services | Number of services | -0.0317 | 0.2200 | -0.0195 | 0.4506 |

Large |r| between predictors (e.g. tenure and total charges) signals multicollinearity: see the EDA report for VIF values.


---

## Appendix: Model Evaluation Report

- **Final model:** Logistic Regression  |  **Trained:** 2026-10-11T20:46:07  |  **Dataset version:** `9f931ab0c3f2`
- **Evaluated on the untouched test set:** 300 customers (80 churners)
- **Decision threshold:** 0.36

## Why this model was selected
Logistic Regression (0.8466) was within 0.003 of the best model XGBoost (0.8477) on cross-validated ROC_AUC; the simpler, more interpretable model was preferred (parsimony rule).

## Cross-validated model comparison (training set only)
| model | display_name | cv_roc_auc | cv_pr_auc | cv_f1 | cv_recall | cv_precision | cv_roc_auc_std | cv_pr_auc_std | cv_f1_std | cv_recall_std | cv_precision_std | best_params | fit_seconds | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | Logistic Regression | 0.8466 | 0.6567 | 0.6259 | 0.7893 | 0.5195 | 0.0035 | 0.0314 | 0.0014 | 0.0311 | 0.0139 | {'C': 0.1} | 11.3000 | 1 |
| random_forest | Random Forest | 0.8468 | 0.6627 | 0.6161 | 0.7516 | 0.5220 | 0.0067 | 0.0275 | 0.0126 | 0.0118 | 0.0130 | {'n_estimators': 40, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'max_depth': 6} | 4.0000 | 0 |
| gradient_boosting | Gradient Boosting | 0.8425 | 0.6533 | 0.5784 | 0.5220 | 0.6506 | 0.0038 | 0.0141 | 0.0389 | 0.0495 | 0.0277 | {'subsample': 0.8, 'n_estimators': 40, 'min_samples_leaf': 10, 'max_depth': 3, 'learning_rate': 0.1} | 3.9000 | 0 |
| xgboost | XGBoost | 0.8477 | 0.6599 | 0.6359 | 0.7987 | 0.5285 | 0.0090 | 0.0360 | 0.0073 | 0.0089 | 0.0127 | {'subsample': 0.8, 'reg_lambda': 1.0, 'n_estimators': 60, 'min_child_weight': 5, 'max_depth': 2, 'learning_rate': 0.1, 'colsample_bytree': 0.7} | 3.6000 | 0 |

## Test-set metrics
| Metric | Value |
| --- | --- |
| accuracy | 0.6967 |
| precision | 0.4615 |
| recall | 0.8250 |
| f1 | 0.5919 |
| roc_auc | 0.8246 |
| pr_auc | 0.6087 |
| balanced_accuracy | 0.7375 |
| specificity | 0.6500 |
| brier_score | 0.1708 |

## Confusion matrix
|  | Predicted retained | Predicted churn |
| --- | --- | --- |
| Actually retained | 143 | 77 |
| Actually churned | 14 | 66 |

Of 80 real churners the model catches 66 (recall 82.5%); 14 are missed. Of 143 customers flagged, 66 truly churn (precision 46.2%).

## Classification report
| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Retained | 0.911 | 0.650 | 0.759 | 220.000 |
| Churned | 0.462 | 0.825 | 0.592 | 80.000 |
| accuracy | 0.697 | 0.697 | 0.697 | 0.697 |
| macro avg | 0.686 | 0.738 | 0.675 | 300.000 |
| weighted avg | 0.791 | 0.697 | 0.714 | 300.000 |

## Precision-recall trade-off and threshold choice
**Precision vs recall for churn.** Recall is the share of real churners we catch; precision is the share of flagged customers who really would have left. Lowering the probability threshold flags more customers: recall rises (fewer missed churners, i.e. less lost revenue) but precision falls (more retention offers wasted on customers who would have stayed). Raising it does the opposite. Because a lost customer typically costs far more than one retention offer, the default threshold is tuned with an F-beta score that weights recall more than precision. The 'net benefit' column/curve uses the offer cost and save-rate assumptions in `config.yaml`; change `threshold.value`/`threshold.mode` there (or use the slider in the dashboard) to see the effect.

| threshold | customers_flagged | precision | recall | f1 | annual_revenue_protected | campaign_cost | net_benefit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 255 | 0.314 | 1.000 | 0.478 | 20,967.480 | 10,200.000 | 10,767.480 |
| 0.100 | 216 | 0.361 | 0.975 | 0.527 | 20,159.640 | 8,640.000 | 11,519.640 |
| 0.150 | 195 | 0.379 | 0.925 | 0.538 | 18,863.640 | 7,800.000 | 11,063.640 |
| 0.200 | 177 | 0.412 | 0.912 | 0.568 | 18,794.160 | 7,080.000 | 11,714.160 |
| 0.250 | 163 | 0.429 | 0.875 | 0.576 | 17,793.180 | 6,520.000 | 11,273.180 |
| 0.300 | 149 | 0.450 | 0.838 | 0.585 | 16,671.240 | 5,960.000 | 10,711.240 |
| 0.350 | 143 | 0.462 | 0.825 | 0.592 | 16,417.620 | 5,720.000 | 10,697.620 |
| 0.400 | 138 | 0.471 | 0.812 | 0.596 | 16,218.900 | 5,520.000 | 10,698.900 |
| 0.450 | 125 | 0.496 | 0.775 | 0.605 | 16,005.420 | 5,000.000 | 11,005.420 |
| 0.500 | 118 | 0.517 | 0.762 | 0.616 | 15,933.060 | 4,720.000 | 11,213.060 |
| 0.550 | 112 | 0.545 | 0.762 | 0.635 | 15,933.060 | 4,480.000 | 11,453.060 |
| 0.600 | 101 | 0.554 | 0.700 | 0.619 | 14,772.060 | 4,040.000 | 10,732.060 |
| 0.650 | 87 | 0.586 | 0.637 | 0.611 | 13,361.760 | 3,480.000 | 9,881.760 |
| 0.700 | 78 | 0.603 | 0.588 | 0.595 | 12,584.160 | 3,120.000 | 9,464.160 |
| 0.750 | 63 | 0.619 | 0.487 | 0.545 | 10,693.620 | 2,520.000 | 8,173.620 |
| 0.800 | 45 | 0.600 | 0.338 | 0.432 | 7,740.720 | 1,800.000 | 5,940.720 |
| 0.850 | 29 | 0.724 | 0.263 | 0.385 | 6,023.880 | 1,160.000 | 4,863.880 |
| 0.900 | 12 | 0.667 | 0.100 | 0.174 | 2,424.600 | 480.000 | 1,944.600 |
| 0.950 | 2 | 1.000 | 0.025 | 0.049 | 649.800 | 80.000 | 569.800 |

## Charts
- `outputs/figures/eval_roc_curve.png`
- `outputs/figures/eval_precision_recall_curve.png`
- `outputs/figures/eval_confusion_matrix.png`
- `outputs/figures/eval_threshold_tradeoff.png`
- `outputs/figures/eval_score_distribution.png`
- `outputs/figures/eval_calibration.png`
- `outputs/figures/eval_cumulative_gains.png`


---

## Appendix: Model Explainability

Model: **Logistic Regression**. All numbers below are computed from this model on the held-out test set; they describe what the *model* relies on, which reflects patterns in the data (association) and not proven causes.

## Global drivers (grouped permutation importance)
Importance = how much test ROC-AUC drops when a driver's columns are shuffled. Related columns (e.g. contract type + month-to-month flag) are shuffled together so credit is not split.

| driver | importance_mean | importance_std | mean_abs_effect_pp |
| --- | --- | --- | --- |
| Tenure / customer age | 0.0962 | 0.0151 | 13.2384 |
| Contract type | 0.0201 | 0.0096 | 8.4151 |
| Monthly price level | 0.0106 | 0.0037 | 2.4266 |
| Internet service type | 0.0102 | 0.0059 | 3.7634 |
| Payment & billing | 0.0075 | 0.0058 | 4.5019 |
| Streaming services | 0.0066 | 0.0046 | 3.9234 |
| Household / demographics | 0.0043 | 0.0025 | 2.0691 |
| Security & tech support | 0.0041 | 0.0035 | 2.6877 |
| Service mix | 0.0040 | 0.0027 | 2.2199 |

## What pushes risk up or down (average model effect, percentage points vs a typical customer)

- **Tenure / customer age**: 0-6 months (+28.4 pp, n=51); 7-12 months (+8.3 pp, n=31); 13-24 months (+7.3 pp, n=34); 25-48 months (-3.3 pp, n=54); 49+ months (-13.3 pp, n=80)
- **Contract type**: Month-to-month (+10.8 pp, n=131); One year (-4.2 pp, n=55); Two year (-7.1 pp, n=64)
- **Monthly price level**: Premium (+4.5 pp, n=57); High (-0.1 pp, n=58); Low (-1.5 pp, n=62); Medium (-2.8 pp, n=73)
- **Internet service type**: Fiber optic (+3.5 pp, n=100); No (-2.6 pp, n=57); DSL (-3.2 pp, n=93)
- **Payment & billing**: Electronic check (+6.5 pp, n=84); Bank transfer (automatic) (+1.0 pp, n=54); Credit card (automatic) (-2.3 pp, n=49); Mailed check (-4.8 pp, n=63)

## How to read a single customer's explanation
Each driver's effect is the customer's predicted churn probability minus the average probability obtained when that driver's values are replaced by those of random training customers. Positive = pushes the customer above a typical customer's risk. Effects do not add up exactly (interactions).

## SHAP global importance (top 15)
| feature | source_column | mean_abs_shap |
| --- | --- | --- |
| Tenure (months) | tenure | 0.6791 |
| Contract risk score | contract_risk_score | 0.2899 |
| Fiber without tech support | fiber_no_support | 0.2355 |
| Tenure band: 0-6 months | tenure_group | 0.1871 |
| Streaming add-ons (count) | streaming_count | 0.1854 |
| Month-to-month contract | is_month_to_month | 0.1848 |
| Billing risk score | billing_risk_score | 0.1645 |
| Pays by electronic check | electronic_check | 0.1126 |
| Service profile: Phone+Internet+Protection+Streaming | service_profile | 0.1102 |
| Monthly charges | MonthlyCharges | 0.0846 |
| Senior citizen: No | SeniorCitizen | 0.0738 |
| Senior citizen: Yes | SeniorCitizen | 0.0737 |
| Average monthly spend | avg_monthly_spend | 0.0657 |
| Customer value segment: Premium | customer_value_segment | 0.0651 |
| Contract type: Two year | Contract | 0.0651 |
