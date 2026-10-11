# Data Dictionary

Source: **IBM Telco Customer Churn** sample dataset (7,043 customers, 21 columns), published by IBM under the
Apache-2.0 license in `IBM/telco-customer-churn-on-icp4d`. `data/sample/telco_churn_sample.csv` is a
stratified 1,500-row subset of the same file for offline use (same churn rate, 26.5%).

## Data layers
| Layer | File | Description |
| --- | --- | --- |
| Raw | `data/raw/telco_churn.csv` | Exactly as downloaded. Never edited. |
| Cleaned | `data/processed/telco_clean.csv` | Types fixed, categories standardised, logical repairs, `ChurnFlag` added. Statistical imputation is *not* done here. |
| Modeling-ready | `data/processed/telco_modeling.csv` | Cleaned + engineered features, un-scaled, with `customerID` and `ChurnFlag`. Scaling / encoding / imputation happen inside the scikit-learn pipeline, fitted on training data only. |

## Raw columns
| Column | Type | Values / meaning |
| --- | --- | --- |
| customerID | text | Unique customer identifier |
| gender | category | Female, Male (analysed in EDA/statistics; **excluded from the model** as a fairness precaution) |
| SeniorCitizen | category | Raw file: 0/1; cleaned: No / Yes |
| Partner, Dependents | category | Yes / No |
| tenure | integer | Months with the company (0-72) |
| PhoneService | category | Yes / No |
| MultipleLines | category | Yes, No, No phone service |
| InternetService | category | DSL, Fiber optic, No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | category | Yes, No, No internet service (TechSupport / OnlineSecurity are the support-related variables) |
| Contract | category | Month-to-month, One year, Two year |
| PaperlessBilling | category | Yes / No |
| PaymentMethod | category | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| MonthlyCharges | float | Current monthly bill (USD) |
| TotalCharges | float | Lifetime billed amount (USD). Blank in the raw file for the 11 customers with tenure 0 |
| Churn | category | **Target.** Yes = customer left within the last month, No = retained |

## Added by cleaning
| Column | Meaning |
| --- | --- |
| ChurnFlag | 1 if Churn = Yes else 0 (the modeling target) |

## Engineered features (`churn/features.py`)
| Feature | Definition |
| --- | --- |
| tenure_group | 0-6, 7-12, 13-24, 25-48, 49+ months |
| avg_monthly_spend | TotalCharges / tenure (MonthlyCharges when tenure = 0) |
| charge_delta | MonthlyCharges - avg_monthly_spend (current bill vs history) |
| bill_increase_flag | 1 if charge_delta > 5 (configurable) |
| total_services | phone + internet + multiple lines + protection add-ons + streaming add-ons (0-9) |
| protection_count | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport = Yes (0-4) |
| streaming_count | StreamingTV, StreamingMovies = Yes (0-2) |
| no_support_flag | Has internet but neither OnlineSecurity nor TechSupport |
| fiber_no_support | Fiber optic customer without TechSupport |
| is_month_to_month | Contract = Month-to-month |
| contract_risk_score | Month-to-month 2, One year 1, Two year 0 |
| electronic_check, auto_pay | Payment method indicators |
| billing_risk_score | month-to-month + electronic check + paperless billing + bill increase (0-4) |
| customer_value_segment | MonthlyCharges: Low (<35), Medium (35-70), High (70-90), Premium (>90) |
| service_profile | Combination of Phone / Internet / Protection / Streaming, e.g. `Phone+Internet+Streaming` |

## Generated analysis tables (`outputs/tables/`)
`customer_scores.csv` (churn probability for every customer: out-of-fold for training customers, hold-out for test
customers), `customer_recommendations.csv` (risk tier, priority score/rank, recommended actions, revenue at risk),
`priority_customers_top500.csv`, `insights_*.csv`, `stats_*.csv`, `quality_*.csv`, `model_comparison.csv`,
`eval_*.csv`, `explain_*.csv`, `revenue_at_risk_summary.json`.
