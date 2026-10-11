# Model Explainability

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
