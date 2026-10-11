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
