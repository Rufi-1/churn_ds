# Statistical Insights

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
