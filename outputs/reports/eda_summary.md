# Exploratory Data Analysis Summary

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
