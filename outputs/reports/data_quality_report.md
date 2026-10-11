# Data Quality Report
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
