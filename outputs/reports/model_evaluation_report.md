# Model Evaluation Report

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
