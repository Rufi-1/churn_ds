# churn_ds - Customer Churn Prediction and Data Science Insights

An end-to-end, reproducible data-science project for a telecom company: **find customers likely to churn, explain why,
quantify the revenue at risk, prioritise who to contact first, and recommend practical retention actions** - from raw CSV
to a Streamlit dashboard and final report. No database server, Docker, Node.js or paid API key is needed.

> Dataset: IBM Telco Customer Churn (7,043 customers). Results in this README come from a full run on that dataset
> (seed 42). Numbers are reproducible with `python run_pipeline.py all`.

## 1. Business problem
A telecom operator loses about **26.5%** of customers in the observation window. Acquiring a replacement costs far more than
keeping a customer, so the business needs to answer:

1. Which customers are most likely to churn?
2. Which factors drive churn?
3. How much monthly and annual revenue is at risk?
4. Which customer segments need urgent retention action?
5. Which retention actions should be prioritised?

**Target variable:** `Churn` (Yes/No), modeled as `ChurnFlag` (1 = churned). Positive class = churner.

**Why these metrics?** Only ~1 in 4 customers churns, so *accuracy* hides missed churners (predicting "nobody churns" is 73%
accurate). **Recall** = share of real churners caught (missed churners are lost revenue). **Precision** = share of flagged
customers who really churn (low precision = wasted offers). **F1** balances the two. **ROC-AUC** measures how well the model
ranks churners above stayers independent of any threshold. PR-AUC is reported too because of the class imbalance.

## 2. Data-science approach
Business understanding -> data collection -> data cleaning -> data-quality assessment -> EDA -> hypothesis testing ->
feature engineering -> model development -> evaluation -> explainability -> business interpretation -> reporting ->
dashboard -> reproducibility. Every stage is a module in `churn/` and a command in `run_pipeline.py`.

## 3. Architecture (text)
```
data/raw/telco_churn.csv --seed-->  (never edited)
        | clean (churn/cleaning.py, quality.py)
        v
data/processed/telco_clean.csv + outputs/tables/quality_*.csv, logs/cleaning_log.csv, reports/data_quality_report.md
        | features (churn/features.py)
        v
data/processed/telco_modeling.csv  (engineered, un-scaled)
        |-- eda (eda.py) ---------> outputs/figures/eda_*.png, reports/eda_summary.md
        |-- stats (stats_tests.py)-> outputs/tables/stats_*.csv, reports/statistical_report.md
        '-- train (modeling.py): stratified split -> CV + tuning of 4 models -> selection -> threshold tuning
                 -> outputs/models/churn_model.joblib (+preprocessor, feature names, metrics, run registry)
                 -> outputs/tables/customer_scores.csv (out-of-fold / hold-out probabilities)
        | evaluate (evaluation.py) -> reports/model_evaluation_report.md + ROC/PR/confusion/threshold charts
        | explain  (explain.py)    -> permutation importance, SHAP, local explanations
        | insights (insights.py)   -> revenue at risk, rule-based recommendations, priorities, segments
        | report   (reporting.py)  -> reports/final_report.{md,html,pdf}
        v
app.py (Streamlit, 11 pages) reads outputs/ and the saved model
```

## 4. Tech stack
Python 3.10+, pandas, NumPy, SciPy (statistical tests), scikit-learn (Pipelines, ColumnTransformer, CV, tuning),
XGBoost (optional), SHAP (optional), Matplotlib + Seaborn, Streamlit, joblib, PyYAML, pytest. Storage: CSV/JSON/joblib only.

## 5. Dataset and data dictionary
See [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for every raw and engineered column. Raw, cleaned and
modeling-ready data are stored separately in `data/raw`, `data/processed/telco_clean.csv`, `data/processed/telco_modeling.csv`.
`data/sample/telco_churn_sample.csv` (1,500 stratified rows) is used automatically if the download fails. To replace the
dataset see [`docs/SETUP.md`](docs/SETUP.md).

## 6. Data-cleaning process (`churn/cleaning.py`)
Ordered, deterministic and logged (`outputs/logs/cleaning_log.csv`): trim/collapse whitespace and blank -> NaN; map
formatting variants (`yes`, `Month to month`) to canonical labels and null invalid categories; fix types (numeric coercion,
negative values); drop exact duplicates and repeated customer IDs; fix cross-column inconsistencies (add-ons recorded for
customers without internet); drop rows with an unusable target; flag IQR outliers (kept - they are real customers).
On the IBM file the only issue is 11 blank `TotalCharges` for customers with tenure 0 (set to 0: not yet billed).
**Statistical imputation is deliberately not done here** - median/most-frequent imputation, winsorising, scaling and
encoding are learned inside the scikit-learn pipeline on training data only, so nothing leaks from the test set.
The data-quality report (`reports/data_quality_report.md`) lists missing values, duplicates, invalid categories, dtypes,
cardinality, distributions and summary statistics.

## 7. EDA findings (full dataset)
* Overall churn **26.5%**; classes are imbalanced.
* **Contract:** month-to-month 42.7% vs two-year 2.8%.
* **Tenure:** 0-6 months 52.9% vs 49+ months 9.5%; churners' median tenure is 10 vs 38 months.
* **Internet:** fibre optic 41.9% vs DSL (about 19%) and no internet 7.4%.
* **Payment:** electronic check 45.3% vs 15-18% for automatic payments.
* **Support:** among internet customers, no tech support 41.6% vs 15.2% with it; same pattern for online security.
* **Senior citizens** 41.7% vs 23.6%; **gender** shows no meaningful difference (26.9% vs 26.2%).
* Churners pay more (median $79.65 vs $64.43 a month).
* **Multicollinearity:** tenure and total charges (r = 0.83); monthly charges, average monthly spend and number of services
  are strongly related (VIFs flagged in `eda_vif.csv`). A regularised linear model and tree models handle this.
Charts: `outputs/figures/eda_*.png` (univariate, bivariate, multivariate heatmaps, correlation, VIF, Cramer's V).

## 8. Statistical-testing approach (`churn/stats_tests.py`)
* **Chi-square** tests of churn vs every categorical variable, with **Cramer's V** effect sizes.
* **Numeric variables** (tenure, monthly/total charges, average spend, services): Shapiro-Wilk checks normality, then
  **Welch t-test** (normal) or **Mann-Whitney U** (otherwise; used for all here) with Cohen's d / rank-biserial r.
* Business comparisons with **two-proportion z-tests**, risk ratios and 95% CIs (e.g. month-to-month vs two-year: 15x risk).
* All p-values are **Benjamini-Hochberg adjusted** for multiple testing; correlations (Pearson/Spearman) are reported.
* Strongest associations: contract (V = 0.41), online security (0.35), tech support (0.34), internet service (0.32),
  payment method (0.30); gender is not significant. Tenure shows the largest numeric effect (r = 0.48).
* Reports explicitly state that these are **associations, not causes**; only an A/B retention test can show cause.

## 9. Feature engineering (`churn/features.py`)
Tenure bands, average monthly spend, current-bill-vs-history delta and bill-increase flag, total services, protection and
streaming counts, "internet without security/support" and "fibre without tech support" flags, month-to-month and contract-risk
score, electronic-check / auto-pay flags, billing-risk score, customer-value segment (fixed business thresholds, so no
fitting on data), and a service-profile combination. All engineered features are row-wise and deterministic (no leakage).
Preprocessing is a `ColumnTransformer`: continuous -> median impute -> **winsorise at training 1%/99% quantiles** -> scale;
discrete -> impute -> scale; categorical -> impute -> one-hot (unknown categories ignored). `gender` is excluded from the
model by default (fairness); change `features.exclude_from_model` in `config.yaml`.

## 10. ML methodology (`churn/modeling.py`)
* Stratified 80/20 split; the **test set is never used for fitting, tuning or selection**.
* 5-fold stratified CV on the training set; Grid search (Logistic Regression) and Randomized search (Random Forest,
  Gradient Boosting, XGBoost). The preprocessor sits inside the CV pipeline, so each fold fits preprocessing on its own
  training part only.
* Imbalance: class weights (LR, RF), `scale_pos_weight` (XGBoost), and **threshold tuning** (all models).
* **Selection:** highest CV ROC-AUC, with a parsimony rule - if Logistic Regression is within 0.003 of the best model it is
  preferred (simpler, interpretable). Here Gradient Boosting scored 0.850 vs 0.847, so Logistic Regression was selected.
* **Threshold:** tuned on *out-of-fold training predictions* to maximise F-beta (beta = 1.5, recall weighted above precision);
  configurable (`threshold.mode: fixed`, `--threshold`).
* Every run (model, params, date, dataset hash, CV metrics, feature importance) is stored in `outputs/models/model_runs.csv`
  and `outputs/models/runs/*.json`. The final model, fitted preprocessing pipeline, feature names and metrics are saved with joblib.
* Customer scores for the whole base are **out-of-fold** (training customers) or **hold-out** (test customers) - nobody is
  scored by a model that saw their own label.

## 11. Evaluation metrics (full IBM dataset)
Cross-validation on the training set (threshold 0.5 for F1/recall/precision; ROC-AUC/PR-AUC threshold-free):

| Model | CV ROC-AUC | CV PR-AUC | CV Recall | CV Precision | CV F1 |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression (selected)** | 0.847 | 0.659 | 0.793 | 0.521 | 0.629 |
| Random Forest | 0.848 | 0.661 | 0.773 | 0.534 | 0.632 |
| Gradient Boosting | 0.850 | 0.668 | 0.523 | 0.668 | 0.586 |
| XGBoost | 0.848 | 0.664 | 0.795 | 0.526 | 0.633 |

Untouched test set (1,409 customers) at the tuned threshold **0.45**:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity |
| --- | --- | --- | --- | --- | --- | --- |
| 0.725 | 0.490 | 0.829 | 0.616 | 0.845 | 0.651 | 0.688 |

Confusion matrix: 712 true negatives, 323 false positives, 64 false negatives, 310 true positives. The model catches 83% of
churners while 49% of the customers it flags truly churn (baseline churn rate: 27%). Lowering the threshold catches more
churners but wastes more offers; raising it does the reverse - see `eval_threshold_tradeoff.png`, the threshold table with
net benefit, and the sliders in the dashboard. Note: Gradient Boosting's low recall at 0.5 is only because it has no class
weights; after threshold tuning all four models are comparable (they score within 0.003 ROC-AUC of each other).

## 12. Explainability (`churn/explain.py`)
* **Global:** grouped permutation importance on the test set (related columns shuffled together so credit is not split),
  feature-level permutation importance, model-native importance and (if installed) **SHAP** summary/bar plots.
* **Local:** for any customer, each business driver's effect = predicted probability minus the average probability when that
  driver's values are replaced by those of random training customers (percentage points). It works for every model type; SHAP
  values are available as a second view. Explanations are computed from the model, not hand-written.
* Top drivers on the IBM data: **tenure** (test AUC drop 0.069), **contract type** (0.038), **security & tech support**,
  **payment & billing**, **service mix**. New customers (0-6 months) average +20 pp risk vs a typical customer; two-year
  contracts about -12 pp. These are model associations, not proven causes.

## 13. Business recommendations (`churn/insights.py`)
Rule-based engine (no external AI) over contract, tenure, price, payment method, services, senior status and predicted
risk tier. Nine rules (e.g. offer a 12-month contract to new month-to-month customers, auto-pay bill credit for electronic-check
payers, free tech-support/security trial for internet customers without them, fibre price/quality review, 30-day onboarding
call). Each flagged customer gets a primary action plus secondary ones; segments (contract x tenure band) get an urgency level.
Results (threshold 0.45): **3,124 customers flagged (44%)**, **$235K monthly / $2.82M annual revenue at risk**
(52% of monthly revenue), $135K/month with the 1,694 "Critical" customers. Month-to-month customers hold 94% of the
flagged revenue; the most urgent segment is month-to-month customers in their first six months (85% flagged).
A campaign calculator (save rate 30%, $40/offer - both configurable assumptions) estimates about $304K annual net benefit.
**Priority score** (0-100) combines churn probability (50%), revenue (30%), early tenure (10%) and number of risk factors (10%).
Outputs: `customer_recommendations.csv`, `priority_customers_top500.csv`, `insights_segments.csv`,
`reports/business_insights_report.md`, and `final_report.{md,html,pdf}` (print the HTML to PDF if you prefer).

## 14. Dashboard (`app.py`) - 12 pages (+ Dataset Analysis for uploads)
Data Upload, Executive Summary, Data Quality, Exploratory Analysis, Statistical Insights, Customer Explorer, Churn Prediction,
Prediction History, Model Performance, Feature Importance, Retention Recommendations, Revenue at Risk. KPI cards, interactive
filters, search/sort/pagination, CSV downloads, an individual prediction form with explanation and recommendations, threshold
sliders and a what-if campaign calculator.

**Bring your own data.** On the *Data Upload* page any visitor can upload a CSV/Excel file. **Any dataset with a yes/no outcome works**: pick
the outcome column and the app profiles, tests, models, explains and scores it (page *Dataset Analysis*). Files in the IBM Telco layout
additionally unlock the full telecom dashboard: **(A)** train on it, or **(B)** score customers with the loaded model in seconds. Each browser session
gets a private temporary workspace, so different users never see each other's files. See [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)
for every action and every page explained in detail.

## 15. Local setup
```bash
python -m venv venv                         # in the folder containing churn_ds/
source venv/bin/activate                    # Windows: venv\Scripts\activate
cd churn_ds
pip install -r requirements.txt            # pinned versions (Python 3.11-3.13); older Python: requirements-flexible.txt
python run_pipeline.py seed                 # download data (offline fallback: bundled sample)
python run_pipeline.py all                  # OPTIONAL: rebuilds the bundled demo (about 5 minutes; add --fast for ~1 min)
streamlit run app.py                        # dashboard
python -m pytest                            # 120 tests
```
The zip already contains a pre-built demo (full IBM data, trained model, reports), so `streamlit run app.py` works immediately.
`make` shortcuts (Linux/macOS/WSL): `make install seed all app test quick`. More detail and troubleshooting: [`docs/SETUP.md`](docs/SETUP.md).

## 16. Project structure
```
churn_ds/
  app.py  run_pipeline.py  config.yaml  requirements.txt  Makefile  pytest.ini  requirements-flexible.txt
  churn/    schema config utils data_loader upload cleaning quality eda stats_tests features modeling
            predict evaluation explain insights reporting pipeline dashboard_helpers generic
  data/     raw/  processed/  sample/telco_churn_sample.csv  DATA_DICTIONARY.md
  outputs/  figures/ reports/ tables/ models/ logs/      (generated)
  tests/    14 test modules + data/mini_telco.csv
  docs/     SETUP.md  WALKTHROUGH.md  DEPLOYMENT.md  screenshots/
```

## 17. Limitations and future improvements
* One snapshot of 7,043 customers; no usage, complaints, or time series. Revenue = monthly charges (no margin / CLV).
* Save rate and offer cost are assumptions; validate with randomised retention experiments (uplift modelling would target
  customers who respond to offers, not just those likely to churn).
* Add probability calibration, drift monitoring and scheduled retraining; a survival model for time-to-churn;
  fairness audits across more groups; batch scoring API; CLV-based prioritisation; CI workflow running `pytest`.

## License / data
Code: MIT-style, use freely. Data: IBM Telco Customer Churn sample dataset (Apache-2.0, IBM).
