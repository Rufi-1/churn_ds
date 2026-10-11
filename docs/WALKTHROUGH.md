# Walkthrough: every action and every page, explained

This document explains what happens when you run the project, step by step, and what you see on every dashboard page.
Numbers quoted come from the full IBM Telco dataset (7,043 customers, 26.5% churn).

---------------------------------------------------------------------------------------------------------------------

## PART 1 - The pipeline: what each action does

`python run_pipeline.py all` runs nine actions in order. Each can also be run alone.

### Action 1 - `seed` (data collection)
* Looks for `data/raw/telco_churn.csv`. If missing it downloads the IBM Telco Customer Churn CSV from the URL in
  `config.yaml`; if the download fails (offline/firewall) it copies the bundled 1,500-row sample instead.
* Checks the downloaded file really is a churn dataset (has `customerID` and `Churn` columns) before accepting it.
* Writes `data/raw/dataset_meta.json` (source, row count, a short SHA-256 hash). The hash becomes the **dataset version**
  stored with every trained model, so you can always tell which data a model was trained on.
* The raw file is never modified afterwards.

### Action 2 - `clean` (data cleaning, data-quality assessment, modeling-ready data)
Performed by `cleaning.py`, `quality.py` and `features.py`. Cleaning steps run in a fixed order and every change is recorded
in `outputs/logs/cleaning_log.csv`:
1. **Check required columns** - stops with a clear message if any of the 21 expected columns is missing.
2. **Whitespace** - trims spaces, collapses repeated spaces, turns blank strings into missing values (the IBM file has 11 blank
   `TotalCharges` cells that pandas reads as text, which would otherwise break the numeric column).
3. **Categories** - maps spelling/case variants (`yes`, `Month to month`) to the canonical label; `SeniorCitizen` 0/1 becomes
   No/Yes; values that are not valid categories become missing and are logged as "invalid category".
4. **Data types** - numeric columns are converted; non-numeric text and negative values become missing; tenure is rounded.
5. **Logical repairs** - a customer with tenure 0 has not been billed, so a blank `TotalCharges` becomes 0; otherwise it is
   derived as tenure x monthly charge.
6. **Duplicates** - exact duplicate rows and repeated customer IDs are dropped (first kept).
7. **Consistency** - e.g. a customer with no internet cannot have OnlineSecurity = Yes; fixes are logged.
8. **Missing target** - rows with no usable churn label are dropped (a label cannot be guessed).
9. **Outliers** - values outside the 1.5xIQR (Tukey) fences are *flagged* in the log, not deleted, because they are real customers.
10. **Target flag** - adds `ChurnFlag` (1 = churned).

Important design choice: cleaning does **not** fill remaining gaps with averages. Average-based imputation is learned later
inside the model pipeline using training data only, which prevents data leakage from the test set.

The **data-quality report** (`outputs/reports/data_quality_report.md` + CSV tables) summarises: rows/columns, duplicates,
missing cells (including blank strings), invalid categories, raw vs clean data types, unique-value counts, numeric statistics
(mean, std, quartiles, skew, outlier counts) and category distributions.

Then **feature engineering** creates the modeling-ready table `data/processed/telco_modeling.csv` (un-scaled): tenure band,
average monthly spend, bill-vs-history difference, bill-increase flag, number of services, protection and streaming counts,
"internet without security/support" flag, "fibre without tech support" flag, month-to-month flag, contract-risk score,
electronic-check and auto-pay flags, billing-risk score, customer-value segment and service profile. These are computed row by
row with fixed rules, so they cannot leak information between customers.

### Action 3 - `eda` (exploratory data analysis)
Creates 20 charts in `outputs/figures/eda_*.png` and a summary in `outputs/reports/eda_summary.md`:
* **Univariate**: class balance, histograms of tenure/monthly/total charges/average spend/services, counts of every category.
* **Bivariate**: churn rate by contract, tenure band, internet service, payment method, senior citizen, gender, tech support and
  online security (each bar labelled with the churn rate and customer count, with the overall rate as a reference line);
  box plots and density plots of numeric variables for churned vs retained customers.
* **Multivariate**: tenure vs monthly-charge scatter coloured by churn; correlation heatmap; **variance inflation factor (VIF)**
  chart to detect multicollinearity; Cramer's V ranking of how strongly each category relates to churn; heatmaps of churn rate
  for contract x internet service, tenure band x contract and payment method x contract.
* The written insights are generated from the data (for example "month-to-month customers churn at 42.7% vs 2.8% for two-year").

### Action 4 - `stats` (hypothesis testing)
* **Chi-square test** for each categorical variable vs churn, with **Cramer's V** as effect size (negligible/small/moderate/large).
* **Numeric variables**: Shapiro-Wilk checks normality; if both groups are normal a Welch t-test is used, otherwise
  Mann-Whitney U. Effect sizes: Cohen's d and rank-biserial correlation. Medians of churned vs retained customers are reported.
* **Group comparisons** with two-proportion z-tests: month-to-month vs two-year, fibre vs DSL, electronic check vs others, seniors
  vs non-seniors, no tech support vs tech support, first-year vs longer customers, paperless vs paper billing, male vs female.
  Each reports churn rates, the difference with a 95% confidence interval and the risk ratio.
* **Multiple-testing correction** (Benjamini-Hochberg) is applied so that running many tests does not produce false "discoveries".
* **Correlations**: Pearson and Spearman among numeric variables and with churn.
* Every report states clearly that these are **associations, not causation**.

### Action 5 - `train` (machine learning)
1. **Stratified split**: 80% train / 20% test, keeping the churn rate equal in both. The test set is locked away: it is never
   used for fitting, tuning or choosing a model. The split IDs are saved in `outputs/models/split.json`.
2. **Preprocessing pipeline** (scikit-learn `ColumnTransformer`): continuous features -> median imputation -> winsorising at
   the training 1st/99th percentile -> standard scaling; discrete features -> imputation -> scaling; categories -> imputation ->
   one-hot encoding (unknown categories ignored). It sits *inside* every cross-validation fold so each fold learns its statistics
   only from its own training part (no leakage). `gender` is excluded from the model by default as a fairness precaution.
3. **Four candidate models** are tuned with 5-fold stratified cross-validation: Logistic Regression (grid search), Random Forest,
   Gradient Boosting and XGBoost (randomised search). Class imbalance is handled with class weights (LR, RF), `scale_pos_weight`
   (XGBoost) and threshold tuning (all).
4. **Model selection**: highest cross-validated ROC-AUC, with a parsimony rule - if Logistic Regression is within 0.003, the
   simpler, more explainable model wins. The reason is written to the log and the report. (On IBM data: Logistic Regression.)
5. **Threshold tuning**: out-of-fold predictions on the training set are used to pick the probability cut-off that maximises an
   F-beta score (beta = 1.5, so catching churners counts more than avoiding extra offers). Result: 0.45. You can override it.
6. **Scoring everyone honestly**: every customer gets a churn probability - out-of-fold for training customers and hold-out for
   test customers - so no customer is ever scored by a model that saw their own label (`customer_scores.csv`).
7. **Saving**: `churn_model.joblib` (fitted pipeline, preprocessor, feature names, threshold, metrics, parameters, training date,
   dataset version, library versions), plus `preprocessor.joblib`, `feature_names.json`, `metrics.json`, and a **run registry**
   (`model_runs.csv` and one JSON per model with parameters, CV metrics and feature importance).

### Action 6 - `evaluate` (model evaluation on the untouched test set)
Computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, balanced accuracy, Brier score, the confusion matrix and
a classification report; draws ROC curve, precision-recall curve, confusion matrix, threshold trade-off (precision/recall/F1 and
campaign net benefit vs threshold), score distribution, calibration curve and cumulative gains; writes a threshold table showing, for
thresholds from 0.05 to 0.95, how many customers are flagged, precision, recall and the estimated net benefit. The report explains
the precision/recall trade-off in plain language.

### Action 7 - `explain` (explainability)
* **Grouped permutation importance**: shuffle a business driver's columns (e.g. Contract + month-to-month flag + contract-risk
  score together) in the test set and measure how much ROC-AUC drops. Bigger drop = the model relies on it more.
* **Feature-level permutation importance**, **model-native importance** (coefficients/feature importances) and, if installed,
  **SHAP** summary and bar charts.
* **Driver effects**: for a sample of customers, how each driver moves the predicted probability versus a typical customer
  (e.g. new customers about +20 percentage points; two-year contracts about -12 points).
* Used again for single-customer explanations on the Churn Prediction page. Nothing is hand-written: every statement is computed
  from the fitted model.

### Action 8 - `insights` (business interpretation)
* **Risk tiers**: Low, Medium (>= half the threshold), High (>= threshold), Critical (>= 70% probability).
* **Rule-based recommendations** (nine rules, no external AI) over contract, tenure, price, payment method, services, senior
  status and risk tier. Each triggered rule adds an action; the highest-impact rule becomes the *primary action*. Low-risk customers
  get "No action - monitor".
* **Revenue at risk** = monthly charges of flagged customers (x12 for annual); also probability-weighted expected loss, the revenue
  held by Critical customers, and an illustrative campaign cost / benefit using the save-rate and offer-cost assumptions in config.
* **Priority score (0-100)** = 50% churn probability + 30% revenue rank + 10% early tenure + 10% number of risk factors.
* **Segments** = contract x tenure band, each with customers, flagged share, revenue at risk, top action and an urgency label
  (Urgent / High / Monitor).

### Action 9 - `report`
Builds `final_report.md`, a print-friendly `final_report.html` (charts embedded) and a `final_report.pdf` (KPI page, model
comparison, segments, actions and charts) combining business insights with the quality, EDA, statistics, evaluation and
explainability reports.

---------------------------------------------------------------------------------------------------------------------

## PART 2 - The dashboard, page by page

The sidebar lists the pages and shows which data you are looking at: the built-in IBM demo data, or **your own uploaded file**
(green box) with a button to switch back to the demo.

### 1. Data Upload
Purpose: let anyone use the app with their own data. After a file is read, the page tells you how many of the 21 telecom columns it
recognised and offers two ways to analyse it:
* **Any dataset** (default unless the file matches the telecom layout) - works with any table that has a yes/no outcome column.
  You choose the outcome column and which value means "churned/positive" (auto-suggested), optionally a customer-ID column, an optional
  numeric **value column** (such as spend) used to estimate value at risk, and any columns to ignore. The app cleans and profiles the
  data, runs hypothesis tests, tunes four models, evaluates on a hold-out set, explains the drivers, scores every row and writes a
  report. Files over 20,000 rows are sampled (stratified) for training; up to 100,000 rows are scored. The result appears on the
  **Dataset Analysis** page (below), and the sidebar shows the telecom pages are replaced by it until you switch back to the demo.
* **Telecom layout** - the 21 IBM Telco columns; gives the full 11-page telecom dashboard with recommendations and revenue at risk
  (described below).

Telecom-layout details:
* **Required-columns table** and a downloadable template CSV.
* **File uploader** (CSV or Excel, up to 30 MB). A preview of the first rows is shown.
* **Automatic column mapping**: names are matched ignoring case, spaces and underscores, plus common aliases
  (`monthly_charges` -> `MonthlyCharges`, `churned` -> `Churn`). Anything not matched can be mapped by hand.
* **Live validation** before anything runs: missing columns, columns that do not look like the expected categories (probably mapped
  wrongly), non-numeric numbers, duplicate IDs, too few rows (training needs at least 300 rows and 30 customers of each outcome),
  and a row cap (20,000 for training, 100,000 for scoring; larger files are sampled with a warning).
* **Option A - analyse my data and train a new model**: choose a depth (Quick ~1-2 min, Balanced ~3-6 min, Thorough 10+ min) and
  run. A progress bar shows each pipeline step. The result is a private workspace for you; every other page then shows *your*
  results, including a model trained on your data.
* **Option B - score my customers with the loaded model**: no training, takes seconds. Works even if your file has no Churn column.
  Shows KPIs, risk-tier chart, top actions, a searchable/sortable/paginated table and downloads (all scored customers; flagged
  priority list). If your file does contain churn labels, it also reports how well the model performed on your data.
* Privacy: each browser session gets a random private folder; other visitors cannot see it; it is deleted after about 6 hours and
  when the server restarts. Do not upload personal data.

### 1b. Dataset Analysis (any-dataset mode)
Nine tabs: **Overview** (KPIs, strongest findings, top drivers, value at risk if you chose a value column, warnings for suspiciously
perfect models = possible data leakage, or for flagging too many rows); **Data Quality** (column profile with roles, numeric summary,
category distributions, cleaning log); **Exploration** (outcome rate by any category, numeric distributions by outcome, correlation
matrix, two-way heatmap); **Statistics** (chi-square and t-test/Mann-Whitney results with effect sizes and adjusted p-values);
**Model Performance** (metrics, live threshold slider, ROC/PR/confusion/trade-off/calibration/gains charts, model comparison);
**Drivers** (permutation importance and how each value moves the probability); **Customers** (every scored row with risk tier,
priority score and its top three risk drivers for the 200 highest-priority rows; filter, search, sort, paginate, download);
**Predict one customer** (a form built from your own columns, with explanation); **Report & downloads** (Markdown report, scored CSV,
cleaning log). Cleaning in this mode: trims text, converts numbers stored as text, drops duplicates and rows with no outcome, drops
constant and ID-like/free-text columns, and leaves remaining gaps for the training pipeline to impute (no leakage).

### 2. Executive Summary
KPI cards: customers, historical churn rate, flagged customers, monthly and annual revenue at risk, model name, ROC-AUC, recall,
precision, critical-risk customers. Charts: customers by risk tier; annual revenue at risk by contract. Then the key data insights
and the top retention priorities, plus a button to download the PDF report.

### 3. Data Quality
Cards for raw rows, clean rows, duplicates and missing cells before/after cleaning. Tabs: missing values, data types and cardinality,
invalid categories (a green "nothing found" message if clean), numeric summary, category distributions (choose a column), outliers
and the **cleaning log** (every change made and how many rows it touched). Each table can be downloaded; the full written report is
in an expander.

### 4. Exploratory Analysis
* *Churn by segment*: pick any dimension (contract, tenure band, internet service, payment method, ...) to see the churn rate per
  group against the overall rate, with a data table.
* *Numeric distributions*: pick a variable to see histogram and box plot split by churn, plus statistics.
* *Multivariate*: choose two variables for a churn-rate heatmap; correlation matrix and VIF table for multicollinearity.
* *Saved charts*: all 20 EDA charts.

### 5. Statistical Insights
A banner reminds you that results are associations, not causes. Tabs: chi-square results (with Cramer's V and highest/lowest
churn groups), numeric tests (test used, effect size, medians), group comparisons (rates, difference, risk ratio) and correlations.
Tick "only significant" to hide non-significant rows. Plain-English interpretations sit under each table; results can be downloaded.

### 6. Customer Explorer
Filter customers by contract, internet service, risk tier, payment method, tenure range, monthly-charge range or "flagged only";
KPI cards update with the filter. Search by ID/contract/action, sort by any column, choose rows per page and page number, download the
filtered list as CSV, and open a customer's detail (probability, tier, priority rank, recommended actions, risk factors).

### 7. Churn Prediction
Enter one customer's details in the form (leave total charges at 0 to estimate it). You get: churn probability, risk tier badge,
the decision at your chosen threshold (slider), revenue at risk, a bar chart of **why** (which drivers raise or lower risk in
percentage points versus a typical customer) with sentences, an optional SHAP table, and the **recommended retention actions**.
Every prediction is saved to Prediction History. Inconsistent entries (add-ons without internet) are corrected automatically.

### 8. Prediction History
Every prediction made on the Churn Prediction page: KPIs (count, flagged, average probability), searchable/sortable/paginated table,
CSV download, a trend chart and a "clear history" button. History belongs to the current workspace.

### 9. Model Performance
Final model, training date, dataset version and test size; metric cards (ROC-AUC, PR-AUC, accuracy, precision, recall, F1); an
explanation of why these metrics matter; the model-selection rationale; a **threshold slider** that recomputes precision, recall,
F1, customers flagged, churners missed and the confusion matrix live; ROC, precision-recall, calibration, threshold-trade-off and
cumulative-gains charts; cross-validated comparison of all four models; run registry; classification report; threshold table with
campaign net benefit.

### 10. Feature Importance
Business-driver importance chart and table; a driver selector showing how each value (for example each contract type) moves risk;
tabs for feature-level permutation importance, model-native importance and SHAP; and "Explain a specific customer" by ID.

### 11. Retention Recommendations
The action playbook (each rule, its trigger, impact, how many flagged customers and how much monthly revenue it covers); the
segment-level plan (contract x tenure band with urgency, filterable); and customer-level recommendations filtered by risk tier and
action with search, sort, pagination and CSV download.

### 12. Revenue at Risk
KPIs for monthly/annual revenue at risk, share of revenue, expected loss and critical-tier revenue. A **what-if campaign calculator**
(threshold, save rate, cost per offer) shows customers contacted, revenue at risk, expected revenue saved, cost and net benefit.
Break revenue at risk down by contract, tenure band, internet service, value segment or risk tier; priority customer list with
search/sort/pagination and downloads.

---------------------------------------------------------------------------------------------------------------------

## PART 3 - Reading the results responsibly
* A high probability means "similar customers churned a lot", not that the customer will certainly leave.
* Associations and drivers are not causes; test retention offers with an A/B experiment before scaling them.
* Save rate ($ per offer, 30%) and offer cost are assumptions in `config.yaml` - replace them with your own numbers.
* Re-train regularly: customer behaviour drifts. Compare the dataset version stored with each model.
