# Setup, data replacement and troubleshooting

## 1. Environment (virtual environment lives OUTSIDE the project folder)
```bash
python -m venv venv                     # run in the folder that CONTAINS churn_ds/
source venv/bin/activate                # Windows PowerShell: venv\Scripts\Activate.ps1   (cmd: venv\Scripts\activate.bat)
cd churn_ds
pip install -r requirements.txt
```
Python 3.10+ recommended. No database server, Docker, Node.js or API key is needed.

## 2. Run
```bash
python run_pipeline.py seed             # download IBM Telco data (falls back to bundled sample if offline)
python run_pipeline.py all              # clean -> quality -> EDA -> stats -> train -> evaluate -> explain -> insights -> report
streamlit run app.py                    # dashboard on http://localhost:8501
python -m pytest                        # automated tests
```
Quick demo (about a minute): `python run_pipeline.py all --sample --fast --force`.

Single steps: `clean`, `eda`, `stats`, `train`, `evaluate`, `explain`, `insights`, `report`
(each prints what it wrote; steps depend on earlier ones, so run them in this order the first time).

## 3. Replacing the dataset
* **Full IBM dataset (default):** `python run_pipeline.py seed --force` re-downloads it.
* **Offline:** download `Telco-Customer-Churn.csv` manually (URL in `config.yaml` -> `data.download_url`), save it as
  `data/raw/telco_churn.csv`, then run `python run_pipeline.py all`.
* **Your own telco data:** it must have the same 21 columns (names and meaning in `data/DATA_DICTIONARY.md`).
  Save it as `data/raw/telco_churn.csv` and run `python run_pipeline.py all`. Invalid categories, blanks and
  duplicates are handled and logged by the cleaning step; a missing required column stops the run with a clear message.
* After replacing data always re-run `all` so the model, scores and reports match the new data.
  The dataset version (a short file hash) is stored with every model run in `outputs/models/model_runs.csv`.

## 4. Configuration (`config.yaml`)
Test split, CV folds, search iterations, threshold mode (`tuned` or `fixed`) and beta, retention offer cost, save rate,
priority weights, recommendation thresholds, excluded model features (default: `gender`). Override on the command line,
e.g. `python run_pipeline.py evaluate --threshold 0.40`.

## 5. Troubleshooting
| Symptom | Fix |
| --- | --- |
| `xgboost` / `shap` fails to install | Delete those two lines from requirements.txt. The project runs without them (LightGBM/boosted model skipped; permutation importance replaces SHAP). |
| Dashboard says artifacts are missing | Run `python run_pipeline.py all`, or press the "Run pipeline now" button in the app. |
| Model fails to load after upgrading scikit-learn | Models are pickles tied to library versions: re-run `python run_pipeline.py train` (the dashboard also offers a rebuild button). |
| Download blocked by a firewall | See "Offline" above; the bundled sample is used automatically. |
| `ModuleNotFoundError: churn` | Run commands from inside the `churn_ds` folder. |
