# Convenience wrapper around run_pipeline.py (Linux/macOS/WSL/Git-Bash). On plain Windows use the
# `python run_pipeline.py ...` commands from the README instead.
PY ?= python

.PHONY: install seed clean-data eda stats train evaluate explain insights report all quick app test lint-clean reset

install:        ## install dependencies into the active virtual environment
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt

seed:           ## download the IBM Telco dataset (falls back to the bundled sample if offline)
	$(PY) run_pipeline.py seed

clean-data:     ## clean data, write data-quality report + modeling-ready dataset
	$(PY) run_pipeline.py clean

eda:            ## exploratory analysis + charts
	$(PY) run_pipeline.py eda

stats:          ## hypothesis tests
	$(PY) run_pipeline.py stats

train:          ## tune, compare, select and save the model (also writes customer scores)
	$(PY) run_pipeline.py train

evaluate:       ## test-set evaluation report and charts
	$(PY) run_pipeline.py evaluate

explain:        ## permutation importance / SHAP / business explanations
	$(PY) run_pipeline.py explain

insights:       ## revenue at risk, recommendations, priorities
	$(PY) run_pipeline.py insights

report:         ## final Markdown / HTML / PDF reports
	$(PY) run_pipeline.py report

all:            ## the complete pipeline, raw data -> final report
	$(PY) run_pipeline.py all

quick:          ## complete pipeline on the bundled sample with tiny grids (about a minute)
	$(PY) run_pipeline.py all --sample --fast --force

app:            ## launch the Streamlit dashboard
	$(PY) -m streamlit run app.py

test:           ## run the automated tests
	$(PY) -m pytest

reset:          ## delete generated data/outputs (keeps code, config and the bundled sample)
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['outputs','data/processed']]; pathlib.Path('data/raw/telco_churn.csv').unlink(missing_ok=True); [pathlib.Path(d).mkdir(parents=True, exist_ok=True) for d in ['outputs','data/processed']]"
