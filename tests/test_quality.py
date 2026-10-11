import pandas as pd

from churn.cleaning import clean_dataset
from churn.config import load_config
from churn.quality import (categorical_distribution, invalid_category_table, missing_values_table, numeric_summary,
                           run_quality_checks)


def test_missing_counts_blank_strings():
    df = pd.DataFrame({"a": ["x", " ", None, "y"]})
    t = missing_values_table(df).set_index("column")
    assert t.loc["a", "missing"] == 2 and t.loc["a", "blank_strings"] == 1


def test_invalid_categories_detected():
    df = pd.DataFrame({"Contract": ["Month-to-month", "Weekly"], "SeniorCitizen": [0, 1]})
    t = invalid_category_table(df)
    assert list(t["invalid_value"]) == ["Weekly"]  # 0/1 are valid raw codes for SeniorCitizen


def test_quality_report_structure(mini_raw):
    cfg = load_config()
    clean, log = clean_dataset(mini_raw, cfg)
    rep = run_quality_checks(mini_raw, clean, log, cfg, save=False)
    assert rep["overview"]["raw_rows"] == len(mini_raw)
    assert rep["overview"]["missing_cells_clean"] == 0
    assert {"quality_dtypes", "quality_missing_raw", "quality_numeric_summary", "cleaning_log"} <= set(rep["tables"])
    assert len(numeric_summary(clean)) == 3
    assert categorical_distribution(clean)["pct"].between(0, 100).all()


def test_quality_files_written(pipeline_run):
    from churn.config import resolve_path
    cfg = pipeline_run["cfg"]
    assert (resolve_path(cfg, "reports_dir") / "data_quality_report.md").exists()
    assert (resolve_path(cfg, "logs_dir") / "cleaning_log.csv").exists()
    assert (resolve_path(cfg, "tables_dir") / "quality_overview.json").exists()
