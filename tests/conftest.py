"""Shared fixtures. The `pipeline_run` fixture runs the real pipeline once (fast settings) on the
500-row sample in tests/data and writes everything into a temporary directory."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from churn.config import load_config
from churn.pipeline import run_step

DATA = Path(__file__).parent / "data" / "mini_telco.csv"


@pytest.fixture(scope="session")
def mini_raw() -> pd.DataFrame:
    return pd.read_csv(DATA)


def _make_cfg(root: Path) -> dict:
    p = lambda *a: str(root.joinpath(*a))  # noqa: E731
    return load_config(overrides={
        "paths": {"raw_data": p("data", "raw", "telco_churn.csv"), "sample_data": str(DATA),
                  "clean_data": p("data", "processed", "clean.csv"),
                  "modeling_data": p("data", "processed", "modeling.csv"), "models_dir": p("out", "models"),
                  "figures_dir": p("out", "figures"), "reports_dir": p("out", "reports"),
                  "tables_dir": p("out", "tables"), "logs_dir": p("out", "logs"),
                  "history_file": p("out", "history.csv")},
        "modeling": {"fast": True, "cv_folds": 3, "n_jobs": 1},
        "explain": {"permutation_repeats": 3, "shap_sample_size": 60, "background_size": 40}})


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("churn_run")
    cfg = _make_cfg(root)
    run_step("seed", cfg, sample=True)
    for step in ["clean", "eda", "stats", "train", "evaluate", "explain", "insights", "report"]:
        run_step(step, cfg)
    cfg_file = root / "config_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"cfg": cfg, "root": root, "cfg_file": cfg_file}


@pytest.fixture(scope="session")
def cfg(pipeline_run):
    return pipeline_run["cfg"]


@pytest.fixture()
def blank_cfg(tmp_path):
    return _make_cfg(tmp_path)


@pytest.fixture(scope="session")
def other_churn_df() -> pd.DataFrame:
    """A churn table with a completely different layout (no telecom columns)."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({"CustomerID": np.arange(1, n + 1), "Age": rng.integers(18, 65, n),
                       "Gender": rng.choice(["Male", "Female"], n), "Tenure": rng.integers(1, 60, n),
                       "Support Calls": rng.integers(0, 10, n), "Payment Delay": rng.integers(0, 30, n),
                       "Subscription Type": rng.choice(["Basic", "Standard", "Premium"], n),
                       "Contract Length": rng.choice(["Monthly", "Quarterly", "Annual"], n),
                       "Total Spend": rng.uniform(100, 1000, n).round(2)})
    z = 0.35 * df["Support Calls"] + 0.05 * df["Payment Delay"] + (df["Contract Length"] == "Monthly") * 1.2 - 0.03 * df["Tenure"] - 1.8
    df["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df.loc[3, "Gender"] = "  Male "           # whitespace
    df.loc[5, "Age"] = None                    # missing
    return df


@pytest.fixture(scope="session")
def generic_run(other_churn_df):
    from churn.generic import run_generic_analysis, suggest_settings
    from churn.upload import TRAINING_PROFILES
    s = suggest_settings(other_churn_df)
    return run_generic_analysis(other_churn_df, s, profile=list(TRAINING_PROFILES)[0], source_name="other.csv")
