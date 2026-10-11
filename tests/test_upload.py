import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import resolve_path
from churn.predict import load_bundle
from churn.upload import (PIPELINE_LOCK, TRAINING_PROFILES, apply_mapping, auto_map_columns, cleanup_old_workspaces,
                          create_workspace, prepare_upload, run_workspace_pipeline, score_batch)


@pytest.fixture()
def renamed(mini_raw):
    return mini_raw.rename(columns={"tenure": "Tenure Months", "MonthlyCharges": "monthly_charges", "Churn": "churned",
                                    "customerID": "cust_id", "Contract": "CONTRACT"})


def test_auto_mapping_handles_naming_variants(renamed):
    m = auto_map_columns(renamed.columns)
    assert m["tenure"] == "Tenure Months" and m["MonthlyCharges"] == "monthly_charges" and m["Churn"] == "churned"
    assert m["Contract"] == "CONTRACT" and m["customerID"] == "cust_id" and m["gender"] == "gender"
    assert len({v for v in m.values() if v}) == len([v for v in m.values() if v])  # no source used twice


def test_prepare_upload_ok_and_schema(renamed):
    d, rep = prepare_upload(renamed, auto_map_columns(renamed.columns), "train")
    assert rep["ok"] and d is not None and "Churn" in d.columns and len(d) == 500
    assert rep["class_counts"]["Yes"] > 100


def test_missing_columns_reported(mini_raw):
    x = mini_raw.drop(columns=["Contract", "TechSupport"])
    d, rep = prepare_upload(x, auto_map_columns(x.columns), "score")
    assert d is None and not rep["ok"] and "Contract" in rep["errors"][0] and "TechSupport" in rep["errors"][0]


def test_score_mode_does_not_need_label_or_id(mini_raw):
    x = mini_raw.drop(columns=["Churn", "customerID"])
    d, rep = prepare_upload(x, auto_map_columns(x.columns), "score")
    assert rep["ok"] and d["customerID"].iloc[0] == "ROW-00001" and any("customer ID" in w for w in rep["warnings"])


def test_training_rejects_tiny_or_one_class_files(mini_raw):
    small = mini_raw.head(100)
    assert not prepare_upload(small, auto_map_columns(small.columns), "train")[1]["ok"]
    one = mini_raw.copy(); one["Churn"] = "No"
    assert any("both outcomes" in e for e in prepare_upload(one, auto_map_columns(one.columns), "train")[1]["errors"])


def test_wrong_mapping_detected(mini_raw):
    m = auto_map_columns(mini_raw.columns)
    m["Contract"] = "gender"       # user maps the wrong column
    d, rep = prepare_upload(mini_raw, m, "score")
    assert d is None and any("Contract" in e for e in rep["errors"])


def test_label_variants_accepted(mini_raw):
    x = mini_raw.copy(); x["Churn"] = (x["Churn"] == "Yes").astype(int)
    assert prepare_upload(x, auto_map_columns(x.columns), "train")[1]["ok"]


def test_large_training_file_is_downsampled(mini_raw, monkeypatch):
    import churn.upload as up
    monkeypatch.setattr(up, "MAX_TRAIN_ROWS", 400)
    d, rep = up.prepare_upload(mini_raw, auto_map_columns(mini_raw.columns), "train")
    assert rep["ok"] and len(d) == 400 and any("limited" in w for w in rep["warnings"])


def test_workspaces_are_private_and_cleaned(tmp_path, monkeypatch):
    import churn.upload as up
    monkeypatch.setattr(up, "WORKSPACE_ROOT", tmp_path / "ws")
    cfg1, ws1 = create_workspace("Quick (about 1-2 minutes, tiny search)")
    cfg2, ws2 = create_workspace()
    assert ws1 != ws2 and str(ws1) in str(resolve_path(cfg1, "models_dir")) and str(ws2) in str(resolve_path(cfg2, "tables_dir"))
    assert cfg1["modeling"]["fast"] is True and cfg2["modeling"]["fast"] is False
    import os, time
    old = time.time() - 10 * 3600
    os.utime(ws1, (old, old))
    assert cleanup_old_workspaces(6) == 1 and not ws1.exists() and ws2.exists()


def test_workspace_pipeline_on_uploaded_file_is_isolated(renamed, cfg, tmp_path):
    prepared, rep = prepare_upload(renamed, auto_map_columns(renamed.columns), "train")
    ws_cfg, ws = create_workspace("Quick (about 1-2 minutes, tiny search)", cfg, root=tmp_path / "mine")
    steps = []
    timings = run_workspace_pipeline(ws_cfg, prepared, lambda s, i, n: steps.append(s))
    assert steps == ["clean", "eda", "stats", "train", "evaluate", "explain", "insights", "report"] and set(timings) == set(steps)
    assert (resolve_path(ws_cfg, "models_dir") / "churn_model.joblib").exists()
    assert (resolve_path(ws_cfg, "reports_dir") / "final_report.pdf").exists()
    assert not PIPELINE_LOCK.locked()
    # the original (demo) workspace is untouched
    assert str(ws) not in str(resolve_path(cfg, "models_dir"))
    ws_model = load_bundle(ws_cfg)
    assert ws_model["n_train"] == 400 and ws_model["dataset_version"] != "unknown"


def test_batch_scoring(cfg, mini_raw):
    b = load_bundle(cfg)
    x = mini_raw.drop(columns=["Churn"])
    prepared, _ = prepare_upload(x, auto_map_columns(x.columns), "score")
    table, rev = score_batch(prepared, b, cfg)
    assert len(table) == 500 and table["churn_probability"].between(0, 1).all()
    assert rev["flagged_customers"] == int(table["flagged"].sum())
    assert rev["monthly_revenue_at_risk"] == pytest.approx(table.loc[table["flagged"], "MonthlyCharges"].sum())
    assert (table["Churn"] == "Unknown").all() and table["priority_rank"].is_unique
    # a higher threshold flags fewer customers
    t2, r2 = score_batch(prepared, b, cfg, threshold=0.9)
    assert r2["flagged_customers"] <= rev["flagged_customers"]
