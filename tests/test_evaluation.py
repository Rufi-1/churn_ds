import numpy as np
import pandas as pd
import pytest

from churn.config import load_config, resolve_path
from churn.evaluation import compute_metrics, evaluate_model, gains_table, threshold_table
from churn.utils import load_json


def test_metrics_match_hand_calculation():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.1, 0.05]
    m = compute_metrics(y, p, 0.5)
    assert m["confusion"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3) and m["accuracy"] == pytest.approx(6 / 8)
    assert m["specificity"] == pytest.approx(4 / 5) and 0.8 < m["roc_auc"] <= 1.0


def test_threshold_extremes_and_monotonic_recall():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 500), 0, 1)
    t = threshold_table(y, p, np.full(500, 70.0), load_config())
    assert (np.diff(t["recall"]) <= 1e-12).all() and (np.diff(t["customers_flagged"]) <= 0).all()
    assert (t["tp"] + t["fp"] == t["customers_flagged"]).all()


def test_net_benefit_arithmetic():
    cfg = load_config()
    y, p, m = np.array([1, 1, 0]), np.array([0.9, 0.8, 0.7]), np.array([100.0, 50.0, 80.0])
    row = threshold_table(y, p, m, cfg, grid=[0.5]).iloc[0]
    b = cfg["business"]
    assert row["annual_revenue_protected"] == pytest.approx(150 * 12 * b["save_rate"])
    assert row["campaign_cost"] == 3 * b["retention_offer_cost"]


def test_gains_table_sums_to_all_churners():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300); p = rng.random(300)
    g = gains_table(y, p)
    assert g["churners"].sum() == y.sum() and g["cum_capture_rate"].iloc[-1] == pytest.approx(1.0)


def test_evaluate_model_consistent_with_training_metrics(pipeline_run):
    cfg = pipeline_run["cfg"]
    r = evaluate_model(cfg, save=False)
    saved = load_json(resolve_path(cfg, "models_dir") / "metrics.json")["test"]
    assert r["metrics"]["roc_auc"] == pytest.approx(saved["roc_auc"])
    assert r["metrics"]["recall"] == pytest.approx(saved["recall"])
    c = r["metrics"]["confusion"]
    assert sum(c.values()) == len(r["y"])


def test_evaluation_outputs_written(pipeline_run):
    cfg = pipeline_run["cfg"]
    f, t = resolve_path(cfg, "figures_dir"), resolve_path(cfg, "tables_dir")
    for name in ["eval_roc_curve.png", "eval_precision_recall_curve.png", "eval_confusion_matrix.png",
                 "eval_threshold_tradeoff.png"]:
        assert (f / name).exists()
    assert (t / "eval_threshold_table.csv").exists() and (t / "test_predictions.csv").exists()
    md = (resolve_path(cfg, "reports_dir") / "model_evaluation_report.md").read_text()
    assert "Precision-recall trade-off" in md and "Confusion matrix" in md


def test_higher_threshold_flags_fewer(pipeline_run):
    cfg = pipeline_run["cfg"]
    lo = evaluate_model(cfg, threshold=0.2, save=False)["metrics"]
    hi = evaluate_model(cfg, threshold=0.8, save=False)["metrics"]
    assert lo["recall"] >= hi["recall"] and lo["confusion"]["tp"] + lo["confusion"]["fp"] >= hi["confusion"]["tp"] + hi["confusion"]["fp"]
