import json

import joblib
import numpy as np
import pandas as pd
import pytest

from churn.config import load_config, resolve_path
from churn.data_loader import load_modeling
from churn.features import build_preprocessor, input_columns
from churn.modeling import get_candidates, select_best, split_data, tune_threshold


def test_split_is_stratified_and_disjoint(pipeline_run):
    cfg = pipeline_run["cfg"]
    m = load_modeling(cfg)
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_data(m, cfg)
    assert set(id_tr).isdisjoint(set(id_te)) and len(id_tr) + len(id_te) == len(m)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert len(X_te) == pytest.approx(len(m) * cfg["split"]["test_size"], abs=1)


def test_threshold_tuning_prefers_recall_with_higher_beta():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.35 * y + rng.normal(0.3, 0.2, 2000), 0, 1)
    t1, _ = tune_threshold(y, p, beta=1.0)
    t2, _ = tune_threshold(y, p, beta=3.0)
    assert t2 <= t1


def test_candidates_include_required_models():
    cfg = load_config(overrides={"modeling": {"fast": True}})
    names = set(get_candidates(cfg, pd.Series([0, 1, 0, 0])))
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= names


def test_parsimony_rule():
    cfg = load_config()
    comp = pd.DataFrame({"model": ["logistic_regression", "random_forest"], "display_name": ["Logistic Regression", "Random Forest"],
                         "cv_roc_auc": [0.8450, 0.8470]})
    assert select_best(comp, cfg)[0] == "logistic_regression"
    comp.loc[1, "cv_roc_auc"] = 0.8600
    assert select_best(comp, cfg)[0] == "random_forest"


def test_bundle_contents(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = joblib.load(resolve_path(cfg, "models_dir") / "churn_model.joblib")
    for k in ["pipeline", "preprocessor", "feature_names", "input_columns", "threshold", "metrics", "params",
              "trained_at", "dataset_version", "model_name"]:
        assert k in b
    assert list(b["pipeline"].named_steps) == ["prep", "clf"]
    assert len(b["feature_names"]) == len(b["pipeline"].named_steps["prep"].get_feature_names_out())
    assert 0 < b["threshold"] < 1


def test_saved_artifacts_exist(pipeline_run):
    mdir = resolve_path(pipeline_run["cfg"], "models_dir")
    for f in ["churn_model.joblib", "preprocessor.joblib", "feature_names.json", "metrics.json", "split.json", "model_runs.csv"]:
        assert (mdir / f).exists(), f
    assert len(list((mdir / "runs").glob("*.json"))) >= 3


def test_run_registry_records_every_model(pipeline_run):
    mdir = resolve_path(pipeline_run["cfg"], "models_dir")
    reg = pd.read_csv(mdir / "model_runs.csv")
    assert {"model", "params", "trained_at", "dataset_version", "cv_roc_auc", "selected"} <= set(reg.columns)
    assert reg["selected"].sum() == 1 and len(reg) >= 3
    run = json.loads(next((mdir / "runs").glob("*.json")).read_text())
    assert run["feature_importance"] and "params" in run


def test_test_set_untouched_by_training(pipeline_run):
    cfg = pipeline_run["cfg"]
    s = json.loads((resolve_path(cfg, "models_dir") / "split.json").read_text())
    assert set(s["train_ids"]).isdisjoint(s["test_ids"])
    scores = pd.read_csv(resolve_path(cfg, "tables_dir") / "customer_scores.csv")
    assert scores.set_index("customerID").loc[s["test_ids"], "score_source"].eq("hold-out (test)").all()
    assert scores.set_index("customerID").loc[s["train_ids"], "score_source"].eq("out-of-fold (train)").all()


def test_saved_preprocessor_was_fit_on_training_data_only(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = joblib.load(resolve_path(cfg, "models_dir") / "churn_model.joblib")
    m = load_modeling(cfg)
    X_tr, X_te, *_ = split_data(m, cfg)
    saved = b["preprocessor"].named_transformers_["cont"].named_steps["scale"].mean_
    ref_train = build_preprocessor(cfg).fit(X_tr).named_transformers_["cont"].named_steps["scale"].mean_
    ref_all = build_preprocessor(cfg).fit(m[input_columns(cfg)]).named_transformers_["cont"].named_steps["scale"].mean_
    assert np.allclose(saved, ref_train) and not np.allclose(saved, ref_all)


def test_probabilities_valid(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = joblib.load(resolve_path(cfg, "models_dir") / "churn_model.joblib")
    X = load_modeling(cfg)[b["input_columns"]]
    p = b["pipeline"].predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all() and p.std() > 0.05
