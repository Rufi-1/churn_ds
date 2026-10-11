import numpy as np
import pandas as pd

from churn.config import resolve_path
from churn.data_loader import load_modeling
from churn.explain import (active_groups, explain_customer, marginal_effects, permutation_importance_table, shap_local)
from churn.predict import load_bundle, predict_proba


def _X(cfg, b, n=30):
    return load_modeling(cfg)[b["input_columns"]].head(n)


def test_groups_cover_only_model_inputs(cfg):
    b = load_bundle(cfg)
    g = active_groups(b)
    assert g and all(c in b["input_columns"] for cols in g.values() for c in cols)
    assert "gender" not in {c for cols in g.values() for c in cols}


def test_marginal_effects_shape_and_finite(cfg):
    b = load_bundle(cfg)
    X = _X(cfg, b)
    e = marginal_effects(b, X)
    assert e.shape == (len(X), len(active_groups(b))) and np.isfinite(e.to_numpy()).all()
    assert (e.abs() <= 1).all().all()


def test_replacing_with_own_values_gives_zero_effect(cfg):
    b = load_bundle(cfg)
    X = _X(cfg, b, 5)
    e = marginal_effects(b, X, background=pd.concat([X.iloc[[0]]] * 5, ignore_index=True))
    # background of identical rows equal to customer 0 -> effect for customer 0 must be exactly 0
    assert np.allclose(e.iloc[0].to_numpy(), 0.0)


def test_explain_customer_output(cfg):
    b = load_bundle(cfg)
    X = _X(cfg, b, 1)
    ex = explain_customer(b, X, top_n=4)
    assert len(ex) == 4 and {"driver", "effect_pp", "explanation", "direction"} <= set(ex.columns)
    assert ex["effect_pp"].abs().is_monotonic_decreasing
    assert all("percentage points" in t for t in ex["explanation"])


def test_permutation_importance_is_nonnegative_on_average(cfg):
    b = load_bundle(cfg)
    data = load_modeling(cfg)
    X, y = data[b["input_columns"]], data["ChurnFlag"].to_numpy()
    t = permutation_importance_table(b, X, y, active_groups(b), repeats=3, seed=1)
    assert t["baseline_auc"].iloc[0] > 0.7
    assert t.iloc[0]["importance_mean"] > 0 and t["importance_mean"].is_monotonic_decreasing


def test_explainability_outputs_written(cfg):
    t, f = resolve_path(cfg, "tables_dir"), resolve_path(cfg, "figures_dir")
    assert (t / "explain_group_importance.csv").exists() and (f / "explain_group_importance.png").exists()
    assert (resolve_path(cfg, "reports_dir") / "explainability_report.md").exists()


def test_shap_local_is_dataframe_or_none(cfg):
    b = load_bundle(cfg)
    r = shap_local(b, _X(cfg, b, 1))
    assert r is None or (isinstance(r, pd.DataFrame) and "shap_value" in r.columns)
