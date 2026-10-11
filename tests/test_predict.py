import numpy as np
import pandas as pd
import pytest

from churn.predict import (assign_risk_tier, load_bundle, make_customer_frame, predict_proba, prepare_features,
                           resolve_threshold, score_customers)


def test_risk_tiers():
    t = assign_risk_tier(np.array([0.05, 0.3, 0.5, 0.8]), threshold=0.45, critical=0.7)
    assert list(t) == ["Low", "Medium", "High", "Critical"]


def test_make_customer_frame_estimates_total_charges():
    f = make_customer_frame(tenure=10, MonthlyCharges=50.0)
    assert f.loc[0, "TotalCharges"] == 500.0


def test_scoring_valid_and_deterministic(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = load_bundle(cfg)
    f = make_customer_frame(tenure=2, MonthlyCharges=95.0, Contract="Month-to-month", PaymentMethod="Electronic check")
    s1, s2 = score_customers(b, f, cfg=cfg), score_customers(b, f, cfg=cfg)
    assert 0 <= s1.loc[0, "churn_probability"] <= 1 and s1.loc[0, "churn_probability"] == s2.loc[0, "churn_probability"]
    assert {"predicted_churn", "risk_tier", "customerID"} <= set(s1.columns)


def test_risky_profile_scores_above_loyal_profile(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = load_bundle(cfg)
    risky = make_customer_frame(tenure=1, MonthlyCharges=90.0, Contract="Month-to-month", PaymentMethod="Electronic check",
                                InternetService="Fiber optic", TechSupport="No", OnlineSecurity="No")
    loyal = make_customer_frame(tenure=66, MonthlyCharges=60.0, Contract="Two year", PaymentMethod="Bank transfer (automatic)",
                                PaperlessBilling="No", TechSupport="Yes", OnlineSecurity="Yes", Partner="Yes", Dependents="Yes")
    p = score_customers(b, pd.concat([risky, loyal], ignore_index=True), cfg=cfg)["churn_probability"]
    assert p.iloc[0] > 0.5 > p.iloc[1] and p.iloc[0] - p.iloc[1] > 0.4


def test_threshold_changes_decision_not_probability(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = load_bundle(cfg)
    f = make_customer_frame(tenure=5, MonthlyCharges=80.0)
    lo = score_customers(b, f, threshold=0.01, cfg=cfg).iloc[0]
    hi = score_customers(b, f, threshold=0.99, cfg=cfg).iloc[0]
    assert lo["churn_probability"] == hi["churn_probability"] and lo["predicted_churn"] == 1 and hi["predicted_churn"] == 0


def test_bad_inputs_are_handled(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = load_bundle(cfg)
    f = make_customer_frame(InternetService="Satellite", Contract="Weekly")  # invalid categories -> NaN -> imputed
    assert np.isfinite(score_customers(b, f, cfg=cfg).loc[0, "churn_probability"])
    with pytest.raises(ValueError):
        prepare_features(f.drop(columns=["Contract"]), b, cfg)


def test_resolve_threshold_modes(pipeline_run):
    cfg = pipeline_run["cfg"]
    b = load_bundle(cfg)
    assert resolve_threshold(cfg, b) == b["threshold"]
    fixed = {**cfg, "threshold": {**cfg["threshold"], "mode": "fixed", "value": 0.33}}
    assert resolve_threshold(fixed, b) == 0.33
