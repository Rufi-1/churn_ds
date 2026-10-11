import numpy as np
import pandas as pd
import pytest

from churn.cleaning import standardize_frame
from churn.config import load_config, resolve_path
from churn.features import engineer_features
from churn.insights import (FALLBACK_ACTION, NO_ACTION, RULES, add_recommendations, priority_score, revenue_summary,
                            segment_summary)
from churn.predict import make_customer_frame

CFG = load_config()


def customer(p, tier=None, **kw):
    e = engineer_features(standardize_frame(make_customer_frame(**kw)), CFG)
    e["churn_probability"] = p
    e["risk_tier"] = tier or ("Critical" if p >= .7 else "High" if p >= .45 else "Low")
    return e


def test_new_month_to_month_customer_gets_contract_offer():
    r = add_recommendations(customer(0.9, tenure=2, Contract="Month-to-month"), CFG).iloc[0]
    assert "12-month contract" in r["primary_action"] and r["n_risk_factors"] >= 2


def test_established_month_to_month_gets_loyalty_upgrade():
    r = add_recommendations(customer(0.8, tenure=40, MonthlyCharges=60.0, Contract="Month-to-month"), CFG).iloc[0]
    assert "loyalty upgrade" in r["primary_action"].lower()


def test_electronic_check_and_missing_support_rules_fire():
    r = add_recommendations(customer(0.8, tenure=30, Contract="One year", PaymentMethod="Electronic check",
                                     InternetService="DSL", TechSupport="No", OnlineSecurity="No"), CFG).iloc[0]
    assert r["rule_R4"] and r["rule_R5"] and not r["rule_R1"]


def test_low_risk_customer_gets_no_action():
    r = add_recommendations(customer(0.05, tenure=2, Contract="Month-to-month", PaymentMethod="Electronic check"), CFG).iloc[0]
    assert r["primary_action"] == NO_ACTION and r["n_risk_factors"] == 0


def test_high_risk_without_triggers_gets_fallback():
    r = add_recommendations(customer(0.8, tenure=50, Contract="Two year", PaymentMethod="Bank transfer (automatic)",
                                     InternetService="No", PhoneService="Yes", MonthlyCharges=25.0,
                                     OnlineSecurity="No internet service", OnlineBackup="No internet service",
                                     DeviceProtection="No internet service", TechSupport="No internet service",
                                     StreamingTV="No internet service", StreamingMovies="No internet service"), CFG).iloc[0]
    assert r["primary_action"] == FALLBACK_ACTION


def test_rule_ids_unique_and_impacts_valid():
    assert len({r.rule_id for r in RULES}) == len(RULES) and all(r.impact in (1, 2, 3) for r in RULES)


def test_priority_score_range_and_ordering():
    df = pd.concat([customer(0.95, tenure=1, MonthlyCharges=110.0), customer(0.2, tenure=60, MonthlyCharges=30.0)], ignore_index=True)
    df = add_recommendations(df, CFG)
    s = priority_score(df, CFG)
    assert s.between(0, 100).all() and s.iloc[0] > s.iloc[1]


def test_revenue_summary_arithmetic():
    df = pd.DataFrame({"MonthlyCharges": [100.0, 50.0, 80.0, 20.0], "flagged": [True, True, False, False],
                       "churn_probability": [0.9, 0.6, 0.2, 0.1], "risk_tier": ["Critical", "High", "Low", "Low"],
                       "Churn": ["Yes", "No", "No", "No"]})
    df["expected_monthly_loss"] = df["churn_probability"] * df["MonthlyCharges"]
    df.attrs["threshold"] = 0.5
    r = revenue_summary(df, CFG)
    b = CFG["business"]
    assert r["monthly_revenue_at_risk"] == 150 and r["annual_revenue_at_risk"] == 1800
    assert r["flagged_customers"] == 2 and r["critical_customers"] == 1
    assert r["estimated_annual_revenue_retained"] == pytest.approx(100 * 12 * b["save_rate"])
    assert r["estimated_net_benefit"] == pytest.approx(r["estimated_annual_revenue_retained"] - 2 * b["retention_offer_cost"])
    assert r["expected_monthly_loss_all_customers"] == pytest.approx(0.9 * 100 + 0.6 * 50 + 0.2 * 80 + 0.1 * 20)


def test_pipeline_insights_consistent_with_threshold(pipeline_run):
    cfg = pipeline_run["cfg"]
    t = resolve_path(cfg, "tables_dir")
    rec = pd.read_csv(t / "customer_recommendations.csv")
    summ = pd.read_json(t / "revenue_at_risk_summary.json", typ="series")
    assert (rec["flagged"] == (rec["churn_probability"] >= summ["threshold"])).all()
    assert summ["monthly_revenue_at_risk"] == pytest.approx(rec.loc[rec["flagged"], "MonthlyCharges"].sum())
    assert rec["priority_rank"].is_unique and rec["priority_score"].between(0, 100).all()
    assert (rec.loc[rec["risk_tier"] == "Low", "primary_action"] == NO_ACTION).all()
    seg = pd.read_csv(t / "insights_segments.csv")
    assert set(seg["urgency"]) <= {"Urgent", "High", "Monitor"} and seg["customers"].sum() == len(rec)
