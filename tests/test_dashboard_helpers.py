import json

import numpy as np
import pandas as pd
import pytest

from churn.dashboard_helpers import (append_history, artifacts_ready, clear_history, filter_customers, format_currency,
                                     friendly_columns, kpi_summary, load_history, metrics_at_threshold, recommend_for_frame,
                                     risk_label, scenario_net_benefit, search_sort_paginate, to_csv_bytes)
from churn.cleaning import standardize_frame
from churn.features import engineer_features
from churn.predict import make_customer_frame


@pytest.fixture()
def table():
    return pd.DataFrame({"customerID": [f"C{i}" for i in range(95)], "Contract": ["Month-to-month", "One year", "Two year"] * 31 + ["One year"] * 2,
                         "InternetService": ["DSL", "Fiber optic"] * 47 + ["No"], "PaymentMethod": ["Electronic check"] * 95,
                         "risk_tier": (["Low", "High", "Critical", "Medium", "High"] * 19), "tenure": np.arange(95) % 72,
                         "MonthlyCharges": np.linspace(20, 115, 95), "churn_probability": np.linspace(0.01, 0.99, 95),
                         "flagged": np.linspace(0.01, 0.99, 95) >= 0.5, "Churn": ["Yes", "No"] * 47 + ["No"]})


def test_filters(table):
    f = filter_customers(table, contracts=["Month-to-month"], tiers=["High"], tenure_range=(0, 30))
    assert (f["Contract"] == "Month-to-month").all() and (f["risk_tier"] == "High").all() and f["tenure"].le(30).all()
    assert len(filter_customers(table)) == len(table)
    assert filter_customers(table, only_flagged=True)["flagged"].all()


def test_search_sort_paginate(table):
    page, total, pages, p = search_sort_paginate(table, "", "MonthlyCharges", False, 1, 25)
    assert len(page) == 25 and total == 95 and pages == 4 and page["MonthlyCharges"].is_monotonic_decreasing
    last, *_ , p_last = search_sort_paginate(table, "", "MonthlyCharges", False, 99, 25)
    assert p_last == 4 and len(last) == 20
    hit, total, *_ = search_sort_paginate(table, "c7", "customerID", True, 1, 10, ["customerID"])
    assert total == 11 and hit["customerID"].str.lower().str.contains("c7").all()
    empty, total, pages, p = search_sort_paginate(table, "zzzz", None, True, 1, 10)
    assert total == 0 and pages == 1 and p == 1 and empty.empty


def test_kpis_and_scenario(table):
    k = kpi_summary(table)
    assert k["customers"] == 95 and k["flagged"] == int(table["flagged"].sum())
    assert k["annual_at_risk"] == pytest.approx(12 * table.loc[table["flagged"], "MonthlyCharges"].sum())
    s = scenario_net_benefit(table, 0.9, 0.5, 10.0)
    sel = table[table["churn_probability"] >= 0.9]
    assert s["customers_contacted"] == len(sel) and s["campaign_cost"] == 10.0 * len(sel)
    assert s["expected_annual_revenue_saved"] == pytest.approx((sel["churn_probability"] * sel["MonthlyCharges"] * 12 * 0.5).sum())
    assert s["net_benefit"] == pytest.approx(s["expected_annual_revenue_saved"] - s["campaign_cost"])


def test_metrics_at_threshold():
    m = metrics_at_threshold([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1) and m["precision"] == 0.5 and m["recall"] == 0.5


def test_formatting_helpers():
    assert format_currency(1234567, "$", True) == "$1.23M" and format_currency(12345, "$", True) == "$12.3K"
    assert format_currency(999.4) == "$999" and format_currency(float("nan")) == "-"
    assert risk_label(0.8, 0.45) == "Critical" and risk_label(0.5, 0.45) == "High" and risk_label(0.25, 0.45) == "Medium" and risk_label(0.05, 0.45) == "Low"
    assert "Customer ID" in friendly_columns(pd.DataFrame({"customerID": [1]})).columns
    assert to_csv_bytes(pd.DataFrame({"a": [1]})).decode().startswith("a")


def test_history_roundtrip(tmp_path):
    p = tmp_path / "h.csv"
    assert load_history(p).empty
    append_history(p, {"customerID": "A", "churn_probability": 0.8, "threshold": 0.45, "predicted_churn": 1,
                       "risk_tier": "Critical", "primary_action": "x", "inputs": {"tenure": 3}})
    append_history(p, {"customerID": "B", "churn_probability": 0.1, "threshold": 0.45, "predicted_churn": 0,
                       "risk_tier": "Low", "primary_action": "none", "inputs": {}})
    h = load_history(p)
    assert len(h) == 2 and json.loads(h.loc[h["customerID"] == "A", "inputs"].iloc[0]) == {"tenure": 3}
    clear_history(p)
    assert load_history(p).empty
    p.write_text("garbage,,\n\x00")
    assert isinstance(load_history(p), pd.DataFrame)


def test_recommend_for_frame(cfg):
    eng = engineer_features(standardize_frame(make_customer_frame(tenure=2, Contract="Month-to-month")), cfg)
    r = recommend_for_frame(eng, 0.9, 0.45, cfg)
    assert r["risk_tier"] == "Critical" and r["primary_action"] and r["all_actions"]


def test_artifacts_ready(pipeline_run, blank_cfg):
    assert artifacts_ready(pipeline_run["cfg"])[0] is True
    ok, missing = artifacts_ready(blank_cfg)
    assert not ok and "trained model" in missing
