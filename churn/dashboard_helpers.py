"""Pure helper functions for the Streamlit dashboard (kept separate so they can be unit-tested)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .schema import ID_COL

REQUIRED_ARTIFACTS = ["clean_data", "models_dir", "tables_dir"]
PAGE_SIZES = [10, 25, 50, 100]

COLUMN_LABELS = {
    "customerID": "Customer ID", "churn_probability": "Churn probability", "risk_tier": "Risk tier",
    "flagged": "Flagged high-risk", "priority_score": "Priority score", "priority_rank": "Priority rank",
    "primary_action": "Recommended action", "all_actions": "All recommended actions", "risk_factors": "Risk factors",
    "n_risk_factors": "# risk factors", "MonthlyCharges": "Monthly charges", "tenure": "Tenure (months)",
    "Contract": "Contract", "InternetService": "Internet service", "PaymentMethod": "Payment method",
    "SeniorCitizen": "Senior citizen", "customer_value_segment": "Value segment", "tenure_group": "Tenure band",
    "segment": "Segment", "monthly_revenue_at_risk": "Monthly revenue at risk",
    "annual_revenue_at_risk": "Annual revenue at risk", "score_source": "Score source", "Churn": "Actually churned",
    "TotalCharges": "Total charges", "customers": "Customers", "flagged_pct": "% flagged",
    "avg_churn_probability": "Avg churn probability", "urgency": "Urgency", "top_recommended_action": "Top action",
}


def artifacts_ready(cfg: dict | None = None) -> tuple[bool, list[str]]:
    """Check that the pipeline has produced what the dashboard needs."""
    cfg = cfg or load_config()
    need = {"cleaned data": resolve_path(cfg, "clean_data"),
            "trained model": resolve_path(cfg, "models_dir") / "churn_model.joblib",
            "customer recommendations": resolve_path(cfg, "tables_dir") / "customer_recommendations.csv",
            "revenue summary": resolve_path(cfg, "tables_dir") / "revenue_at_risk_summary.json"}
    missing = [k for k, p in need.items() if not Path(p).exists()]
    return not missing, missing


def format_currency(x: float, symbol: str = "$", compact: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if compact and abs(x) >= 1e6:
        return f"{symbol}{x / 1e6:,.2f}M"
    if compact and abs(x) >= 1e4:
        return f"{symbol}{x / 1e3:,.1f}K"
    return f"{symbol}{x:,.0f}"


def friendly_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_LABELS.get(c, c) for c in df.columns})


def filter_customers(df: pd.DataFrame, contracts=None, internet=None, tiers=None, payments=None,
                     tenure_range=None, charge_range=None, only_flagged: bool = False) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for col, vals in [("Contract", contracts), ("InternetService", internet), ("risk_tier", tiers),
                      ("PaymentMethod", payments)]:
        if vals:
            m &= df[col].isin(vals)
    if tenure_range:
        m &= df["tenure"].between(*tenure_range)
    if charge_range:
        m &= df["MonthlyCharges"].between(*charge_range)
    if only_flagged:
        m &= df["flagged"].astype(bool)
    return df[m]


def search_sort_paginate(df: pd.DataFrame, query: str = "", sort_by: str | None = None, ascending: bool = False,
                         page: int = 1, page_size: int = 25, search_cols: list[str] | None = None):
    """Returns (page_df, total_rows, total_pages, page_used)."""
    d = df
    if query and query.strip():
        cols = search_cols or [c for c in d.columns if d[c].dtype == object or str(d[c].dtype).startswith("str")]
        q = query.strip().lower()
        mask = pd.Series(False, index=d.index)
        for c in cols:
            mask |= d[c].astype(str).str.lower().str.contains(q, regex=False, na=False)
        d = d[mask]
    if sort_by and sort_by in d.columns:
        d = d.sort_values(sort_by, ascending=ascending, kind="mergesort")
    total = len(d)
    pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, int(page)), pages)
    start = (page - 1) * page_size
    return d.iloc[start:start + page_size], total, pages, page


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def kpi_summary(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """KPIs for any (possibly filtered) customer table."""
    n = len(df)
    fl = df[df["flagged"].astype(bool)]
    return {"customers": n, "flagged": len(fl), "flagged_pct": len(fl) / n if n else 0.0,
            "avg_probability": float(df["churn_probability"].mean()) if n else 0.0,
            "monthly_revenue": float(df["MonthlyCharges"].sum()),
            "monthly_at_risk": float(fl["MonthlyCharges"].sum()),
            "annual_at_risk": float(12 * fl["MonthlyCharges"].sum()),
            "critical": int((df["risk_tier"] == "Critical").sum()),
            "actual_churn_rate": float((df["Churn"] == "Yes").mean()) if n and "Churn" in df else float("nan")}


def scenario_net_benefit(df: pd.DataFrame, threshold: float, save_rate: float, offer_cost: float) -> dict:
    """What-if campaign economics for customers at/above a threshold (probability-weighted)."""
    fl = df[df["churn_probability"] >= threshold]
    exp_saved = float((fl["churn_probability"] * fl["MonthlyCharges"] * 12 * save_rate).sum())
    cost = float(len(fl) * offer_cost)
    return {"customers_contacted": len(fl), "monthly_revenue_at_risk": float(fl["MonthlyCharges"].sum()),
            "annual_revenue_at_risk": float(12 * fl["MonthlyCharges"].sum()),
            "expected_annual_revenue_saved": exp_saved, "campaign_cost": cost, "net_benefit": exp_saved - cost}


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, float)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": f1, "accuracy": (tp + tn) / max(len(y), 1), "flagged": tp + fp}


def risk_label(p: float, threshold: float, critical: float = 0.70) -> str:
    if p >= critical:
        return "Critical"
    if p >= threshold:
        return "High"
    return "Medium" if p >= 0.5 * threshold else "Low"


RISK_COLORS = {"Low": "#2E9E5B", "Medium": "#E0A100", "High": "#E4572E", "Critical": "#B0171F"}

HISTORY_COLUMNS = ["timestamp", "customerID", "churn_probability", "threshold", "predicted_churn", "risk_tier",
                   "primary_action", "inputs"]


def append_history(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {c: record.get(c) for c in HISTORY_COLUMNS}
    row["timestamp"] = row["timestamp"] or datetime.now().replace(microsecond=0).isoformat()
    if isinstance(row["inputs"], dict):
        row["inputs"] = json.dumps(row["inputs"], default=str)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_history(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        return pd.read_csv(path).sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception:  # noqa: BLE001 - corrupt/empty history file
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def clear_history(path: Path) -> None:
    Path(path).unlink(missing_ok=True)


def recommend_for_frame(row_df: pd.DataFrame, probability: float, threshold: float, cfg: dict) -> dict:
    """Rule-based recommendations for one customer (engineered features in row_df, one row)."""
    from .insights import add_recommendations
    d = row_df.copy()
    d["churn_probability"] = probability
    d["risk_tier"] = risk_label(probability, threshold, cfg["threshold"]["critical_probability"])
    out = add_recommendations(d, cfg).iloc[0]
    return {"risk_tier": out["risk_tier"], "primary_action": out["primary_action"],
            "all_actions": [a for a in out["all_actions"].split(" | ") if a],
            "risk_factors": [a for a in out["risk_factors"].split(", ") if a]}
