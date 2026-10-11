"""Business insights: rule-based retention recommendations (no external AI), revenue at risk,
customer prioritisation and segment-level action plans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .data_loader import load_clean
from .features import engineer_features
from .predict import assign_risk_tier, load_bundle, resolve_threshold
from .schema import ID_COL, TARGET
from .utils import get_logger, now_iso, save_json

log = get_logger()


@dataclass(frozen=True)
class Rule:
    rule_id: str
    driver: str            # short risk-factor description
    action: str            # what the retention team should do
    impact: int            # 1 (nice to have) .. 3 (highest expected effect) - used to pick the primary action
    mask: Callable[[pd.DataFrame, dict], pd.Series]


def build_rules() -> list[Rule]:
    return [
        Rule("R1", "Month-to-month contract, first year",
             "Offer a 12-month contract with a first-year discount or price lock", 3,
             lambda d, c: (d["is_month_to_month"] == 1) & (d["tenure"] <= c["early_tenure_months"])),
        Rule("R2", "Month-to-month contract, established customer",
             "Offer a loyalty upgrade to a 1- or 2-year contract with a free add-on", 3,
             lambda d, c: (d["is_month_to_month"] == 1) & (d["tenure"] > c["early_tenure_months"])),
        Rule("R3", "Onboarding window (very new customer)",
             "Schedule a 30-day onboarding call and first-bill walkthrough", 3,
             lambda d, c: d["tenure"] <= c["new_customer_months"]),
        Rule("R4", "Pays by electronic check",
             "Offer a one-off bill credit for switching to automatic bank/credit-card payment", 2,
             lambda d, c: d["electronic_check"] == 1),
        Rule("R5", "Internet without security or tech support",
             "Offer a free 3-month Tech Support + Online Security trial", 2,
             lambda d, c: d["no_support_flag"] == 1),
        Rule("R6", "High-priced fibre plan",
             "Run a service-quality check and a plan/price review call", 2,
             lambda d, c: (d["InternetService"] == "Fiber optic") & (d["MonthlyCharges"] >= c["high_charge"])),
        Rule("R7", "Bill above historical average",
             "Proactive bill review: explain the increase or offer a price lock", 2,
             lambda d, c: d["bill_increase_flag"] == 1),
        Rule("R8", "Internet-only, few add-ons",
             "Bundle security/backup add-ons at a discount to deepen engagement", 1,
             lambda d, c: (d["InternetService"].isin(["DSL", "Fiber optic"])) & (d["protection_count"] == 0)
             & (d["streaming_count"] == 0)),
        Rule("R9", "Senior customer",
             "Offer assisted support: priority line and a simplified plan", 1,
             lambda d, c: d["SeniorCitizen"] == "Yes"),
    ]


RULES = build_rules()
FALLBACK_ACTION = "Proactive satisfaction call and personalised offer"
NO_ACTION = "No action - monitor"


def add_recommendations(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adds rule flags, risk_factors, primary_action and all_actions. Needs engineered columns + risk_tier."""
    rc = cfg["recommendations"]
    out = df.copy()
    flags = pd.DataFrame({r.rule_id: r.mask(out, rc).fillna(False).astype(bool) for r in RULES}, index=out.index)
    actionable = out["risk_tier"] != "Low"
    flags = flags.mul(actionable, axis=0).astype(bool)
    out["n_risk_factors"] = flags.sum(axis=1)
    by_impact = sorted(RULES, key=lambda r: -r.impact)

    def pick(i):
        if not actionable.loc[i]:
            return NO_ACTION
        for r in by_impact:
            if flags.loc[i, r.rule_id]:
                return r.action
        return FALLBACK_ACTION

    out["primary_action"] = [pick(i) for i in out.index]
    out["all_actions"] = [" | ".join(r.action for r in by_impact if flags.loc[i, r.rule_id]) for i in out.index]
    out["risk_factors"] = [", ".join(r.driver for r in RULES if flags.loc[i, r.rule_id]) for i in out.index]
    for r in RULES:
        out[f"rule_{r.rule_id}"] = flags[r.rule_id]
    return out


def priority_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """0-100 score from churn probability, revenue value, early tenure and number of risk factors."""
    w = cfg["business"]["priority_weights"]
    rev = df["MonthlyCharges"].rank(pct=True)
    ten = 1 - df["tenure"].clip(upper=72) / 72
    fac = df["n_risk_factors"] / len(RULES)
    s = w["probability"] * df["churn_probability"] + w["revenue"] * rev + w["tenure"] * ten + w["risk_factors"] * fac
    return (100 * s / sum(w.values())).round(2)


def assemble_customer_table(eng: pd.DataFrame, cfg: dict, thr: float) -> pd.DataFrame:
    """Engineered customers + churn_probability (+ Churn, score_source) -> full action table:
    risk tier, recommendations, priority score/rank, revenue at risk, segment."""
    df = eng.copy()
    df["risk_tier"] = assign_risk_tier(df["churn_probability"].to_numpy(), thr, cfg["threshold"]["critical_probability"])
    df["flagged"] = df["churn_probability"] >= thr
    df = add_recommendations(df, cfg)
    df["priority_score"] = priority_score(df, cfg)
    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="first").astype(int)
    df["monthly_revenue_at_risk"] = np.where(df["flagged"], df["MonthlyCharges"].fillna(0.0), 0.0)
    df["annual_revenue_at_risk"] = 12 * df["monthly_revenue_at_risk"]
    df["expected_monthly_loss"] = df["churn_probability"] * df["MonthlyCharges"].fillna(0.0)
    df["segment"] = df["Contract"].astype(str) + " | " + df["tenure_group"].astype(str)
    df.attrs["threshold"] = thr
    return df.sort_values("priority_rank").reset_index(drop=True)


def build_customer_table(cfg: dict | None = None, clean: pd.DataFrame | None = None,
                         scores: pd.DataFrame | None = None, threshold: float | None = None,
                         bundle: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    bundle = bundle or load_bundle(cfg)
    thr = float(threshold) if threshold is not None else resolve_threshold(cfg, bundle)
    clean = load_clean(cfg) if clean is None else clean
    if scores is None:
        scores = pd.read_csv(resolve_path(cfg, "tables_dir") / "customer_scores.csv")
    eng = engineer_features(clean, cfg).merge(scores[[ID_COL, "churn_probability", "score_source"]], on=ID_COL, how="inner")
    return assemble_customer_table(eng, cfg, thr)


def revenue_summary(df: pd.DataFrame, cfg: dict) -> dict:
    b = cfg["business"]
    fl = df[df["flagged"]]
    monthly = float(fl["MonthlyCharges"].sum())
    annual = 12 * monthly
    tp_mask = fl["Churn"] == "Yes"
    retained_value = float(12 * fl.loc[tp_mask, "MonthlyCharges"].sum() * b["save_rate"])
    cost = float(len(fl) * b["retention_offer_cost"])
    return {
        "threshold": float(df.attrs.get("threshold", np.nan)), "customers": int(len(df)),
        "total_monthly_revenue": float(df["MonthlyCharges"].sum()),
        "flagged_customers": int(len(fl)), "flagged_share": float(len(fl) / max(len(df), 1)),
        "monthly_revenue_at_risk": monthly, "annual_revenue_at_risk": annual,
        "pct_of_monthly_revenue_at_risk": float(monthly / max(df["MonthlyCharges"].sum(), 1e-9)),
        "expected_monthly_loss_all_customers": float(df["expected_monthly_loss"].sum()),
        "critical_customers": int((df["risk_tier"] == "Critical").sum()),
        "critical_monthly_revenue": float(df.loc[df["risk_tier"] == "Critical", "MonthlyCharges"].sum()),
        "historical_churned_monthly_revenue": float(df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()),
        "campaign_cost": cost, "assumed_save_rate": b["save_rate"],
        "assumed_offer_cost": b["retention_offer_cost"],
        "estimated_annual_revenue_retained": retained_value,
        "estimated_net_benefit": retained_value - cost,
        "currency": b["currency"], "generated_at": now_iso(),
    }


def revenue_breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(customers=("flagged", "size"), flagged=("flagged", "sum"),
                                         avg_probability=("churn_probability", "mean"),
                                         monthly_revenue_at_risk=("monthly_revenue_at_risk", "sum"),
                                         expected_monthly_loss=("expected_monthly_loss", "sum")).reset_index()
    g["annual_revenue_at_risk"] = 12 * g["monthly_revenue_at_risk"]
    g["flagged_pct"] = g["flagged"] / g["customers"]
    return g.sort_values("monthly_revenue_at_risk", ascending=False).reset_index(drop=True)


def segment_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = cfg["business"]
    rows = []
    for seg, d in df.groupby("segment"):
        fl = d[d["flagged"]]
        top = fl["primary_action"].value_counts()
        rows.append({"segment": seg, "customers": len(d), "flagged": len(fl),
                     "flagged_pct": len(fl) / len(d), "avg_churn_probability": d["churn_probability"].mean(),
                     "monthly_revenue_at_risk": fl["MonthlyCharges"].sum(),
                     "annual_revenue_at_risk": 12 * fl["MonthlyCharges"].sum(),
                     "top_recommended_action": top.index[0] if len(top) else NO_ACTION})
    s = pd.DataFrame(rows)
    med = s["monthly_revenue_at_risk"].median()
    s["urgency"] = np.where((s["flagged_pct"] >= b["segment_urgent_flag_rate"]) & (s["monthly_revenue_at_risk"] >= med),
                            "Urgent", np.where(s["flagged_pct"] >= b["segment_high_flag_rate"], "High", "Monitor"))
    return s.sort_values("monthly_revenue_at_risk", ascending=False).reset_index(drop=True)


def recommendation_summary(df: pd.DataFrame) -> pd.DataFrame:
    fl = df[df["flagged"]]
    rows = []
    for r in RULES:
        m = fl[f"rule_{r.rule_id}"]
        rows.append({"rule_id": r.rule_id, "risk_factor": r.driver, "recommended_action": r.action, "impact": r.impact,
                     "flagged_customers_affected": int(m.sum()),
                     "monthly_revenue_covered": float(fl.loc[m, "MonthlyCharges"].sum())})
    out = pd.DataFrame(rows)
    return out.sort_values(["impact", "monthly_revenue_covered"], ascending=False).reset_index(drop=True)


def run_business_insights(cfg: dict | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    df = build_customer_table(cfg)
    rev = revenue_summary(df, cfg)
    res = {"customers": df, "revenue": rev, "segments": segment_summary(df, cfg),
           "recommendations": recommendation_summary(df),
           "by_contract": revenue_breakdown(df, "Contract"), "by_tier": revenue_breakdown(df, "risk_tier"),
           "by_tenure": revenue_breakdown(df, "tenure_group"),
           "by_internet": revenue_breakdown(df, "InternetService"),
           "by_value": revenue_breakdown(df, "customer_value_segment")}
    if save:
        tdir = resolve_path(cfg, "tables_dir", create=True)
        keep = [ID_COL, "churn_probability", "risk_tier", "flagged", "priority_score", "priority_rank",
                "primary_action", "all_actions", "risk_factors", "n_risk_factors", "MonthlyCharges", "tenure",
                "Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "customer_value_segment",
                "tenure_group", "segment", "monthly_revenue_at_risk", "annual_revenue_at_risk", "score_source",
                "Churn"]
        df[keep].to_csv(tdir / "customer_recommendations.csv", index=False)
        df[df["flagged"]][keep].head(500).to_csv(tdir / "priority_customers_top500.csv", index=False)
        for k in ["segments", "recommendations", "by_contract", "by_tier", "by_tenure", "by_internet", "by_value"]:
            res[k].to_csv(tdir / f"insights_{k}.csv", index=False)
        save_json(rev, tdir / "revenue_at_risk_summary.json")
        log.info("Insights: %d flagged customers, monthly revenue at risk %s%s (annual %s%s)", rev["flagged_customers"],
                 rev["currency"], f"{rev['monthly_revenue_at_risk']:,.0f}", rev["currency"],
                 f"{rev['annual_revenue_at_risk']:,.0f}")
    return res
