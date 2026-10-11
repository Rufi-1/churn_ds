"""Prediction logic shared by the CLI, reports and the Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .cleaning import standardize_frame
from .config import load_config, resolve_path
from .features import engineer_features
from .schema import ID_COL

RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def bundle_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return resolve_path(cfg, "models_dir") / "churn_model.joblib"


def load_bundle(cfg: dict | None = None) -> dict:
    p = bundle_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python run_pipeline.py train` first.")
    return joblib.load(p)


def resolve_threshold(cfg: dict, bundle: dict) -> float:
    th = cfg["threshold"]
    return float(th["value"]) if th["mode"] == "fixed" else float(bundle["threshold"])


def assign_risk_tier(p, threshold: float, critical: float = 0.70):
    """Critical >= critical prob; High >= decision threshold; Medium >= half the threshold; else Low."""
    p = np.asarray(p, dtype=float)
    tier = np.where(p >= critical, "Critical",
           np.where(p >= threshold, "High", np.where(p >= 0.5 * threshold, "Medium", "Low")))
    return tier if tier.ndim else str(tier)


def prepare_features(raw_like: pd.DataFrame, bundle: dict, cfg: dict | None = None) -> pd.DataFrame:
    """Raw-schema customers -> engineered model inputs (same code path as training)."""
    cfg = cfg or load_config()
    std = standardize_frame(raw_like)
    feats = engineer_features(std, cfg)
    missing = [c for c in bundle["input_columns"] if c not in feats.columns]
    if missing:
        raise ValueError(f"Cannot build model inputs, missing columns: {missing}")
    return feats[bundle["input_columns"]]


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["pipeline"].predict_proba(X[bundle["input_columns"]])[:, 1]


def score_customers(bundle: dict, raw_like: pd.DataFrame, threshold: float | None = None,
                    cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    thr = float(bundle["threshold"] if threshold is None else threshold)
    X = prepare_features(raw_like, bundle, cfg)
    p = predict_proba(bundle, X)
    out = pd.DataFrame({"churn_probability": p})
    if ID_COL in raw_like.columns:
        out.insert(0, ID_COL, raw_like[ID_COL].to_numpy())
    out["predicted_churn"] = (p >= thr).astype(int)
    out["risk_tier"] = assign_risk_tier(p, thr, cfg["threshold"]["critical_probability"])
    return out


DEFAULT_CUSTOMER = {
    "customerID": "NEW-CUSTOMER", "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0, "TotalCharges": None,
}


def make_customer_frame(**fields) -> pd.DataFrame:
    """Single-customer raw-schema frame; TotalCharges defaults to tenure x MonthlyCharges."""
    rec = {**DEFAULT_CUSTOMER, **fields}
    if rec.get("TotalCharges") in (None, ""):
        rec["TotalCharges"] = float(rec["tenure"]) * float(rec["MonthlyCharges"])
    return pd.DataFrame([rec])
