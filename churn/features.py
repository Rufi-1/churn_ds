"""Feature engineering and leakage-safe preprocessing (scikit-learn Pipelines + ColumnTransformer)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_config, resolve_path
from .schema import (ADDON_SERVICES, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, DISCRETE_FEATURES, FEATURE_LABELS,
                     ID_COL, TARGET_FLAG)

_CONTRACT_RISK = {"Month-to-month": 2, "One year": 1, "Two year": 0}


def engineer_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Deterministic, row-wise features. Nothing here is learned from data, so there is no leakage."""
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    f = df.copy()
    f["tenure_group"] = pd.cut(f["tenure"], bins=fcfg["tenure_bins"], labels=fcfg["tenure_labels"]).astype(object)
    safe_t = f["tenure"].clip(lower=1)
    f["avg_monthly_spend"] = np.where(f["tenure"] == 0, f["MonthlyCharges"], f["TotalCharges"] / safe_t)
    f["charge_delta"] = f["MonthlyCharges"] - f["avg_monthly_spend"]
    f["bill_increase_flag"] = (f["charge_delta"] > fcfg["bill_increase_threshold"]).astype(int)

    has_phone = (f["PhoneService"] == "Yes").astype(int)
    has_internet = f["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    yes = {c: (f[c] == "Yes").astype(int) for c in ADDON_SERVICES}
    f["protection_count"] = sum(yes[c] for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"])
    f["streaming_count"] = yes["StreamingTV"] + yes["StreamingMovies"]
    f["total_services"] = has_phone + has_internet + (f["MultipleLines"] == "Yes").astype(int) \
        + f["protection_count"] + f["streaming_count"]
    f["no_support_flag"] = ((has_internet == 1) & (yes["OnlineSecurity"] == 0) & (yes["TechSupport"] == 0)).astype(int)
    f["fiber_no_support"] = ((f["InternetService"] == "Fiber optic") & (yes["TechSupport"] == 0)).astype(int)

    f["is_month_to_month"] = (f["Contract"] == "Month-to-month").astype(int)
    f["contract_risk_score"] = f["Contract"].map(_CONTRACT_RISK)
    f["electronic_check"] = (f["PaymentMethod"] == "Electronic check").astype(int)
    f["auto_pay"] = f["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
    f["billing_risk_score"] = (f["is_month_to_month"] + f["electronic_check"]
                               + (f["PaperlessBilling"] == "Yes").astype(int) + f["bill_increase_flag"])
    f["customer_value_segment"] = pd.cut(f["MonthlyCharges"], bins=fcfg["value_bins"],
                                         labels=fcfg["value_labels"]).astype(object)

    def profile(i):
        toks = []
        if has_phone.iloc[i]:
            toks.append("Phone")
        if has_internet.iloc[i]:
            toks.append("Internet")
            if f["protection_count"].iloc[i] > 0:
                toks.append("Protection")
            if f["streaming_count"].iloc[i] > 0:
                toks.append("Streaming")
        return "+".join(toks) if toks else "None"

    f["service_profile"] = [profile(i) for i in range(len(f))]
    return f


def model_columns(cfg: dict | None = None) -> tuple[list[str], list[str], list[str]]:
    """(continuous, discrete, categorical) model inputs after configured exclusions."""
    cfg = cfg or load_config()
    ex = set(cfg["features"].get("exclude_from_model", []))
    return ([c for c in CONTINUOUS_FEATURES if c not in ex], [c for c in DISCRETE_FEATURES if c not in ex],
            [c for c in CATEGORICAL_FEATURES if c not in ex])


def input_columns(cfg: dict | None = None) -> list[str]:
    a, b, c = model_columns(cfg)
    return a + b + c


def prepare_modeling_data(clean: pd.DataFrame, cfg: dict | None = None, save: bool = True) -> pd.DataFrame:
    """Engineered, UN-scaled, modeling-ready table: id + target + model inputs."""
    cfg = cfg or load_config()
    feats = engineer_features(clean, cfg)
    cols = [ID_COL, TARGET_FLAG, "MonthlyCharges"] + [c for c in input_columns(cfg) if c != "MonthlyCharges"]
    out = feats[cols].copy()
    if save:
        p = resolve_path(cfg, "modeling_data", create=True)
        out.to_csv(p, index=False)
    return out


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the TRAINING data only."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_q, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(cfg: dict | None = None) -> ColumnTransformer:
    """Unfitted preprocessor. Imputation, winsorising, scaling and encoding are all learned inside
    fit(), so when it sits in a Pipeline and is fitted on the training fold only, nothing leaks."""
    cfg = cfg or load_config()
    cont, disc, cat = model_columns(cfg)
    lo, hi = cfg["features"]["winsor_quantiles"]
    cont_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("winsor", Winsorizer(lo, hi)),
                          ("scale", StandardScaler())])
    disc_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("cont", cont_pipe, cont), ("disc", disc_pipe, disc), ("cat", cat_pipe, cat)],
                             remainder="drop", verbose_feature_names_out=True)


def pretty_feature_name(name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract type: Month-to-month'."""
    base = name.split("__", 1)[-1]
    for col in sorted(FEATURE_LABELS, key=len, reverse=True):
        if base == col:
            return FEATURE_LABELS[col]
        if base.startswith(col + "_"):
            return f"{FEATURE_LABELS[col]}: {base[len(col) + 1:]}"
    return base


def source_column(name: str, columns: list[str]) -> str:
    """Map a transformed feature name back to the original input column."""
    base = name.split("__", 1)[-1]
    for col in sorted(columns, key=len, reverse=True):
        if base == col or base.startswith(col + "_"):
            return col
    return base
