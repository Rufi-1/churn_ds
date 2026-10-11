"""Reusable, reproducible data-cleaning functions with a full audit log.

Design rule: cleaning performs only *deterministic / logical* repairs (whitespace, formats,
types, duplicates, impossible values, cross-column consistency). Statistical imputation
(median / most-frequent) is NOT done here - it happens inside the scikit-learn pipeline and is
fitted on training data only, which prevents train/test leakage.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import load_config
from .schema import (ADDON_SERVICES, ALLOWED_VALUES, CATEGORICAL_RAW, ID_COL, NUMERIC_RAW, RAW_COLUMNS,
                     TARGET, TARGET_FLAG)

_INVALID = "__INVALID__"
_YES_NO_SYNONYMS = {"y": "Yes", "n": "No", "true": "Yes", "false": "No", "1": "Yes", "0": "No",
                    "10": "Yes", "00": "No"}


class CleaningLog:
    """Collects every cleaning action so the workflow is auditable."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, step: str, column: str, issue: str, rows_affected: int, action: str) -> None:
        self.entries.append({"step": step, "column": column, "issue": issue,
                             "rows_affected": int(rows_affected), "action": action})

    def to_frame(self) -> pd.DataFrame:
        cols = ["step", "column", "issue", "rows_affected", "action"]
        return pd.DataFrame(self.entries, columns=cols)


def _norm_key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _build_canon() -> dict[str, dict[str, str]]:
    canon = {}
    for col, vals in ALLOWED_VALUES.items():
        m = {_norm_key(v): v for v in vals}
        if "Yes" in vals and "No" in vals:
            for k, v in _YES_NO_SYNONYMS.items():
                m.setdefault(k, v)
        canon[col] = m
    return canon


_CANON = _build_canon()


# --------------------------------------------------------------------------------------------
# individual steps
# --------------------------------------------------------------------------------------------
def check_required_columns(df: pd.DataFrame, require_target: bool = True) -> None:
    required = [c for c in RAW_COLUMNS if require_target or c != TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def strip_whitespace(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Trim leading/trailing whitespace, collapse internal runs of spaces, blank -> NaN."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith(("str", "string")):
            s = df[col]
            mask = s.notna()
            cleaned = s[mask].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            changed = int((cleaned != s[mask].astype(str)).sum())
            out = pd.Series(np.nan, index=df.index, dtype=object)
            out[mask] = cleaned
            blanks = out == ""
            n_blank = int(blanks.sum())
            out[blanks] = np.nan
            df[col] = out
            if changed - n_blank > 0 or changed > 0:
                log.add("whitespace", col, "leading/trailing/duplicate whitespace", changed,
                        "trimmed and collapsed spaces")
            if n_blank:
                log.add("missing", col, "blank string", n_blank, "converted to NaN")
    return df


def standardize_categories(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Map formatting variants to the canonical category; unknown categories become NaN."""
    df = df.copy()
    for col, canon in _CANON.items():
        if col not in df.columns:
            continue
        orig = df[col]
        new_vals, n_fmt, n_inv, n_conv = [], 0, 0, 0
        for v in orig:
            if pd.isna(v):
                new_vals.append(np.nan)
                continue
            vv = v
            if isinstance(vv, float) and vv.is_integer():
                vv = int(vv)
            mapped = canon.get(_norm_key(vv), _INVALID)
            if mapped == _INVALID:
                n_inv += 1
                new_vals.append(np.nan)
            else:
                if col == "SeniorCitizen" and _norm_key(vv) in {"0", "1"}:
                    n_conv += 1
                elif str(v) != mapped:
                    n_fmt += 1
                new_vals.append(mapped)
        df[col] = pd.Series(new_vals, index=df.index, dtype=object)
        if n_conv:
            log.add("categories", col, "numeric 0/1 coding", n_conv, "mapped 0/1 to No/Yes")
        if n_fmt:
            log.add("categories", col, "inconsistent spelling/casing", n_fmt, "mapped to canonical label")
        if n_inv:
            log.add("categories", col, "invalid category", n_inv,
                    "set to NaN (imputed later inside the training pipeline)")
    return df


def coerce_numeric(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Fix data types, derive logically-determined values, null impossible values."""
    df = df.copy()
    for col in NUMERIC_RAW:
        if col not in df.columns:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        coerced = int((df[col].notna() & num.isna()).sum())
        if coerced:
            log.add("dtype", col, "non-numeric text in numeric column", coerced, "coerced to NaN")
        neg = int((num < 0).sum())
        if neg:
            log.add("validity", col, "negative value", neg, "set to NaN")
            num = num.mask(num < 0)
        df[col] = num.astype(float)
    if "tenure" in df.columns:
        frac = int(((df["tenure"] % 1) != 0).sum())
        if frac:
            log.add("dtype", "tenure", "non-integer tenure", frac, "rounded to whole months")
            df["tenure"] = df["tenure"].round()
    # Logical imputation: a customer with tenure 0 has not been billed yet -> TotalCharges = 0.
    if {"tenure", "TotalCharges"} <= set(df.columns):
        m = df["TotalCharges"].isna() & (df["tenure"] == 0)
        if m.sum():
            df.loc[m, "TotalCharges"] = 0.0
            log.add("missing", "TotalCharges", "missing for tenure = 0 (not yet billed)", m.sum(),
                    "set to 0.0 (logical rule)")
        if "MonthlyCharges" in df.columns:
            m2 = df["TotalCharges"].isna() & df["tenure"].notna() & df["MonthlyCharges"].notna()
            if m2.sum():
                df.loc[m2, "TotalCharges"] = df.loc[m2, "tenure"] * df.loc[m2, "MonthlyCharges"]
                log.add("missing", "TotalCharges", "missing TotalCharges", m2.sum(),
                        "derived as tenure x MonthlyCharges")
    return df


def drop_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log.add("duplicates", "all columns", "exact duplicate rows", before - len(df), "dropped (kept first)")
    if ID_COL in df.columns:
        before = len(df)
        has_id = df[ID_COL].notna()
        dup = df[ID_COL].duplicated(keep="first") & has_id
        df = df[~dup]
        if before - len(df):
            log.add("duplicates", ID_COL, "repeated customer id", before - len(df), "dropped (kept first)")
    return df.reset_index(drop=True)


def fix_consistency(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Cross-column business-rule consistency (internet add-ons vs InternetService, etc.)."""
    df = df.copy()
    if "InternetService" in df.columns:
        no_int = df["InternetService"] == "No"
        for col in ADDON_SERVICES:
            if col not in df.columns:
                continue
            bad = no_int & (df[col] != "No internet service")
            if bad.sum():
                df.loc[bad, col] = "No internet service"
                log.add("consistency", col, "add-on recorded although customer has no internet", bad.sum(),
                        "set to 'No internet service'")
            has_int = df["InternetService"].isin(["DSL", "Fiber optic"])
            bad2 = has_int & (df[col] == "No internet service")
            if bad2.sum():
                df.loc[bad2, col] = "No"
                log.add("consistency", col, "'No internet service' although customer has internet",
                        bad2.sum(), "set to 'No'")
    if {"PhoneService", "MultipleLines"} <= set(df.columns):
        bad = (df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")
        if bad.sum():
            df.loc[bad, "MultipleLines"] = "No phone service"
            log.add("consistency", "MultipleLines", "multiple lines recorded without phone service", bad.sum(),
                    "set to 'No phone service'")
        bad2 = (df["PhoneService"] == "Yes") & (df["MultipleLines"] == "No phone service")
        if bad2.sum():
            df.loc[bad2, "MultipleLines"] = "No"
            log.add("consistency", "MultipleLines", "'No phone service' although phone service is Yes",
                    bad2.sum(), "set to 'No'")
    return df


def detect_outliers(df: pd.DataFrame, cols=NUMERIC_RAW, k: float = 1.5) -> pd.DataFrame:
    """IQR outlier summary (Tukey fences). Outliers are flagged, not deleted - they are genuine
    customers; extreme values are winsorised inside the model pipeline using training quantiles."""
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": col, "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": n_out, "pct_outliers": 100 * n_out / len(s),
                     "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows)


def drop_missing_target(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    if TARGET not in df.columns:
        return df
    m = df[TARGET].isna()
    if m.sum():
        log.add("target", TARGET, "missing or invalid churn label", m.sum(),
                "rows dropped (a label cannot be imputed)")
        df = df[~m].reset_index(drop=True)
    return df


def finalize_dtypes(df: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    df = df.copy()
    if "tenure" in df.columns and df["tenure"].notna().all():
        df["tenure"] = df["tenure"].astype(int)
    if TARGET in df.columns:
        df[TARGET_FLAG] = (df[TARGET] == "Yes").astype(int)
    return df


# --------------------------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------------------------
def clean_dataset(raw: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full, ordered cleaning workflow. Returns (clean_df, cleaning_log_df)."""
    cfg = cfg or load_config()
    log = CleaningLog()
    check_required_columns(raw)
    df = strip_whitespace(raw, log)
    df = standardize_categories(df, log)
    df = coerce_numeric(df, log)
    df = drop_duplicates(df, log)
    df = fix_consistency(df, log)
    df = drop_missing_target(df, log)
    out = detect_outliers(df, k=cfg["cleaning"]["outlier_iqr_multiplier"])
    for r in out.itertuples():
        if r.n_outliers:
            log.add("outliers", r.column, "values outside Tukey IQR fences", r.n_outliers,
                    "kept (genuine values); winsorised on training data in the model pipeline")
    remaining = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    for c, n in remaining.items():
        log.add("missing", c, "remaining missing values", n,
                "left as NaN; imputed inside the training pipeline (no leakage)")
    df = finalize_dtypes(df, log)
    log.add("summary", "-", f"{len(raw)} raw rows -> {len(df)} clean rows", len(raw) - len(df),
            "rows removed in total")
    return df, log.to_frame()


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row-preserving cleaning for scoring new customers (no dedup, no row drops)."""
    log = CleaningLog()
    check_required_columns(df, require_target=False)
    out = strip_whitespace(df, log)
    out = standardize_categories(out, log)
    out = coerce_numeric(out, log)
    out = fix_consistency(out, log)
    return out
