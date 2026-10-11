"""User uploads: column auto-mapping + validation, private per-session workspaces that run the full
pipeline on the uploaded file, and batch scoring with the pre-trained model.

Privacy model: an uploaded file is written only to a temporary workspace folder that belongs to that
browser session (random id). Nothing is committed, shared between users or sent to any external service.
Workspaces older than WORKSPACE_TTL_HOURS are deleted whenever a new one is created.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaning import _CANON, _norm_key, standardize_frame
from .config import load_config
from .features import engineer_features
from .insights import assemble_customer_table, revenue_summary
from .predict import load_bundle, predict_proba, resolve_threshold
from .schema import ALLOWED_VALUES, CATEGORICAL_RAW, ID_COL, NUMERIC_RAW, RAW_COLUMNS, TARGET
from .utils import get_logger

log = get_logger()
WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "churn_ds_workspaces"
WORKSPACE_TTL_HOURS = 6
MAX_TRAIN_ROWS = 20_000
MAX_SCORE_ROWS = 100_000
MIN_TRAIN_ROWS = 300
MIN_MINORITY = 30
PIPELINE_LOCK = threading.Lock()   # one heavy analysis at a time per server process (protects CPU/RAM)

TRAINING_PROFILES = {
    "Quick (about 1-2 minutes, tiny search)": {"modeling": {"fast": True, "cv_folds": 3, "n_jobs": -1},
                                               "explain": {"permutation_repeats": 3, "shap_sample_size": 150}},
    "Balanced (about 3-6 minutes)": {"modeling": {"fast": False, "cv_folds": 3, "n_iter": 4, "n_jobs": -1},
                                     "explain": {"permutation_repeats": 5, "shap_sample_size": 200}},
    "Thorough (10+ minutes, 5-fold CV, full search)": {},
}

_ALIASES = {
    ID_COL: ["customerid", "custid", "customer", "id", "accountid", "subscriberid", "clientid"],
    "tenure": ["tenure", "tenuremonths", "monthsactive", "customertenure", "accountlengthmonths"],
    "MonthlyCharges": ["monthlycharges", "monthlycharge", "monthlybill", "monthlyfee", "monthlyrevenue"],
    "TotalCharges": ["totalcharges", "totalcharge", "totalbilled", "lifetimecharges", "totalspend"],
    "SeniorCitizen": ["seniorcitizen", "senior", "issenior", "seniorflag"],
    "Contract": ["contract", "contracttype", "contractterm"],
    "InternetService": ["internetservice", "internettype", "internetplan"],
    "PaymentMethod": ["paymentmethod", "payment", "paymenttype"],
    "PaperlessBilling": ["paperlessbilling", "paperless"],
    TARGET: ["churn", "churned", "churnlabel", "exited", "attrition", "left", "target", "customerchurn"],
}


def auto_map_columns(columns) -> dict[str, str | None]:
    """required column -> best matching source column (case/space/underscore-insensitive + aliases)."""
    cols = list(columns)
    keyed = {_norm_key(c): c for c in cols}
    used: set[str] = set()
    out: dict[str, str | None] = {}
    for req in RAW_COLUMNS:
        found = None
        for k in [_norm_key(req), *_ALIASES.get(req, [])]:
            c = keyed.get(k)
            if c is not None and c not in used:
                found = c
                break
        out[req] = found
        if found:
            used.add(found)
    return out


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for req in RAW_COLUMNS:
        src = mapping.get(req)
        if src and src in df.columns:
            out[req] = df[src]
    if ID_COL not in out.columns:
        out.insert(0, ID_COL, [f"ROW-{i + 1:05d}" for i in range(len(out))])
    return out


def _invalid_share(series: pd.Series, col: str) -> float:
    canon, s = _CANON[col], series.dropna()
    if s.empty:
        return 0.0
    bad = 0
    for v in s:
        vv = int(v) if isinstance(v, float) and float(v).is_integer() else v
        if _norm_key(vv) not in canon:
            bad += 1
    return bad / len(s)


def prepare_upload(df: pd.DataFrame, mapping: dict[str, str | None], mode: str,
                   seed: int = 42) -> tuple[pd.DataFrame | None, dict]:
    """Validate + standardise an upload. mode: 'train' (needs Churn labels) or 'score'.
    Returns (frame in the project's raw schema or None, report with errors/warnings)."""
    errors, warnings = [], []
    need_target = mode == "train"
    required = [c for c in RAW_COLUMNS if c != ID_COL and (need_target or c != TARGET)]
    missing = [c for c in required if not mapping.get(c)]
    if missing:
        errors.append("Missing required columns (map them manually or add them to your file): " + ", ".join(missing))
    report = {"errors": errors, "warnings": warnings, "n_rows_in": int(len(df)), "missing_columns": missing,
              "mapping": mapping, "ok": False}
    if errors:
        return None, report
    d = apply_mapping(df, mapping)
    if not mapping.get(ID_COL):
        warnings.append("No customer ID column found: row numbers were used as IDs.")
    elif d[ID_COL].duplicated().any():
        warnings.append(f"{int(d[ID_COL].duplicated().sum())} duplicate customer IDs - duplicates are dropped when training.")
    limit = MAX_TRAIN_ROWS if need_target else MAX_SCORE_ROWS
    for col in NUMERIC_RAW:
        if col in d.columns:
            share = float(pd.to_numeric(d[col], errors="coerce").isna().mean() - d[col].isna().mean())
            if share > 0.30:
                errors.append(f"Column '{col}' is mostly non-numeric ({100 * share:.0f}% unparsable) - check the mapping.")
            elif share > 0:
                warnings.append(f"{100 * share:.1f}% of '{col}' values are not numeric and will be treated as missing.")
    for col in CATEGORICAL_RAW + ([TARGET] if need_target else []):
        if col in d.columns:
            sh = _invalid_share(d[col], col)
            if sh > 0.50:
                errors.append(f"Column '{col}' contains values that do not look like the expected categories "
                              f"{ALLOWED_VALUES[col]} ({100 * sh:.0f}% unrecognised) - check the mapping.")
            elif sh > 0.02:
                warnings.append(f"{100 * sh:.1f}% of '{col}' values are unrecognised (expected {ALLOWED_VALUES[col]}); "
                                "they are treated as missing and imputed.")
    if len(d) == 0:
        errors.append("The file has no rows.")
    if need_target and not errors:
        lab = d[TARGET].map(lambda v: _CANON[TARGET].get(_norm_key(int(v) if isinstance(v, float) and float(v).is_integer() else v)))
        if lab.isna().all():
            errors.append("Could not read the churn labels (use Yes/No, 1/0 or True/False).")
        else:
            counts = lab.value_counts()
            report["class_counts"] = counts.to_dict()
            if len(counts) < 2 or counts.min() < MIN_MINORITY:
                errors.append(f"Training needs both outcomes with at least {MIN_MINORITY} customers each "
                              f"(found {counts.to_dict()}). Use 'Score with the pre-trained model' for small files.")
            if len(d) < MIN_TRAIN_ROWS:
                errors.append(f"Training needs at least {MIN_TRAIN_ROWS} rows (found {len(d)}). "
                              "Use 'Score with the pre-trained model' instead.")
    if not errors and len(d) > limit:
        if need_target:
            lab = d[TARGET].astype(str)
            _, d = train_test_split(d, test_size=limit, stratify=lab, random_state=seed)
            d = d.reset_index(drop=True)
        else:
            d = d.head(limit)
        warnings.append(f"File has {report['n_rows_in']:,} rows; limited to {limit:,} rows to keep the hosted app responsive.")
    report["n_rows_used"] = int(len(d))
    report["ok"] = not errors
    return (d if report["ok"] else None), report


# ------------------------------------------------------------------------------ workspaces
def cleanup_old_workspaces(max_age_hours: float = WORKSPACE_TTL_HOURS) -> int:
    if not WORKSPACE_ROOT.exists():
        return 0
    n, cutoff = 0, time.time() - max_age_hours * 3600
    for p in WORKSPACE_ROOT.iterdir():
        try:
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n


def create_workspace(profile: str | None = None, base_cfg: dict | None = None,
                     root: Path | None = None) -> tuple[dict, Path]:
    """Private folder + config whose every path points inside it."""
    cleanup_old_workspaces()
    base_cfg = base_cfg or load_config()
    ws = Path(root) if root else WORKSPACE_ROOT / uuid.uuid4().hex[:12]
    ws.mkdir(parents=True, exist_ok=True)
    p = lambda *a: str(ws.joinpath(*a))  # noqa: E731
    overrides = {"paths": {"raw_data": p("data", "raw", "upload.csv"), "clean_data": p("data", "processed", "clean.csv"),
                           "modeling_data": p("data", "processed", "modeling.csv"), "models_dir": p("outputs", "models"),
                           "figures_dir": p("outputs", "figures"), "reports_dir": p("outputs", "reports"),
                           "tables_dir": p("outputs", "tables"), "logs_dir": p("outputs", "logs"),
                           "history_file": p("outputs", "prediction_history.csv")}}
    for k, v in TRAINING_PROFILES.get(profile or "", {}).items():
        overrides.setdefault(k, {}).update(v)
    return load_config(overrides=_merge_over(base_cfg, overrides)), ws


def _merge_over(base_cfg: dict, overrides: dict) -> dict:
    """Overrides on top of the *given* base config (load_config merges onto config.yaml; re-apply base values)."""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_cfg.items()}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def run_workspace_pipeline(cfg: dict, prepared: pd.DataFrame, progress=None) -> dict:
    """Write the validated upload as the workspace's raw data and run every analysis step on it."""
    from .config import resolve_path
    from .pipeline import run_step
    steps = ["clean", "eda", "stats", "train", "evaluate", "explain", "insights", "report"]
    timings = {}
    with PIPELINE_LOCK:
        raw_path = resolve_path(cfg, "raw_data", create=True)
        prepared.to_csv(raw_path, index=False)
        for i, step in enumerate(steps):
            if progress:
                progress(step, i, len(steps))
            t0 = time.time()
            run_step(step, cfg)
            timings[step] = round(time.time() - t0, 1)
    return timings


# ------------------------------------------------------------------------------ batch scoring
def score_batch(df_prepared: pd.DataFrame, bundle: dict, cfg: dict, threshold: float | None = None):
    """Score customers with an already-trained model. Returns (action_table, revenue_summary)."""
    thr = float(threshold) if threshold is not None else resolve_threshold(cfg, bundle)
    std = standardize_frame(df_prepared)
    eng = engineer_features(std, cfg)
    eng["churn_probability"] = predict_proba(bundle, eng)
    eng["score_source"] = "pre-trained model"
    if TARGET not in eng.columns:
        eng[TARGET] = "Unknown"
    table = assemble_customer_table(eng, cfg, thr)
    return table, revenue_summary(table, cfg)
