"""Data-quality assessment: missing values, duplicates, invalid categories, types, distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .schema import ALLOWED_VALUES, CATEGORICAL_RAW, ID_COL, NUMERIC_RAW, TARGET
from .utils import df_to_markdown, get_logger, now_iso, save_json

log = get_logger()
_RAW_EQUIV = {"SeniorCitizen": {"0", "1", "0.0", "1.0"}}


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts NaN *and* blank/whitespace-only strings."""
    rows = []
    for c in df.columns:
        s = df[c]
        blank = s.astype(str).str.strip().eq("") & s.notna()
        n = int(s.isna().sum() + blank.sum())
        rows.append({"column": c, "missing": n, "missing_pct": 100 * n / max(len(df), 1),
                     "blank_strings": int(blank.sum())})
    return pd.DataFrame(rows).sort_values("missing", ascending=False).reset_index(drop=True)


def dtype_table(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in raw.columns:
        rows.append({"column": c, "raw_dtype": str(raw[c].dtype),
                     "clean_dtype": str(clean[c].dtype) if c in clean else "-",
                     "unique_values_raw": int(raw[c].nunique(dropna=True)),
                     "unique_values_clean": int(clean[c].nunique(dropna=True)) if c in clean else 0})
    return pd.DataFrame(rows)


def invalid_category_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, allowed in ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        ok = {str(a) for a in allowed} | _RAW_EQUIV.get(col, set())
        vc = df[col].dropna().astype(str).str.strip().value_counts()
        for val, n in vc.items():
            if val not in ok and val != "":
                rows.append({"column": col, "invalid_value": val, "count": int(n)})
    return pd.DataFrame(rows, columns=["column", "invalid_value", "count"])


def numeric_summary(df: pd.DataFrame, cols=NUMERIC_RAW) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        rows.append({"column": c, "count": len(s), "mean": s.mean(), "std": s.std(), "min": s.min(),
                     "25%": q1, "median": s.median(), "75%": q3, "max": s.max(), "skew": s.skew(),
                     "zeros": int((s == 0).sum()),
                     "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
    return pd.DataFrame(rows)


def categorical_distribution(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = [c for c in (cols or CATEGORICAL_RAW + [TARGET]) if c in df.columns]
    parts = []
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        parts.append(pd.DataFrame({"column": c, "value": vc.index.astype(str), "count": vc.values,
                                   "pct": 100 * vc.values / len(df)}))
    return pd.concat(parts, ignore_index=True)


def run_quality_checks(raw: pd.DataFrame, clean: pd.DataFrame, cleaning_log: pd.DataFrame,
                       cfg: dict | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    overview = {
        "generated_at": now_iso(),
        "raw_rows": int(len(raw)), "raw_columns": int(raw.shape[1]),
        "clean_rows": int(len(clean)), "clean_columns": int(clean.shape[1]),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "duplicate_ids_raw": int(raw[ID_COL].duplicated().sum()) if ID_COL in raw else 0,
        "duplicate_rows_clean": int(clean.duplicated().sum()),
        "missing_cells_raw": int(missing_values_table(raw)["missing"].sum()),
        "missing_cells_clean": int(clean.isna().sum().sum()),
        "churn_rate": float((clean[TARGET] == "Yes").mean()) if TARGET in clean else None,
        "memory_mb_clean": float(clean.memory_usage(deep=True).sum() / 1e6),
    }
    tables = {
        "quality_missing_raw": missing_values_table(raw),
        "quality_missing_clean": missing_values_table(clean),
        "quality_dtypes": dtype_table(raw, clean),
        "quality_invalid_categories": invalid_category_table(raw),
        "quality_numeric_summary": numeric_summary(clean),
        "quality_categorical_distribution": categorical_distribution(clean),
        "quality_outliers": __import__("churn.cleaning", fromlist=["x"]).detect_outliers(clean),
        "cleaning_log": cleaning_log,
    }
    report = {"overview": overview, "tables": tables}
    if save:
        tdir = resolve_path(cfg, "tables_dir", create=True)
        rdir = resolve_path(cfg, "reports_dir", create=True)
        ldir = resolve_path(cfg, "logs_dir", create=True)
        for name, t in tables.items():
            t.to_csv((ldir if name == "cleaning_log" else tdir) / f"{name}.csv", index=False)
        save_json(overview, tdir / "quality_overview.json")
        save_json(tables["cleaning_log"].to_dict("records"), ldir / "cleaning_log.json")
        (rdir / "data_quality_report.md").write_text(quality_markdown(report), encoding="utf-8")
        log.info("Data-quality report saved to %s", rdir / "data_quality_report.md")
    return report


def quality_markdown(report: dict) -> str:
    o, t = report["overview"], report["tables"]
    inv = t["quality_invalid_categories"]
    lines = [
        "# Data Quality Report", f"_Generated {o['generated_at']}_", "",
        "## Overview",
        f"- Raw rows x columns: **{o['raw_rows']:,} x {o['raw_columns']}**; clean rows: **{o['clean_rows']:,}**",
        f"- Duplicate rows (raw): **{o['duplicate_rows_raw']}**; duplicate customer IDs (raw): **{o['duplicate_ids_raw']}**",
        f"- Missing cells raw (incl. blank strings): **{o['missing_cells_raw']}**; after cleaning: **{o['missing_cells_clean']}**",
        f"- Invalid category values found in raw data: **{len(inv)}** distinct values",
        f"- Overall churn rate: **{100 * (o['churn_rate'] or 0):.2f}%**", "",
        "## Missing values (raw)", df_to_markdown(t["quality_missing_raw"].query("missing > 0"), 2), "",
        "## Invalid categories (raw)", df_to_markdown(inv), "",
        "## Data types and cardinality", df_to_markdown(t["quality_dtypes"]), "",
        "## Numeric summary (clean)", df_to_markdown(t["quality_numeric_summary"], 2), "",
        "## Cleaning log", df_to_markdown(t["cleaning_log"]), "",
    ]
    return "\n".join(lines)
