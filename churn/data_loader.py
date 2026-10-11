"""Dataset download / seeding and loading helpers (raw, cleaned, modeling-ready)."""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from .config import load_config, resolve_path
from .schema import ID_COL, TARGET
from .utils import get_logger, now_iso, save_json, sha256_file

log = get_logger()


def download_dataset(url: str, dest: Path, timeout: int = 30) -> Path:
    """Download the CSV and validate that it looks like the Telco churn file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "churn_ds/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted, configurable URL)
        payload = resp.read()
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(payload)
    header = pd.read_csv(tmp, nrows=5)
    if ID_COL not in header.columns or TARGET not in header.columns:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid churn dataset (columns: {list(header.columns)})")
    tmp.replace(dest)
    return dest


def seed_data(cfg: dict | None = None, use_sample: bool = False, force: bool = False) -> dict:
    """Make sure data/raw/telco_churn.csv exists.

    Order: --sample flag -> existing file (unless force) -> download -> bundled sample fallback.
    """
    cfg = cfg or load_config()
    raw = resolve_path(cfg, "raw_data", create=True)
    sample = resolve_path(cfg, "sample_data")
    source, note = None, ""
    if use_sample:
        shutil.copyfile(sample, raw)
        source = "bundled sample"
    elif raw.exists() and not force:
        source = "existing file"
    else:
        try:
            log.info("Downloading dataset from %s", cfg["data"]["download_url"])
            download_dataset(cfg["data"]["download_url"], raw)
            source = "downloaded (IBM Telco Customer Churn)"
        except Exception as exc:  # noqa: BLE001 - offline fallback is intentional
            log.warning("Download failed (%s). Falling back to the bundled sample dataset.", exc)
            shutil.copyfile(sample, raw)
            source, note = "bundled sample (download failed)", str(exc)
    df = pd.read_csv(raw)
    meta = {"source": source, "note": note, "rows": int(len(df)), "columns": int(df.shape[1]),
            "sha256_12": sha256_file(raw), "path": str(raw), "seeded_at": now_iso(),
            "url": cfg["data"]["download_url"]}
    save_json(meta, raw.parent / "dataset_meta.json")
    log.info("Raw data ready: %s rows from %s -> %s", meta["rows"], source, raw)
    return meta


def _need(path: Path, hint: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `{hint}` first.")
    return Path(path)


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(_need(resolve_path(cfg, "raw_data"), "python run_pipeline.py seed"))


def load_clean(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(_need(resolve_path(cfg, "clean_data"), "python run_pipeline.py clean"))


def load_modeling(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(_need(resolve_path(cfg, "modeling_data"), "python run_pipeline.py clean"))


def dataset_version(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    raw = resolve_path(cfg, "raw_data")
    return sha256_file(raw) if raw.exists() else "unknown"
