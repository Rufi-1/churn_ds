"""End-to-end orchestration: raw data -> final report. Used by run_pipeline.py and the dashboard."""
from __future__ import annotations

import time

from .cleaning import clean_dataset
from .config import load_config, resolve_path
from .data_loader import load_clean, load_raw, seed_data
from .eda import run_eda
from .evaluation import evaluate_model
from .explain import run_explainability
from .features import prepare_modeling_data
from .insights import run_business_insights
from .modeling import train_and_save
from .quality import run_quality_checks
from .reporting import generate_reports
from .stats_tests import run_statistical_tests
from .utils import get_logger, now_iso, save_json

STEPS = ["seed", "clean", "eda", "stats", "train", "evaluate", "explain", "insights", "report"]


def run_step(name: str, cfg: dict, sample: bool = False, force: bool = False):
    log = get_logger("churn", resolve_path(cfg, "logs_dir", create=True) / "pipeline.log")
    if name == "seed":
        return seed_data(cfg, use_sample=sample, force=force)
    if name in ("clean", "quality"):
        raw = load_raw(cfg)
        clean, clog = clean_dataset(raw, cfg)
        p = resolve_path(cfg, "clean_data", create=True)
        clean.to_csv(p, index=False)
        run_quality_checks(raw, clean, clog, cfg)
        prepare_modeling_data(clean, cfg)
        log.info("Clean data -> %s (%d rows); modeling data -> %s", p, len(clean), resolve_path(cfg, "modeling_data"))
        return clean
    if name == "eda":
        return run_eda(load_clean(cfg), cfg)
    if name == "stats":
        return run_statistical_tests(load_clean(cfg), cfg)
    if name == "train":
        return train_and_save(cfg)
    if name == "evaluate":
        return evaluate_model(cfg)
    if name == "explain":
        return run_explainability(cfg)
    if name == "insights":
        return run_business_insights(cfg)
    if name == "report":
        return generate_reports(cfg)
    raise ValueError(f"Unknown step '{name}'. Choose from {STEPS}")


def run_all(cfg: dict | None = None, sample: bool = False, force: bool = False, progress=None) -> dict:
    """progress(step_name, index, total) is called before each step (used by the dashboard)."""
    cfg = cfg or load_config()
    timings = {}
    for i, step in enumerate(STEPS):
        if progress:
            progress(step, i, len(STEPS))
        t0 = time.time()
        run_step(step, cfg, sample=sample, force=force)
        timings[step] = round(time.time() - t0, 1)
    save_json({"finished_at": now_iso(), "seconds_per_step": timings, "total_seconds": round(sum(timings.values()), 1)},
              resolve_path(cfg, "logs_dir", create=True) / "pipeline_run.json")
    return timings
