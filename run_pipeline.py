#!/usr/bin/env python
"""Command-line entry point.

  python run_pipeline.py seed [--sample] [--force]   download/load the dataset into data/raw
  python run_pipeline.py clean                       clean + data-quality report + modeling-ready data
  python run_pipeline.py eda | stats                 exploratory analysis | hypothesis tests
  python run_pipeline.py train [--fast]              tune, compare, select and save the model
  python run_pipeline.py evaluate | explain          test-set evaluation | explainability
  python run_pipeline.py insights | report           revenue at risk + recommendations | final reports
  python run_pipeline.py all [--sample] [--fast]     everything, in order
"""
import argparse
import sys
import time

from churn.config import load_config
from churn.pipeline import STEPS, run_all, run_step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="churn_ds pipeline")
    ap.add_argument("command", choices=STEPS + ["quality", "all"])
    ap.add_argument("--sample", action="store_true", help="use the bundled sample dataset (offline)")
    ap.add_argument("--force", action="store_true", help="re-download even if data/raw exists")
    ap.add_argument("--fast", action="store_true", help="tiny hyperparameter grids / 3-fold CV (quick demo)")
    ap.add_argument("--threshold", type=float, default=None, help="fixed probability threshold override")
    ap.add_argument("--config", default=None, help="path to an alternative config.yaml")
    a = ap.parse_args(argv)

    overrides = {}
    if a.fast:
        overrides["modeling"] = {"fast": True, "cv_folds": 3}
    if a.threshold is not None:
        overrides["threshold"] = {"mode": "fixed", "value": a.threshold}
    cfg = load_config(a.config, overrides or None)
    t0 = time.time()
    if a.command == "all":
        timings = run_all(cfg, sample=a.sample, force=a.force)
        print("\nStep timings (s):", timings)
    else:
        run_step(a.command, cfg, sample=a.sample, force=a.force)
    print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
