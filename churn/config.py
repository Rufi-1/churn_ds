"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load config.yaml (or $CHURN_CONFIG) and apply optional nested overrides."""
    cfg_path = Path(path or os.environ.get("CHURN_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def resolve_path(cfg: dict, key: str, create: bool = False) -> Path:
    """Resolve a path from cfg['paths'] (relative paths are relative to the project root)."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if create:
        target = p if key.endswith("_dir") else p.parent
        target.mkdir(parents=True, exist_ok=True)
    return p
