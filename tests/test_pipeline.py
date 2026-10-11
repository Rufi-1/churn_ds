import subprocess
import sys
from pathlib import Path

import pandas as pd

from churn.config import resolve_path
from churn.data_loader import seed_data, download_dataset
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_seed_falls_back_to_sample_when_download_fails(blank_cfg, monkeypatch):
    import churn.data_loader as dl
    monkeypatch.setattr(dl, "download_dataset", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    meta = seed_data(blank_cfg)
    assert "sample" in meta["source"] and resolve_path(blank_cfg, "raw_data").exists() and meta["rows"] == 500


def test_seed_keeps_existing_file(pipeline_run):
    meta = seed_data(pipeline_run["cfg"])
    assert meta["source"] == "existing file"


def test_download_rejects_invalid_file(tmp_path, monkeypatch):
    import urllib.request

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"a,b\n1,2\n"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: R())
    with pytest.raises(ValueError):
        download_dataset("http://x", tmp_path / "d.csv")
    assert not (tmp_path / "d.csv").exists()


def test_cli_help_and_bad_command():
    ok = subprocess.run([sys.executable, "run_pipeline.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0 and "train" in ok.stdout
    bad = subprocess.run([sys.executable, "run_pipeline.py", "nonsense"], cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode != 0


def test_modeling_data_reproducible(pipeline_run):
    cfg = pipeline_run["cfg"]
    a = pd.read_csv(resolve_path(cfg, "modeling_data"))
    from churn.pipeline import run_step
    run_step("clean", cfg)
    b = pd.read_csv(resolve_path(cfg, "modeling_data"))
    pd.testing.assert_frame_equal(a, b)
