"""Shared helpers for the data pipeline: config loading, environment setup, memory logging.

`setup_env()` must run before any Hugging Face library is imported so caches and
temp files land on D: (C: has ~5 GB free) and console output is UTF-8 (the
Windows console defaults to cp1252, which crashes on Bengali text and emoji).

Run standalone to print the resolved configuration:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.common
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_config(path: Path = DATA_CONFIG) -> dict[str, Any]:
    """Load configs/data.yaml."""
    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def setup_env(config: dict[str, Any] | None = None) -> None:
    """Point HF caches and temp dirs at the D: cache dir and force UTF-8 stdout/stderr."""
    config = config or load_config()
    cache = Path(config["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "hf_datasets")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(cache / "tmp")
    (cache / "tmp").mkdir(exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path from the config."""
    return REPO_ROOT / rel


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Windows: peak working set)."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return float(peak) / 2**20


def sha256_file(path: Path, chunk: int = 2**22) -> str:
    """Return the hex sha256 of a file, streaming it in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Write pretty JSON with UTF-8 and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cfg = load_config()
    setup_env(cfg)
    print("repo root:", REPO_ROOT)
    print("sources:", ", ".join(cfg["sources"]))
    print("HF_HOME:", os.environ["HF_HOME"])
    print(f"peak RSS: {peak_rss_mb():.1f} MB")
