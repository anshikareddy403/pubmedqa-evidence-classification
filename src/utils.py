"""Shared utilities: seeding, paths, timing."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


@contextmanager
def timer(name: str) -> Generator[Dict[str, float], None, None]:
    """Context manager that records elapsed wall time in seconds."""
    record: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed_seconds"] = time.perf_counter() - start
        record["label"] = name  # type: ignore[assignment]
