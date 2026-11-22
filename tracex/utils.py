"""Small shared utilities: reproducibility, IDs and safe filesystem helpers."""

from __future__ import annotations

import os
import random
import re
import uuid
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def safe_filename(name: str, fallback: str = "upload.bin") -> str:
    base = Path(name or fallback).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or fallback


def ensure_within(path: str | Path, root: str | Path) -> Path:
    """Resolve a path and refuse traversal outside ``root``."""

    root = Path(root).resolve()
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Path escapes the run directory")
    return candidate
