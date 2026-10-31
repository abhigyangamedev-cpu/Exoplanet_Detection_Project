"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve every entry under ``paths`` to an absolute Path."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["paths"] = {
        name: (REPO_ROOT / value).resolve() for name, value in config.get("paths", {}).items()
    }
    return config


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create all configured directories if they do not exist."""
    for directory in config["paths"].values():
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs.

    Returns a seeded ``np.random.Generator`` for project code; the legacy global NumPy
    state is also seeded because third-party libraries still draw from it.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
