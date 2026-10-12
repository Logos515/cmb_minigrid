"""Reproducibility helpers used by every future experiment entry point."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def experiment_id(experiment_name: str, seed: int, *, config: Any = None) -> str:
    if not experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    payload = {"experiment_name": experiment_name, "seed": seed, "config": config}
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:10]
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in experiment_name)
    return f"{safe_name}-{digest}"
