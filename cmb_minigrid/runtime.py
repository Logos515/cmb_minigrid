"""Device selection and run-directory management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RuntimeConfig
from .reproducibility import experiment_id, set_seed


def select_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    return device


@dataclass(frozen=True)
class RunContext:
    config: RuntimeConfig
    device: object
    experiment_id: str
    log_dir: Path

    @classmethod
    def create(cls, config: RuntimeConfig) -> RunContext:
        set_seed(config.seed, deterministic=config.deterministic)
        device = select_device(config.device)
        run_id = experiment_id(config.experiment_name, config.seed, config=vars(config))
        log_dir = config.output_dir / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        return cls(config, device, run_id, log_dir)
