"""YAML configuration loading and validation for Step 0."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for an invalid or unreadable experiment configuration."""


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("runs")
    experiment_name: str = "cmb_minigrid"
    deterministic: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RuntimeConfig:
        if not isinstance(values, Mapping):
            raise ConfigError("configuration root must be a YAML mapping")
        seed = values.get("seed", cls.seed)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigError("seed must be a non-negative integer")
        device = values.get("device", cls.device)
        if not isinstance(device, str) or not device:
            raise ConfigError("device must be a non-empty string (auto, cpu, or cuda[:N])")
        if device != "auto" and device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", device):
            raise ConfigError(f"unsupported device {device!r}; use auto, cpu, or cuda[:N]")
        output_dir = values.get("output_dir", cls.output_dir)
        if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
            raise ConfigError("output_dir must be a non-empty path")
        experiment_name = values.get("experiment_name", cls.experiment_name)
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ConfigError("experiment_name must be a non-empty string")
        deterministic = values.get("deterministic", cls.deterministic)
        if not isinstance(deterministic, bool):
            raise ConfigError("deterministic must be true or false")
        return cls(seed, device, Path(output_dir), experiment_name.strip(), deterministic)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ConfigError(f"configuration root in {config_path} must be a mapping")
    return values


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.from_mapping({} if path is None else load_yaml(path))
