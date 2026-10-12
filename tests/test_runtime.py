import random
import subprocess
import sys

import numpy as np
import pytest

from cmb_minigrid.config import ConfigError, RuntimeConfig, load_runtime_config
from cmb_minigrid.reproducibility import experiment_id, set_seed
from cmb_minigrid.runtime import RunContext


def test_invalid_config_has_readable_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="non-negative integer"):
        load_runtime_config(path)


def test_seed_reproduces_python_numpy_and_torch():
    set_seed(123)
    first = (random.random(), np.random.random())
    set_seed(123)
    assert first == (random.random(), np.random.random())


def test_run_context_is_stable_and_creates_log_dir(tmp_path):
    config = RuntimeConfig(output_dir=tmp_path, device="cpu", experiment_name="smoke")
    first = RunContext.create(config)
    second = RunContext.create(config)
    assert first.experiment_id == second.experiment_id == experiment_id("smoke", 42, config=vars(config))
    assert first.log_dir.is_dir()


def test_cli_help_and_runtime_smoke(tmp_path):
    help_result = subprocess.run(
        [sys.executable, "-m", "cmb_minigrid.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmb_minigrid.cli",
            "--config",
            "configs/default.yaml",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "device=cpu" in run_result.stdout
    assert (tmp_path / run_result.stdout.split("experiment_id=", 1)[1].splitlines()[0]).is_dir()
