import json
import subprocess
import sys
from dataclasses import fields

import numpy as np

from cmb_minigrid.environments.memory import ACTION_NAMES, TASK_ID, make_memory_env
from cmb_minigrid.environments.rollout import rollout


def test_memory_adapter_exposes_only_three_actions_and_public_observation():
    env = make_memory_env()
    try:
        obs, info = env.reset(seed=42)
        assert env.action_space.n == 3
        assert ACTION_NAMES == ("left", "right", "forward")
        assert info == {}
        assert env.observation_space.contains(obs)
        assert obs.image.shape == (7, 7, 3)
        assert obs.image.dtype == np.uint8
        assert obs.direction in range(4)
        assert obs.task_id == TASK_ID
        assert {field.name for field in fields(obs)} == {"image", "direction", "task_id"}
        assert not any(
            hasattr(obs, name)
            for name in ("global_map", "env_state", "hidden_label", "correct_branch", "expert_action")
        )

        next_obs, _, terminated, truncated, step_info = env.step(2)
        assert env.observation_space.contains(next_obs)
        assert not terminated and not truncated
        assert step_info == {}
    finally:
        env.close()


def test_same_seed_and_actions_replay_identical_observations():
    actions = (2, 2, 1, 2, 0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 0)
    env = make_memory_env()
    try:
        first = rollout(env, seed=42, steps=len(actions), actions=actions)
        second = rollout(env, seed=42, steps=len(actions), actions=actions)
    finally:
        env.close()

    assert first.actions == second.actions == actions
    assert first.observation_hashes == second.observation_hashes
    assert len(first.observations) == len(first.actions) + 1
    assert all(len(value) == 64 for value in first.observation_hashes)


def test_rgb_array_render_is_available():
    env = make_memory_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[-1] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_smoke_cli_writes_twenty_step_sample_without_privileged_fields(tmp_path):
    sample = tmp_path / "rollout.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmb_minigrid.scripts.smoke_env",
            "--seed",
            "42",
            "--steps",
            "20",
            "--output",
            str(sample),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "transitions=20" in result.stdout
    data = json.loads(sample.read_text(encoding="utf-8"))
    assert len(data["actions"]) == 20
    assert len(data["observations"]) == 21
    assert len(data["observation_hashes"]) == 21
    assert set(data["observations"][0]) == {"image", "direction", "task_id"}
    assert data["observations"][0]["task_id"] == TASK_ID
