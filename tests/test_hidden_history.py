import json
import subprocess
import sys

import numpy as np

from cmb_minigrid.environments.hidden_history import (
    HANDOFF_T,
    HiddenHistoryNavEnv,
    generate_counterfactual_pair,
    load_snapshot,
    save_snapshot,
    validate_pair,
)


def test_counterfactual_pair_has_exact_shared_handoff_history_and_different_branch():
    pair = generate_counterfactual_pair(42)
    report = validate_pair(pair, k=8)
    assert report.passed, report.checks
    assert pair.left.observations[HANDOFF_T] == pair.right.observations[HANDOFF_T]
    assert pair.left.actions[:HANDOFF_T] == pair.right.actions[:HANDOFF_T]
    assert pair.left.expert_actions[HANDOFF_T] != pair.right.expert_actions[HANDOFF_T]
    assert pair.left.reached_goal and pair.right.reached_goal
    assert pair.left.observations[0].task_id == "hidden_history_nav"


def test_snapshot_restore_replays_same_public_observation_and_future():
    pair = generate_counterfactual_pair(7)
    snapshot = pair.left.snapshots[HANDOFF_T]
    env = HiddenHistoryNavEnv(hidden_label="right_blocked")
    restored = env.restore(snapshot)
    expected = pair.left.observations[HANDOFF_T]
    assert restored == expected
    observed = []
    for action in pair.left.actions[HANDOFF_T:]:
        obs, _, terminated, truncated, _ = env.step(action)
        observed.append(obs)
        if terminated or truncated:
            break
    env.close()
    assert tuple(observed) == pair.left.observations[HANDOFF_T + 1 :]


def test_snapshot_json_round_trip(tmp_path):
    pair = generate_counterfactual_pair(9)
    path = save_snapshot(pair.left.snapshots[HANDOFF_T], tmp_path / "handoff.json")
    loaded = load_snapshot(path)
    assert loaded == pair.left.snapshots[HANDOFF_T]


def test_hidden_history_observation_does_not_include_privileged_label_or_map():
    env = HiddenHistoryNavEnv(hidden_label="left_blocked")
    obs, info = env.reset(seed=3, options={"hidden_label": "left_blocked"})
    assert info == {}
    assert obs.task_id == "hidden_history_nav"
    assert obs.image.dtype == np.uint8
    assert not hasattr(obs, "hidden_label")
    assert not hasattr(obs, "global_map")
    env.close()


def test_pair_cli_generates_requested_batch_and_reports_sample_size(tmp_path):
    output = tmp_path / "pairs.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmb_minigrid.scripts.smoke_hidden_history",
            "--count",
            "5",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "pairs=5 validated=5 reachable=5" in result.stdout
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["count"] == 5
    assert data["random_no_history_correct"] <= 5
