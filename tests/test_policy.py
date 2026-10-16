"""Step 6 policy training and closed-loop contract tests."""

import json
from dataclasses import asdict

import pytest
import torch

from cmb_minigrid.data.dataset import split_records, write_manifest
from cmb_minigrid.data.trajectory import collect_records, write_jsonl
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.models.policy import ActorPolicy
from cmb_minigrid.training.policy import (
    PolicyTrainingConfig,
    _class_weights,
    _policy_examples,
    load_policy,
    run_closed_loop,
    train_policy,
)
from cmb_minigrid.training.rssm import fingerprint


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def policy_data(tmp_path):
    records = collect_records(
        pair_count=1,
        seed=42,
        map_family_ids=["hidden_history_nav_v1", "hidden_history_nav_v2", "hidden_history_nav_v3"],
        include_exploration=False,
    )
    input_path = write_jsonl(records, tmp_path / "data.jsonl")
    manifest_path = write_manifest(split_records(records), tmp_path / "manifest.json")
    return records, input_path, manifest_path


def make_rssm_checkpoint(path, *, deter=32):
    config = RSSMConfig(deter=deter, encoder_type="cnn")
    model = RSSM(config)
    torch.save({
        "schema_version": 1,
        "config": {"rssm": asdict(config), "seed": 42},
        "model": model.state_dict(),
    }, path)
    return path


def test_actor_shape_goal_is_public_constant_and_bad_shape_rejected():
    policy = ActorPolicy(64, hidden_dim=16)
    features = torch.zeros(3, 64)
    assert policy(features).shape == (3, 3)
    assert torch.equal(policy._goal(features, None), torch.ones(3, 1))
    with pytest.raises(Exception, match="features"):
        policy(torch.zeros(3, 63))


def test_class_weighting_modes_are_distinct_and_normalized():
    counts = torch.tensor([60.0, 60.0, 1120.0])
    none = _class_weights(counts, "none")
    inverse = _class_weights(counts, "inverse")
    square_root = _class_weights(counts, "sqrt")
    assert torch.equal(none, torch.ones(3))
    assert inverse[0] > square_root[0] > none[0]
    assert inverse[2] < square_root[2] < none[2]
    torch.testing.assert_close(inverse.mean(), torch.tensor(1.0))
    torch.testing.assert_close(square_root.mean(), torch.tensor(1.0))


def test_policy_examples_are_detached_and_rssm_is_frozen(policy_data):
    records, _, _ = policy_data
    model = RSSM(RSSMConfig(deter=32, encoder_type="cnn"))
    features, labels = _policy_examples(model, [record for record in records if record.episode_role == "expert"][:2])
    assert features.requires_grad is False
    assert features.shape[0] == labels.shape[0]
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())


def test_smoke_training_writes_loadable_checkpoints(policy_data, tmp_path):
    _, input_path, manifest_path = policy_data
    rssm_path = make_rssm_checkpoint(tmp_path / "rssm.pt")
    report = train_policy(
        rssm_checkpoint=rssm_path,
        input_path=input_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "policy",
        seed=45,
        device="cpu",
        mode="smoke",
        training=PolicyTrainingConfig(max_steps=2, val_every=1, patience=2, batch_size=8,
                                      hidden_dim=16, num_threads=1),
    )
    assert report["status"] == "loss_decreased"
    assert report["steps"] == 100
    actor, checkpoint = load_policy(report["best_checkpoint"])
    assert actor.feature_dim == 32 + 16 * 16
    assert checkpoint["config"]["rssm_checkpoint_sha256"] == fingerprint(rssm_path)
    assert report["class_weighting"] is True
    assert len(report["class_weights"]) == 3
    assert 0.0 <= report["final_train"]["accuracy"] <= 1.0
    validation_events = [
        json.loads(line) for line in (tmp_path / "policy" / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "validation"
    ]
    assert validation_events
    assert "train_metrics" in validation_events[-1]
    assert "accuracy" in validation_events[-1]["train_metrics"]
    assert (tmp_path / "policy" / "last.pt").is_file()
    assert (tmp_path / "policy" / "report.json").is_file()


def test_closed_loop_records_all_requested_fields(policy_data):
    records, _, _ = policy_data
    record = next(record for record in records if record.episode_role == "expert")
    rssm = RSSM(RSSMConfig(deter=32, encoder_type="cnn"))
    policy = ActorPolicy(32 + 16 * 16, hidden_dim=16)
    row = run_closed_loop(record, rssm, policy, mode="reset_b", device="cpu")
    assert row["mode"] == "reset_b"
    assert len(row["predicted_actions"]) == row["steps"]
    assert row["predicted_handoff_action"] in (0, 1, 2)
    assert isinstance(row["branch_correct"], bool)
    assert "first_error_t" in row
    assert row["steps_before_first_error"] <= row["steps"]
