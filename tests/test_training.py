"""Step 5 lifecycle, full-history and frozen-probe checks on CPU."""

import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest
import torch

from cmb_minigrid.data.dataset import read_manifest, split_records, write_manifest
from cmb_minigrid.data.trajectory import collect_records, write_jsonl
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.scripts.train_rssm import main
from cmb_minigrid.training.probe import (
    extract_pair_features,
    fit_linear_probe,
    probe_metrics,
    run_probes,
)
from cmb_minigrid.training.rssm import (
    TrainingConfig,
    evaluate,
    fingerprint,
    load_rssm,
    model_batch,
    train_rssm,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data(tmp_path):
    records = collect_records(pair_count=1, seed=42, map_family_ids=[
        "hidden_history_nav_v1", "hidden_history_nav_v2", "hidden_history_nav_v3"])
    path = write_jsonl(records, tmp_path / "data.jsonl")
    manifest = write_manifest(split_records(records), tmp_path / "manifest.json")
    return path, manifest, records


def config(data, *, seed=42, deter=128, steps=4):
    path, manifest, _ = data
    return {"experiment_name": f"rssm_{deter}", "seed": seed, "device": "cpu",
            "deterministic": True, "rssm": asdict(RSSMConfig(deter=deter)),
            "training": asdict(TrainingConfig(max_steps=steps, batch_size=3,
                                               val_every=2, patience=10)),
            "input": str(path), "manifest": str(manifest),
            "input_sha256": fingerprint(path), "manifest_sha256": fingerprint(manifest),
            "mode": "formal"}


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("recall_scale", [0.0, 5.0])
def test_sampled_resume_matches_uninterrupted_optimizer_rng_and_states(data, tmp_path, recall_scale):
    cfg = config(data)
    cfg["rssm"]["recall_scale"] = recall_scale
    full = train_rssm(cfg, tmp_path / "full")
    short = cfg | {"training": cfg["training"] | {"max_steps": 2}}
    partial = train_rssm(short, tmp_path / "partial")
    resumed = train_rssm(cfg, tmp_path / "unused", resume=partial["last_checkpoint"])
    first_model, first = load_rssm(full["last_checkpoint"])
    second_model, second = load_rssm(resumed["last_checkpoint"])
    for field in ("model", "optimizer", "rng", "sampler", "best_val", "best_step",
                  "bad_validations"):
        assert_nested_equal(first[field], second[field])
    batch = model_batch(data[2][:2], "cpu")
    with torch.no_grad():
        a = first_model.observe(batch["agent_obs"], batch["actions"], batch["observation_mask"])
        b = second_model.observe(batch["agent_obs"], batch["actions"], batch["observation_mask"])
    torch.testing.assert_close(a.features, b.features, rtol=0, atol=0)
    events = [json.loads(line) for line in (
        tmp_path / "partial" / partial["experiment_id"] / "metrics.jsonl").read_text().splitlines()]
    assert [event["step"] for event in events if event["event"] == "train"] == [1, 2, 3, 4]
    assert all(event["gradient_norm_before_clip"] > 0 for event in events
               if event["event"] == "train")


@pytest.mark.parametrize("recall_scale", [0.0, 5.0])
def test_independent_checkpoints_filter_full_history_and_steps(data, tmp_path, recall_scale):
    cfg_a = config(data, steps=2)
    cfg_b = config(data, seed=43, deter=256, steps=2)
    cfg_a["rssm"]["recall_scale"] = cfg_b["rssm"]["recall_scale"] = recall_scale
    a = train_rssm(cfg_a, tmp_path / "A")
    b = train_rssm(cfg_b, tmp_path / "B")
    model_a, ca = load_rssm(a["best_checkpoint"])
    model_b, cb = load_rssm(b["best_checkpoint"])
    assert ca["config"]["seed"] != cb["config"]["seed"]
    assert not torch.equal(model_a.encoder.embeddings[0].weight, model_b.encoder.embeddings[0].weight)
    record = data[2][0]
    batch = model_batch([record], "cpu")
    assert set(batch) == {"agent_obs", "actions", "reward", "terminated", "truncated",
                          "observation_mask", "transition_mask"}
    for model, dimension in ((model_a, 384), (model_b, 512)):
        with torch.no_grad():
            sequence = model.observe(batch["agent_obs"], batch["actions"])
            state = model.initial(1)
            for t in range(record.horizon + 1):
                obs = {"image": batch["agent_obs"]["image"][:, t],
                       "direction": batch["agent_obs"]["direction"][:, t],
                       "task_id": [record.agent_obs[t].task_id]}
                state = model.observe_step(state, None if t == 0 else batch["actions"][:, t - 1], obs)
                torch.testing.assert_close(model.features(state), sequence.features[:, t])
        assert sequence.features.shape == (1, record.horizon + 1, dimension)
    if recall_scale:
        report = run_probes(a["best_checkpoint"], b["best_checkpoint"], *data[:2],
                            tmp_path / "probes", iterations=10)
        for result in report["models"].values():
            assert result["handoff_times"] == [12]
            assert "cue_retention_status" in result
            assert result["recall_head"]["val"]["full_history"]["n"] == 2
            for control in ("current_state", "recent_state"):
                assert result["recall_head"]["val"][control]["accuracy"] == 0.5


def test_validation_weighted_by_valid_targets_not_batch_count(data):
    model = RSSM().eval()
    records = data[2]
    assert min(record.horizon for record in records) < max(record.horizon for record in records)
    first = evaluate(model, records, 1)
    second = evaluate(model, records, 5)
    assert first == pytest.approx(second, abs=2e-6)


def test_validation_patience_and_best_last_are_distinct(data, tmp_path):
    cfg = config(data, steps=8)
    cfg["training"]["patience"] = 1
    # A controlled validation sequence tests the stopping decision independently of convergence.
    with patch("cmb_minigrid.training.rssm.evaluate", side_effect=[
        {"total": 5.0}, {"total": 5.0}, {"total": 4.0}, {"total": 4.1},
        {"total": 3.0}, {"total": 4.1}]):
        report = train_rssm(cfg, tmp_path / "stopping")
    assert report["early_stopped"] and report["steps"] == 4
    assert load_rssm(report["best_checkpoint"])[1]["step"] == 2
    assert load_rssm(report["last_checkpoint"])[1]["step"] == 4


def test_nonfinite_training_fails_with_a_record_and_preserves_checkpoint(data, tmp_path):
    original = RSSM.loss

    def loss(model, batch, *, sample=None):
        losses = original(model, batch, sample=sample)
        if sample is None:
            losses["total"] = torch.tensor(float("nan"))
        return losses

    with patch.object(RSSM, "loss", loss), pytest.raises(RuntimeError, match="non-finite"):
        train_rssm(config(data), tmp_path / "failure")
    path = next((tmp_path / "failure").iterdir())
    assert json.loads((path / "failure.json").read_text())["status"] == "training_failed"
    assert load_rssm(path / "last.pt")[1]["step"] == 0


def test_resume_rejects_changed_config_and_dataset(data, tmp_path):
    report = train_rssm(config(data, steps=2), tmp_path / "original")
    with pytest.raises(ValueError, match="configuration changed"):
        train_rssm(config(data, seed=999), tmp_path, resume=report["last_checkpoint"])
    cfg = config(data)
    cfg["input_sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed"):
        train_rssm(cfg, tmp_path, resume=report["last_checkpoint"])
    with pytest.raises(FileExistsError):
        train_rssm(config(data, steps=2), tmp_path / "original")


def test_frozen_probe_uses_prefix_only_and_current_recent_pairs_are_identical(data):
    model = RSSM().eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    records = [record for record in data[2] if record.episode_role == "expert"]
    features, labels, pairs = extract_pair_features(model, records)
    assert len(pairs) == 3 and labels.tolist() == [0, 1] * 3
    for name in ("current_state", "recent_state", "current_observation"):
        torch.testing.assert_close(features[name][::2], features[name][1::2], rtol=0, atol=0)
    assert not features["full_history"].requires_grad
    poisoned = [replace(record, agent_obs=(record.agent_obs[:record.handoff_t + 1]
                + (record.agent_obs[0],) * (record.horizon - record.handoff_t)),
                actions=(record.actions[:record.handoff_t]
                         + (0,) * (record.horizon - record.handoff_t))) for record in records]
    after, _, _ = extract_pair_features(model, poisoned)
    for name in features:
        torch.testing.assert_close(features[name], after[name], rtol=0, atol=0)
    probe, mean, scale = fit_linear_probe(features["current_state"], labels, iterations=20)
    assert probe_metrics(probe, mean, scale, features["current_state"], labels)["accuracy"] == 0.5
    probe, mean, scale = fit_linear_probe(features["early_observation"], labels, iterations=20)
    assert probe_metrics(probe, mean, scale, features["early_observation"], labels)["accuracy"] == 1
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    assert_nested_equal(before, model.state_dict())


def test_cli_and_empty_validation_are_explicit_failures(data, tmp_path):
    assert main(["--config", "/does/not/exist.yaml"]) == 1
    cfg = config(data)
    cfg["training"]["max_steps"] = 0
    with pytest.raises(ValueError, match="max_steps"):
        train_rssm(cfg, tmp_path)
    manifest = read_manifest(data[1])
    manifest.assignment.update({key: "train" for key in manifest.assignment})
    write_manifest(manifest, data[1])
    cfg = config(data)
    with pytest.raises(ValueError, match="nonempty"):
        train_rssm(cfg, tmp_path)
