from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

from cmb_minigrid.contracts import AgentObs
from cmb_minigrid.data import (
    TrajectoryDataset,
    TrajectoryRecord,
    collate_trajectories,
    collect_records,
    read_jsonl,
    replay_handoff,
    split_records,
    validate_counterfactual_records,
    validate_expert_actions,
    validate_split_manifest,
    write_jsonl,
    write_manifest,
)
from cmb_minigrid.environments.hidden_history import MAP_FAMILY_IDS


@pytest.fixture(scope="module")
def family_records():
    return collect_records(pair_count=2, seed=13, map_family_ids=MAP_FAMILY_IDS)


@pytest.fixture()
def records():
    return collect_records(pair_count=2, seed=7, exploration_rate=0.0)


def test_collection_has_expert_and_failed_exploration(records):
    assert len(records) == 8
    assert {record.episode_role for record in records} == {"expert", "exploration"}
    assert any(record.truncated[-1] for record in records)
    assert all(len(record.agent_obs) == len(record.actions) + 1 for record in records)
    assert all(
        record.expert_actions[12] != record.actions[12]
        for record in records
        if record.episode_role == "exploration" and record.truncated[-1]
    )


def test_jsonl_replay_and_snapshot(records, tmp_path):
    path = write_jsonl(records, tmp_path / "data.jsonl")
    loaded = read_jsonl(path)
    assert len(loaded) == len(records)
    assert all(replay_handoff(record) for record in loaded)
    assert all(validate_expert_actions(record) for record in loaded)


def test_replay_rejects_public_observation_and_privileged_state_tampering(records):
    record = records[0]
    changed_image = record.agent_obs[1].image.copy()
    changed_image[0, 0, 0] ^= 1
    changed_obs = AgentObs(
        image=changed_image,
        direction=record.agent_obs[1].direction,
        task_id=record.agent_obs[1].task_id,
    )
    public_tamper = replace(
        record,
        agent_obs=record.agent_obs[:1] + (changed_obs,) + record.agent_obs[2:],
    )
    assert not replay_handoff(public_tamper)

    altered_state = dict(record.privileged_state[1])
    altered_state["agent_pos"] = [3, 7]
    privileged_tamper = replace(
        record,
        privileged_state=record.privileged_state[:1] + (altered_state,) + record.privileged_state[2:],
    )
    assert not replay_handoff(privileged_tamper)


def test_expert_label_tampering_is_rejected(records):
    record = records[0]
    labels = list(record.expert_actions)
    labels[0] = (labels[0] + 1) % 3
    tampered = replace(record, expert_actions=tuple(labels))
    assert not validate_expert_actions(tampered)


def test_cpu_batch_masks_track_variable_lengths(records):
    batch = collate_trajectories([records[0], records[3]])
    assert batch["agent_obs"]["image"].shape[1] == max(len(records[0].agent_obs), len(records[3].agent_obs))
    assert batch["actions"].shape[1] == max(records[0].horizon, records[3].horizon)
    assert batch["observation_mask"].sum(dim=1).tolist() == [
        len(records[0].agent_obs),
        len(records[3].agent_obs),
    ]
    assert batch["transition_mask"].sum(dim=1).tolist() == [
        records[0].horizon,
        records[3].horizon,
    ]
    assert torch.all(batch["actions"][~batch["transition_mask"]] == -1)


def test_collate_padding_does_not_expose_privileged_state(records):
    batch = collate_trajectories([records[0], records[-1]])
    assert batch["agent_obs"]["image"].shape[0] == 2
    assert batch["agent_obs"]["image"].dtype == __import__("torch").uint8
    assert batch["observation_mask"].shape[1] == max(len(r.agent_obs) for r in [records[0], records[-1]])
    assert "privileged_state" not in batch
    assert "env_snapshot" not in batch
    assert batch["transition_mask"].sum().item() == records[0].horizon + records[-1].horizon


def test_split_manifest_rejects_cross_split_pair(records):
    manifest = split_records(records)
    tampered = type(manifest)(
        assignment={**manifest.assignment, records[0].episode_id: "val"},
        map_family_by_episode=manifest.map_family_by_episode,
        pair_by_episode=manifest.pair_by_episode,
    )
    errors = validate_split_manifest(tampered, records=records)
    assert any("pair" in error for error in errors)


def test_manifest_rejects_unknown_ids(records):
    manifest = split_records(records)
    tampered = type(manifest)(
        assignment={**manifest.assignment, "intruder": "train"},
        map_family_by_episode={**manifest.map_family_by_episode, "intruder": "fake"},
        pair_by_episode={**manifest.pair_by_episode, "intruder": "fake"},
    )
    assert validate_split_manifest(tampered, records=records)


def test_five_family_collection_keeps_pairs_and_splits_grouped(family_records):
    collected = family_records
    assert len(collected) == 40
    assert {record.map_family_id for record in collected} == set(MAP_FAMILY_IDS)
    manifest = split_records(collected, seed=13)
    assert not validate_split_manifest(manifest, records=collected)
    assert {split for split in manifest.assignment.values()} == {"train", "val", "test"}
    pair_splits = {
        pair_id: {manifest.assignment[record.episode_id] for record in collected if record.pair_id == pair_id}
        for pair_id in {record.pair_id for record in collected}
    }
    assert all(len(splits) == 1 for splits in pair_splits.values())
    assert manifest == split_records(list(reversed(collected)), seed=13)


def test_all_families_persist_replay_and_validate_pairs(family_records, tmp_path):
    loaded = read_jsonl(write_jsonl(family_records, tmp_path / "multi.jsonl"))
    assert not validate_counterfactual_records(loaded)
    assert all(replay_handoff(record) for record in loaded)
    assert all(validate_expert_actions(record) for record in loaded)
    for family in MAP_FAMILY_IDS:
        records = [record for record in loaded if record.map_family_id == family]
        assert len({record.pair_id for record in records}) == 2
        assert sum(record.episode_role == "expert" for record in records) == 4
        assert sum(record.truncated[-1] for record in records) >= 2


@pytest.mark.parametrize("split", ("train", "val", "test"))
def test_each_split_loads_cpu_batch(family_records, tmp_path, split):
    data = write_jsonl(family_records, tmp_path / "multi.jsonl")
    manifest = split_records(family_records, seed=13)
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    dataset = TrajectoryDataset(data, split=split, manifest_path=manifest_path)
    assert len(dataset) > 0
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_trajectories)))
    assert batch["agent_obs"]["image"].device.type == "cpu"
    assert set(batch["agent_obs"]) == {"image", "direction", "task_id"}
    assert batch["observation_mask"].sum(1).tolist() == [len(r.agent_obs) for r in dataset.records[:4]]
    assert batch["transition_mask"].sum(1).tolist() == [r.horizon for r in dataset.records[:4]]
    assert torch.equal(batch["padding_mask"], ~batch["transition_mask"])
    assert "privileged_state" not in batch and "env_snapshot" not in batch


def test_legacy_record_without_snapshot_family_still_replays(records):
    raw = records[0].to_dict()
    raw["env_snapshot"] = dict(raw["env_snapshot"])
    raw["env_snapshot"].pop("map_family_id")
    raw["privileged_state"] = [
        {key: value for key, value in item.items() if key != "map_family_id"}
        for item in raw["privileged_state"]
    ]
    loaded = TrajectoryRecord.from_dict(raw)
    assert replay_handoff(loaded) and validate_expert_actions(loaded)


def test_pair_validator_rejects_missing_and_misgrouped_endpoints(family_records):
    assert validate_counterfactual_records(family_records[1:])
    first, second = family_records[0], family_records[1]
    moved = replace(second, pair_id="other-pair")
    assert validate_counterfactual_records([first, moved, *family_records[2:]])


@pytest.mark.parametrize("ratios, expected", [((1, 0, 0), "train"), ((0, 1, 0), "val"), ((0, 0, 1), "test")])
def test_zero_ratio_split_remains_empty(family_records, ratios, expected):
    manifest = split_records(family_records, ratios=ratios)
    assert set(manifest.assignment.values()) == {expected}


def test_collection_rejects_duplicate_families():
    with pytest.raises(ValueError, match="unique"):
        collect_records(pair_count=1, seed=42, map_family_ids=[MAP_FAMILY_IDS[0]] * 2)


def test_seeded_multifamily_collection_is_reproducible():
    first = collect_records(pair_count=1, seed=42, map_family_ids=MAP_FAMILY_IDS)
    second = collect_records(pair_count=1, seed=42, map_family_ids=MAP_FAMILY_IDS)
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert split_records(first, seed=42) == split_records(second, seed=42)
