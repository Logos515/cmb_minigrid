"""Grouped split manifests and CPU padding batches for trajectory records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from cmb_minigrid.contracts import ContractError
from cmb_minigrid.data.trajectory import TrajectoryRecord, read_jsonl


@dataclass(frozen=True)
class SplitManifest:
    assignment: dict[str, str]
    map_family_by_episode: dict[str, str]
    pair_by_episode: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment": self.assignment,
            "map_family_by_episode": self.map_family_by_episode,
            "pair_by_episode": self.pair_by_episode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SplitManifest:
        required = {"assignment", "map_family_by_episode", "pair_by_episode"}
        missing = required.difference(value)
        if missing:
            raise ContractError(f"split manifest is missing fields: {sorted(missing)}")
        return cls(
            assignment={str(k): str(v) for k, v in value["assignment"].items()},
            map_family_by_episode={str(k): str(v) for k, v in value["map_family_by_episode"].items()},
            pair_by_episode={str(k): str(v) for k, v in value["pair_by_episode"].items()},
        )


def split_records(
    records: list[TrajectoryRecord],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> SplitManifest:
    """Split whole map families, keeping every pair in one split.

    With at least three families and non-zero ratios, each split receives at
    least one complete family. A one-family smoke dataset remains train-only.
    """

    if not records:
        raise ValueError("cannot split an empty record list")
    if len(ratios) != 3 or any(not math.isfinite(value) or value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must contain three non-negative values with positive sum")
    family_to_records: dict[str, list[TrajectoryRecord]] = {}
    pair_family: dict[str, str] = {}
    episode_ids: set[str] = set()
    for record in records:
        if record.episode_id in episode_ids:
            raise ContractError(f"duplicate episode_id: {record.episode_id}")
        episode_ids.add(record.episode_id)
        previous = pair_family.setdefault(record.pair_id, record.map_family_id)
        if previous != record.map_family_id:
            raise ContractError(f"pair {record.pair_id} spans multiple map families")
        family_to_records.setdefault(record.map_family_id, []).append(record)

    import numpy as np

    families = sorted(family_to_records)
    np.random.default_rng(seed).shuffle(families)
    total = len(families)
    quotas = [total * value / sum(ratios) for value in ratios]
    counts = [math.floor(value) for value in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    active = [i for i, value in enumerate(ratios) if value > 0]
    if total >= len(active):
        for index in active:
            if counts[index] == 0:
                donor = max((i for i in active if counts[i] > 1), key=lambda i: counts[i] - quotas[i])
                counts[donor] -= 1
                counts[index] += 1
    target_train, target_val, _ = counts
    family_split: dict[str, str] = {}
    for index, family in enumerate(families):
        family_split[family] = "train" if index < target_train else "val" if index < target_train + target_val else "test"
    assignment = {record.episode_id: family_split[record.map_family_id] for record in records}
    manifest = SplitManifest(
        assignment=assignment,
        map_family_by_episode={record.episode_id: record.map_family_id for record in records},
        pair_by_episode={record.episode_id: record.pair_id for record in records},
    )
    leakage = validate_split_manifest(manifest, records=records)
    if leakage:
        raise ContractError("split leakage detected: " + "; ".join(leakage))
    return manifest


def validate_split_manifest(
    manifest: SplitManifest,
    *,
    records: list[TrajectoryRecord] | None = None,
) -> list[str]:
    """Return human-readable family/pair leakage errors, or an empty list."""

    errors: list[str] = []
    allowed_splits = {"train", "val", "test"}
    for episode_id, split in manifest.assignment.items():
        if split not in allowed_splits:
            errors.append(f"episode {episode_id} has invalid split {split!r}")
    assignment_ids = set(manifest.assignment)
    if set(manifest.map_family_by_episode) != assignment_ids:
        errors.append("map_family_by_episode keys do not match assignment keys")
    if set(manifest.pair_by_episode) != assignment_ids:
        errors.append("pair_by_episode keys do not match assignment keys")
    if records is not None:
        record_ids = {record.episode_id for record in records}
        if record_ids != assignment_ids:
            errors.append("manifest does not cover exactly the supplied episode IDs")
        for record in records:
            if manifest.map_family_by_episode.get(record.episode_id) != record.map_family_id:
                errors.append(f"episode {record.episode_id} has mismatched map family")
            if manifest.pair_by_episode.get(record.episode_id) != record.pair_id:
                errors.append(f"episode {record.episode_id} has mismatched pair ID")
    for field_name, values in (
        ("map_family", manifest.map_family_by_episode),
        ("pair", manifest.pair_by_episode),
    ):
        groups: dict[str, set[str]] = {}
        for episode_id, group in values.items():
            split = manifest.assignment.get(episode_id)
            if split is not None:
                groups.setdefault(group, set()).add(split)
        for group, splits in groups.items():
            if len(splits) > 1:
                errors.append(f"{field_name} {group} appears in {sorted(splits)}")
    return errors


def write_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class TrajectoryDataset(Dataset[TrajectoryRecord]):
    """Map a JSONL dataset and optional split manifest to episode records."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str | None = None,
        manifest_path: str | Path | None = None,
    ):
        self.records = read_jsonl(path)
        if split is not None:
            if manifest_path is None:
                raise ValueError("manifest_path is required when selecting a split")
            manifest = read_manifest(manifest_path)
            errors = validate_split_manifest(manifest, records=self.records)
            if errors:
                raise ContractError("invalid split manifest: " + "; ".join(errors))
            self.records = [
                record for record in self.records if manifest.assignment.get(record.episode_id) == split
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrajectoryRecord:
        return self.records[index]


def collate_trajectories(records: list[TrajectoryRecord]) -> dict[str, Any]:
    """Pad variable-length trajectories while returning deployment-safe tensors."""

    if not records:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(records)
    max_obs = max(len(record.agent_obs) for record in records)
    max_transitions = max(record.horizon for record in records)
    image_shape = records[0].agent_obs[0].image.shape
    images = torch.zeros((batch_size, max_obs, *image_shape), dtype=torch.uint8)
    directions = torch.zeros((batch_size, max_obs), dtype=torch.long)
    actions = torch.full((batch_size, max_transitions), -1, dtype=torch.long)
    rewards = torch.zeros((batch_size, max_transitions), dtype=torch.float32)
    terminated = torch.zeros((batch_size, max_transitions), dtype=torch.bool)
    truncated = torch.zeros((batch_size, max_transitions), dtype=torch.bool)
    expert_actions = torch.full((batch_size, max_transitions), -1, dtype=torch.long)
    observation_mask = torch.zeros((batch_size, max_obs), dtype=torch.bool)
    transition_mask = torch.zeros((batch_size, max_transitions), dtype=torch.bool)
    task_ids: list[list[str]] = []
    metadata: list[dict[str, Any]] = []

    for row, record in enumerate(records):
        if any(obs.image.shape != image_shape for obs in record.agent_obs):
            raise ContractError("all observations in a batch must have the same image shape")
        obs_count = len(record.agent_obs)
        transition_count = record.horizon
        for index, obs in enumerate(record.agent_obs):
            images[row, index] = torch.from_numpy(obs.image.copy())
            directions[row, index] = obs.direction
        observation_mask[row, :obs_count] = True
        if transition_count:
            actions[row, :transition_count] = torch.tensor(record.actions, dtype=torch.long)
            rewards[row, :transition_count] = torch.tensor(record.rewards, dtype=torch.float32)
            terminated[row, :transition_count] = torch.tensor(record.terminated, dtype=torch.bool)
            truncated[row, :transition_count] = torch.tensor(record.truncated, dtype=torch.bool)
            expert_actions[row, :transition_count] = torch.tensor(record.expert_actions, dtype=torch.long)
            transition_mask[row, :transition_count] = True
        task_ids.append([obs.task_id for obs in record.agent_obs])
        metadata.append(
            {
                "episode_id": record.episode_id,
                "map_family_id": record.map_family_id,
                "pair_id": record.pair_id,
                "handoff_t": record.handoff_t,
                "episode_role": record.episode_role,
            }
        )

    return {
        "agent_obs": {"image": images, "direction": directions, "task_id": task_ids},
        "actions": actions,
        "reward": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "expert_actions": expert_actions,
        "observation_mask": observation_mask,
        "transition_mask": transition_mask,
        "padding_mask": ~transition_mask,
        "metadata": metadata,
    }
