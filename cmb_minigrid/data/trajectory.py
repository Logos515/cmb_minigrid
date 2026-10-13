"""Trajectory records, collection, persistence, and replay checks for Step 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cmb_minigrid.contracts import AgentObs, ContractError
from cmb_minigrid.environments.hidden_history import (
    HANDOFF_T,
    MAP_FAMILY_ID,
    MISSION,
    CounterfactualPair,
    EnvSnapshot,
    HiddenHistoryEpisode,
    HiddenHistoryNavEnv,
    HiddenLabel,
    _plan_actions,
    generate_counterfactual_pair,
    get_map_layout,
    validate_pair,
)

SCHEMA_VERSION = 1
PUBLIC_ACTIONS = (0, 1, 2)


def _obs_to_dict(obs: AgentObs) -> dict[str, Any]:
    return {
        "image": obs.image.tolist(),
        "direction": obs.direction,
        "task_id": obs.task_id,
    }


def _obs_from_dict(value: dict[str, Any]) -> AgentObs:
    return AgentObs(
        image=np.asarray(value["image"], dtype=np.uint8),
        direction=int(value["direction"]),
        task_id=str(value["task_id"]),
    )


@dataclass(frozen=True)
class TrajectoryRecord:
    """One episode with deployment fields separated from privileged metadata."""

    episode_id: str
    seed: int
    agent_obs: tuple[AgentObs, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    expert_actions: tuple[int, ...]
    map_family_id: str
    pair_id: str
    handoff_t: int
    env_snapshot: dict[str, Any] | None
    privileged_state: tuple[dict[str, Any], ...]
    mission: str = MISSION
    episode_role: str = "expert"

    def __post_init__(self) -> None:
        transition_count = len(self.actions)
        if len(self.agent_obs) != transition_count + 1:
            raise ContractError("trajectory requires obs[0:T+1] and action[0:T]")
        for name, values in (
            ("rewards", self.rewards),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
            ("expert_actions", self.expert_actions),
        ):
            if len(values) != transition_count:
                raise ContractError(f"{name} must have length T={transition_count}")
        if len(self.privileged_state) != len(self.agent_obs):
            raise ContractError("privileged_state must align with every agent observation")
        if not 0 <= self.handoff_t <= transition_count:
            raise ContractError("handoff_t must be between 0 and the trajectory horizon")
        if not self.episode_id or not self.map_family_id or not self.pair_id:
            raise ContractError("episode_id, map_family_id, and pair_id must be non-empty")
        if self.seed < 0:
            raise ContractError("seed must be non-negative")
        if self.mission != MISSION:
            raise ContractError("mission does not match the fixed environment task")
        if not all(isinstance(obs, AgentObs) for obs in self.agent_obs):
            raise ContractError("agent_obs must contain AgentObs values")
        if not all(isinstance(action, (int, np.integer)) and int(action) in PUBLIC_ACTIONS for action in self.actions):
            raise ContractError("actions must contain only public actions 0, 1, or 2")
        if not all(isinstance(action, (int, np.integer)) and int(action) in PUBLIC_ACTIONS for action in self.expert_actions):
            raise ContractError("expert_actions must contain only public actions 0, 1, or 2")
        if not all(np.isfinite(float(reward)) for reward in self.rewards):
            raise ContractError("rewards must be finite")
        for index, (is_terminal, is_truncated) in enumerate(zip(self.terminated, self.truncated, strict=True)):
            if bool(is_terminal) and bool(is_truncated):
                raise ContractError(f"transition {index} cannot be both terminated and truncated")
            if (bool(is_terminal) or bool(is_truncated)) and index != transition_count - 1:
                raise ContractError("terminated/truncated may only occur on the final transition")
        if self.episode_role not in {"expert", "exploration"}:
            raise ContractError("episode_role must be expert or exploration")
        if self.env_snapshot is not None:
            snapshot = EnvSnapshot.from_dict(self.env_snapshot)
            if snapshot.hidden_label not in ("left_blocked", "right_blocked"):
                raise ContractError("env_snapshot has an invalid hidden label")
            if snapshot.step_count != self.handoff_t:
                raise ContractError("env_snapshot must be the snapshot at handoff_t")
            if snapshot.step_count >= len(self.privileged_state):
                raise ContractError("env_snapshot step is outside privileged_state")
            if snapshot.map_family_id != self.map_family_id:
                raise ContractError("env_snapshot map family does not match trajectory")
        previous: EnvSnapshot | None = None
        for index, raw_snapshot in enumerate(self.privileged_state):
            current = EnvSnapshot.from_dict(raw_snapshot)
            if current.seed != self.seed:
                raise ContractError("privileged_state seed does not match trajectory seed")
            if current.map_family_id != self.map_family_id:
                raise ContractError("privileged_state map family does not match trajectory")
            if current.step_count != index:
                raise ContractError("privileged_state step_count must align with observation index")
            if previous is not None and current.hidden_label != previous.hidden_label:
                raise ContractError("hidden label cannot change within one trajectory")
            previous = current

    @property
    def horizon(self) -> int:
        return len(self.actions)

    @property
    def handoff_reached(self) -> bool:
        return self.env_snapshot is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "agent_obs": [_obs_to_dict(obs) for obs in self.agent_obs],
            "actions": list(self.actions),
            "reward": list(self.rewards),
            "terminated": list(self.terminated),
            "truncated": list(self.truncated),
            "expert_actions": list(self.expert_actions),
            "map_family_id": self.map_family_id,
            "pair_id": self.pair_id,
            "handoff_t": self.handoff_t,
            "env_snapshot": self.env_snapshot,
            "privileged_state": list(self.privileged_state),
            "mission": self.mission,
            "episode_role": self.episode_role,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrajectoryRecord:
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ContractError(f"unsupported trajectory schema: {value.get('schema_version')}")
        return cls(
            episode_id=str(value["episode_id"]),
            seed=int(value["seed"]),
            agent_obs=tuple(_obs_from_dict(item) for item in value["agent_obs"]),
            actions=tuple(int(item) for item in value["actions"]),
            rewards=tuple(float(item) for item in value["reward"]),
            terminated=tuple(bool(item) for item in value["terminated"]),
            truncated=tuple(bool(item) for item in value["truncated"]),
            expert_actions=tuple(int(item) for item in value["expert_actions"]),
            map_family_id=str(value["map_family_id"]),
            pair_id=str(value["pair_id"]),
            handoff_t=int(value["handoff_t"]),
            env_snapshot=(
                EnvSnapshot.from_dict(value["env_snapshot"]).to_dict()
                if value.get("env_snapshot") is not None else None
            ),
            privileged_state=tuple(
                EnvSnapshot.from_dict(item).to_dict() for item in value["privileged_state"]
            ),
            mission=str(value.get("mission", MISSION)),
            episode_role=str(value.get("episode_role", "expert")),
        )


def record_from_episode(
    episode: HiddenHistoryEpisode,
    *,
    episode_id: str,
    pair_id: str,
    role: str = "expert",
) -> TrajectoryRecord:
    """Convert the privileged expert rollout into a persistable record."""

    handoff_snapshot = (
        episode.snapshots[episode.handoff_t].to_dict()
        if len(episode.snapshots) > episode.handoff_t
        else None
    )
    return TrajectoryRecord(
        episode_id=episode_id,
        seed=episode.seed,
        agent_obs=episode.observations,
        actions=episode.actions,
        rewards=episode.rewards,
        terminated=episode.terminated,
        truncated=episode.truncated,
        expert_actions=episode.expert_actions[: len(episode.actions)],
        map_family_id=episode.map_family_id,
        pair_id=pair_id,
        handoff_t=episode.handoff_t,
        env_snapshot=handoff_snapshot,
        privileged_state=tuple(snapshot.to_dict() for snapshot in episode.snapshots),
        mission=episode.mission,
        episode_role=role,
    )


def _shortest_action(env: HiddenHistoryNavEnv, hidden_label: HiddenLabel) -> int:
    """Plan one action from the actual simulator pose using privileged map access."""

    actions = _plan_actions(env, env.goal_for_hidden_label(hidden_label))
    return actions[0] if actions else 2


def _collect_exploration_episode(
    seed: int,
    hidden_label: HiddenLabel,
    *,
    pair_id: str,
    exploration_rate: float,
    max_steps: int,
    map_family_id: str = MAP_FAMILY_ID,
    force_failure: bool = False,
) -> TrajectoryRecord:
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be between 0 and 1")
    env = HiddenHistoryNavEnv(hidden_label=hidden_label, map_family_id=map_family_id)
    rng = np.random.default_rng(seed + 10_000)
    observations: list[AgentObs] = []
    snapshots: list[EnvSnapshot] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    expert_labels: list[int] = []
    try:
        obs, _ = env.reset(seed=seed, options={"hidden_label": hidden_label, "map_family_id": map_family_id})
        observations.append(obs)
        snapshots.append(env.snapshot())
        # Keep a fixed prefix for the handoff, then inject exploration. The
        # expert label is recomputed from the actual state before every action.
        for step in range(max_steps):
            expert_action = _shortest_action(env, hidden_label)
            expert_labels.append(expert_action)
            if step < HANDOFF_T:
                # Preserve the exact public prefix used by the paired expert
                # rollouts so every exploratory record has a valid handoff.
                action = get_map_layout(map_family_id).prefix_actions[step]
            elif force_failure and step == HANDOFF_T:
                # Turn toward the blocked branch and then keep moving until the
                # real environment time limit. This creates a reproducible
                # failure without fabricating terminal flags.
                action = 0 if hidden_label == "left_blocked" else 1
            elif force_failure and step > HANDOFF_T:
                action = 2
            else:
                explore = step >= HANDOFF_T and rng.random() < exploration_rate
                action = int(rng.integers(0, 3)) if explore else expert_action
            obs, reward, done, cut, _ = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            terminated.append(bool(done))
            truncated.append(bool(cut))
            observations.append(obs)
            snapshots.append(env.snapshot())
            if done or cut:
                break
    finally:
        env.close()
    handoff_snapshot = snapshots[HANDOFF_T].to_dict() if len(snapshots) > HANDOFF_T else None
    return TrajectoryRecord(
        episode_id=f"{pair_id}-{hidden_label}-explore",
        seed=seed,
        agent_obs=tuple(observations),
        actions=tuple(actions),
        rewards=tuple(rewards),
        terminated=tuple(terminated),
        truncated=tuple(truncated),
        expert_actions=tuple(expert_labels),
        map_family_id=map_family_id,
        pair_id=pair_id,
        handoff_t=HANDOFF_T,
        env_snapshot=handoff_snapshot,
        privileged_state=tuple(snapshot.to_dict() for snapshot in snapshots),
        mission=MISSION,
        episode_role="exploration",
    )


def collect_records(
    *,
    pair_count: int,
    seed: int,
    exploration_rate: float = 0.15,
    include_exploration: bool = True,
    map_family_ids: tuple[str, ...] | list[str] | None = None,
) -> list[TrajectoryRecord]:
    """Collect paired expert episodes and optional exploratory episodes."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(exploration_rate) or not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be between 0 and 1")
    families = tuple(map_family_ids) if map_family_ids is not None else (MAP_FAMILY_ID,)
    if not families:
        raise ValueError("map_family_ids must not be empty")
    if len(set(families)) != len(families):
        raise ValueError("map_family_ids must be unique")
    for family in families:
        get_map_layout(family)
    records: list[TrajectoryRecord] = []
    for family_index, map_family_id in enumerate(families):
        for index in range(pair_count):
            pair_seed = seed + family_index * pair_count + index
            pair: CounterfactualPair = generate_counterfactual_pair(pair_seed, map_family_id)
            pair_id = f"pair-{map_family_id}-{pair_seed:04d}"
            records.extend(
                (
                    record_from_episode(
                        pair.left,
                        episode_id=f"{pair_id}-left-expert",
                        pair_id=pair_id,
                    ),
                    record_from_episode(
                        pair.right,
                        episode_id=f"{pair_id}-right-expert",
                        pair_id=pair_id,
                    ),
                )
            )
            if include_exploration:
                records.extend(
                    (
                        _collect_exploration_episode(
                            pair_seed,
                            "left_blocked",
                            pair_id=pair_id,
                            exploration_rate=exploration_rate,
                            max_steps=64,
                            map_family_id=map_family_id,
                        ),
                        _collect_exploration_episode(
                            pair_seed,
                            "right_blocked",
                            pair_id=pair_id,
                            exploration_rate=exploration_rate,
                            max_steps=64,
                            force_failure=True,
                            map_family_id=map_family_id,
                        ),
                    )
                )
    return records


def write_jsonl(records: list[TrajectoryRecord], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    record = TrajectoryRecord.from_dict(json.loads(line))
                    if record.episode_id in seen_ids:
                        raise ContractError(f"duplicate episode_id {record.episode_id!r}")
                    seen_ids.add(record.episode_id)
                    records.append(record)
                except (KeyError, TypeError, ValueError, ContractError) as exc:
                    raise ContractError(f"invalid trajectory at line {line_number}: {exc}") from exc
    return records


def validate_counterfactual_records(records: list[TrajectoryRecord]) -> list[str]:
    """Validate the actual saved expert endpoints, including their grouping."""

    grouped: dict[str, list[TrajectoryRecord]] = {}
    for record in records:
        grouped.setdefault(record.pair_id, [])
        if record.episode_role == "expert":
            grouped[record.pair_id].append(record)
    errors: list[str] = []
    for pair_id, endpoints in grouped.items():
        if len(endpoints) != 2:
            errors.append(f"pair {pair_id} requires two expert endpoints, got {len(endpoints)}")
            continue
        episodes = []
        for record in endpoints:
            snapshots = tuple(EnvSnapshot.from_dict(item) for item in record.privileged_state)
            episodes.append(HiddenHistoryEpisode(
                seed=record.seed,
                hidden_label=snapshots[0].hidden_label,
                map_family_id=record.map_family_id,
                mission=record.mission,
                handoff_t=record.handoff_t,
                observations=record.agent_obs,
                actions=record.actions,
                expert_actions=record.expert_actions,
                snapshots=snapshots,
                rewards=record.rewards,
                terminated=record.terminated,
                truncated=record.truncated,
                reached_goal=bool(record.terminated[-1] and not record.truncated[-1]),
            ))
        pair = CounterfactualPair(seed=episodes[0].seed, left=episodes[0], right=episodes[1])
        if pair.handoff_t < 8 or any(episode.handoff_t >= len(episode.actions) for episode in episodes):
            errors.append(f"pair {pair_id} has insufficient handoff history or suffix")
            continue
        report = validate_pair(pair)
        if not report:
            failed = [name for name, passed in report.checks.items() if not passed]
            errors.append(f"pair {pair_id} fails {failed}")
    return errors


def replay_handoff(record: TrajectoryRecord) -> bool:
    """Replay from reset and verify every public and privileged time step.

    This is intentionally stricter than checking only the terminal reward: a
    corrupted action, observation, or simulator snapshot must invalidate the
    record before it can be used for offline supervision.
    """

    if record.env_snapshot is None:
        return False
    snapshot = EnvSnapshot.from_dict(record.env_snapshot)
    env = HiddenHistoryNavEnv(hidden_label=snapshot.hidden_label, map_family_id=snapshot.map_family_id)
    try:
        obs, _ = env.reset(seed=record.seed, options={"hidden_label": snapshot.hidden_label, "map_family_id": snapshot.map_family_id})
        if obs != record.agent_obs[0]:
            return False
        if env.snapshot().to_dict() != record.privileged_state[0]:
            return False
        completed = 0
        for index, action in enumerate(record.actions):
            obs, reward, done, cut, _ = env.step(action)
            if obs != record.agent_obs[index + 1] or float(reward) != record.rewards[index]:
                return False
            if bool(done) != record.terminated[index] or bool(cut) != record.truncated[index]:
                return False
            if env.snapshot().to_dict() != record.privileged_state[index + 1]:
                return False
            if done or cut:
                completed = index + 1
                break
            completed = index + 1
        if completed != len(record.actions):
            return False
    finally:
        env.close()
    return replay_from_handoff(record)


def replay_from_handoff(record: TrajectoryRecord) -> bool:
    """Restore a fresh environment at ``handoff_t`` and replay the suffix.

    The fresh environment is rebuilt from the serialized handoff snapshot,
    rather than continuing the reset-and-replay environment used by
    :func:`replay_handoff`. This catches snapshots that happen to match during
    the first replay but cannot actually reconstruct the continuation.
    """

    if record.env_snapshot is None:
        return False
    snapshot = EnvSnapshot.from_dict(record.env_snapshot)
    handoff = record.handoff_t
    if handoff >= len(record.agent_obs) or handoff > record.horizon:
        return False
    env = HiddenHistoryNavEnv(hidden_label=snapshot.hidden_label, map_family_id=snapshot.map_family_id)
    try:
        restored = env.restore(snapshot)
        if restored != record.agent_obs[handoff]:
            return False
        if env.snapshot().to_dict() != record.privileged_state[handoff]:
            return False
        for index in range(handoff, record.horizon):
            obs, reward, done, cut, _ = env.step(record.actions[index])
            if obs != record.agent_obs[index + 1]:
                return False
            if float(reward) != record.rewards[index]:
                return False
            if bool(done) != record.terminated[index] or bool(cut) != record.truncated[index]:
                return False
            if env.snapshot().to_dict() != record.privileged_state[index + 1]:
                return False
            if done or cut:
                return index + 1 == record.horizon
        return True
    finally:
        env.close()


def validate_expert_actions(record: TrajectoryRecord) -> bool:
    """Verify every stored expert label against the privileged simulator state."""

    if len(record.expert_actions) != record.horizon:
        return False
    if not record.privileged_state:
        return False
    first = EnvSnapshot.from_dict(record.privileged_state[0])
    env = HiddenHistoryNavEnv(hidden_label=first.hidden_label, map_family_id=first.map_family_id)
    try:
        env.reset(seed=record.seed, options={"hidden_label": first.hidden_label, "map_family_id": first.map_family_id})
        for index, expected in enumerate(record.expert_actions):
            current = env.snapshot().to_dict()
            if current != record.privileged_state[index]:
                return False
            planned = _shortest_action(env, first.hidden_label)
            if int(expected) != planned:
                return False
            env.step(record.actions[index])
        return env.snapshot().to_dict() == record.privileged_state[-1]
    finally:
        env.close()
