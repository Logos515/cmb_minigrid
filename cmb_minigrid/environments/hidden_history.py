"""Strict counterfactual HiddenHistoryNav environment and pair utilities.

The map deliberately separates an early clue from the handoff. Two worlds use
the same public corridor and the same final eight-step prefix, but one blocks
the left branch and the other blocks the right branch. The differing map cells
are outside the agent's local view at handoff time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from cmb_minigrid.contracts import AgentObs, ContractError
from cmb_minigrid.environments.memory import AgentObsSpace

HiddenLabel = Literal["left_blocked", "right_blocked"]
Branch = Literal["left", "right"]

ENV_ID = "HiddenHistoryNav-v0"
TASK_ID = "hidden_history_nav"
MISSION = "reach the open branch indicated by the earlier channel clue"
MAP_FAMILY_ID = "hidden_history_nav_v1"
WIDTH = 19
HEIGHT = 15
HANDOFF_T = 12
RECENT_HISTORY_K = 8
SHARED_PREFIX: tuple[int, ...] = (2,) * HANDOFF_T
_BRANCH_FORWARD_STEPS = 5


def _hidden_label(value: str) -> HiddenLabel:
    if value not in ("left_blocked", "right_blocked"):
        raise ValueError(f"hidden_label must be left_blocked or right_blocked, got {value!r}")
    return value  # type: ignore[return-value]


def correct_branch(hidden_label: HiddenLabel) -> Branch:
    """Return the open branch for a privileged world label."""

    return "right" if hidden_label == "left_blocked" else "left"


def expert_actions(hidden_label: HiddenLabel) -> tuple[int, ...]:
    """Return a deterministic 3-action expert route to the open goal."""

    branch_turn = 0 if correct_branch(hidden_label) == "left" else 1
    return SHARED_PREFIX + (branch_turn,) + (2,) * _BRANCH_FORWARD_STEPS


@dataclass(frozen=True)
class EnvSnapshot:
    """JSON-safe simulator state needed to resume one exact episode."""

    seed: int
    hidden_label: HiddenLabel
    agent_pos: tuple[int, int]
    agent_dir: int
    step_count: int
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "hidden_label": self.hidden_label,
            "agent_pos": list(self.agent_pos),
            "agent_dir": self.agent_dir,
            "step_count": self.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnvSnapshot:
        return cls(
            seed=int(value["seed"]),
            hidden_label=_hidden_label(str(value["hidden_label"])),
            agent_pos=(int(value["agent_pos"][0]), int(value["agent_pos"][1])),
            agent_dir=int(value["agent_dir"]),
            step_count=int(value["step_count"]),
            terminated=bool(value["terminated"]),
            truncated=bool(value["truncated"]),
        )


class HiddenHistoryNavEnv(MiniGridEnv):
    """A MiniGrid map with an early clue and a hidden later branch choice."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        hidden_label: HiddenLabel = "left_blocked",
        render_mode: str | None = None,
    ):
        self.hidden_label = _hidden_label(hidden_label)
        self._episode_seed = 0
        self._episode_done = True
        mission_space = MissionSpace(mission_func=lambda: MISSION)
        super().__init__(
            mission_space=mission_space,
            width=WIDTH,
            height=HEIGHT,
            max_steps=64,
            see_through_walls=False,
            agent_view_size=7,
            render_mode=render_mode,
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = AgentObsSpace((7, 7, 3), task_id=TASK_ID)

    @property
    def correct_branch(self) -> Branch:
        return correct_branch(self.hidden_label)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Fill the interior with walls, then carve a public corridor and a
        # vertical branch. Hidden endpoints are five cells from the handoff,
        # outside the 7x7 local view.
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())
        for x in range(1, 14):
            self.grid.set(x, 7, None)
        for y in range(2, 13):
            self.grid.set(13, y, None)
        self.grid.set(2, 6, Key("red" if self.hidden_label == "left_blocked" else "blue"))

        if self.hidden_label == "left_blocked":
            self.grid.set(13, 2, Wall())
            self.grid.set(13, 12, Goal())
        else:
            self.grid.set(13, 12, Wall())
            self.grid.set(13, 2, Goal())

        self.agent_pos = (1, 7)
        self.agent_dir = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        if "hidden_label" in options:
            self.hidden_label = _hidden_label(str(options["hidden_label"]))
        self._episode_seed = 0 if seed is None else int(seed)
        super().reset(seed=seed, options=options)
        self._episode_done = False
        return self._to_agent_obs(self.gen_obs()), {}

    def step(self, action: int):
        if self._episode_done:
            raise RuntimeError("reset() must be called before stepping a completed episode")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, 2], got {action!r}")
        native_action = (self.actions.left, self.actions.right, self.actions.forward)[int(action)]
        raw_obs, reward, terminated, truncated, _ = super().step(native_action)
        self._episode_done = bool(terminated or truncated)
        return self._to_agent_obs(raw_obs), float(reward), bool(terminated), bool(truncated), {}

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            seed=self._episode_seed,
            hidden_label=self.hidden_label,
            agent_pos=tuple(int(v) for v in self.agent_pos),
            agent_dir=int(self.agent_dir),
            step_count=int(self.step_count),
            terminated=self._episode_done and not self.step_count >= self.max_steps,
            truncated=self._episode_done and self.step_count >= self.max_steps,
        )

    def restore(self, snapshot: EnvSnapshot | Mapping[str, Any]) -> AgentObs:
        if not isinstance(snapshot, EnvSnapshot):
            snapshot = EnvSnapshot.from_dict(snapshot)
        self.reset(seed=snapshot.seed, options={"hidden_label": snapshot.hidden_label})
        self.agent_pos = tuple(snapshot.agent_pos)
        self.agent_dir = snapshot.agent_dir
        self.step_count = snapshot.step_count
        self._episode_done = snapshot.terminated or snapshot.truncated
        return self._to_agent_obs(self.gen_obs())

    def render(self):
        return super().render()

    @staticmethod
    def _to_agent_obs(raw_obs: Mapping[str, Any]) -> AgentObs:
        return AgentObs(
            image=np.array(raw_obs["image"], dtype=np.uint8, copy=True),
            direction=int(raw_obs["direction"]),
            task_id=TASK_ID,
        )


@dataclass(frozen=True)
class HiddenHistoryEpisode:
    seed: int
    hidden_label: HiddenLabel
    map_family_id: str
    mission: str
    handoff_t: int
    observations: tuple[AgentObs, ...]
    actions: tuple[int, ...]
    expert_actions: tuple[int, ...]
    snapshots: tuple[EnvSnapshot, ...]
    reached_goal: bool

    @property
    def correct_branch(self) -> Branch:
        return correct_branch(self.hidden_label)


@dataclass(frozen=True)
class CounterfactualPair:
    seed: int
    left: HiddenHistoryEpisode
    right: HiddenHistoryEpisode

    @property
    def handoff_t(self) -> int:
        return self.left.handoff_t


@dataclass(frozen=True)
class PairValidationReport:
    passed: bool
    checks: Mapping[str, bool]

    def __bool__(self) -> bool:
        return self.passed


def _run_episode(seed: int, hidden_label: HiddenLabel) -> HiddenHistoryEpisode:
    env = HiddenHistoryNavEnv(hidden_label=hidden_label)
    observations: list[AgentObs] = []
    snapshots: list[EnvSnapshot] = []
    actions = expert_actions(hidden_label)
    terminated = truncated = False
    try:
        obs, _ = env.reset(seed=seed, options={"hidden_label": hidden_label})
        observations.append(obs)
        snapshots.append(env.snapshot())
        for action in actions:
            obs, _, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            snapshots.append(env.snapshot())
            if terminated or truncated:
                break
    finally:
        env.close()
    return HiddenHistoryEpisode(
        seed=seed,
        hidden_label=hidden_label,
        map_family_id=MAP_FAMILY_ID,
        mission=MISSION,
        handoff_t=HANDOFF_T,
        observations=tuple(observations),
        actions=tuple(actions[: len(observations) - 1]),
        expert_actions=actions,
        snapshots=tuple(snapshots),
        reached_goal=bool(terminated and not truncated),
    )


def generate_counterfactual_pair(seed: int) -> CounterfactualPair:
    """Generate two paired worlds with a common public handoff prefix."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    pair = CounterfactualPair(
        seed=seed,
        left=_run_episode(seed, "left_blocked"),
        right=_run_episode(seed, "right_blocked"),
    )
    report = validate_pair(pair, k=RECENT_HISTORY_K)
    if not report:
        raise ContractError(f"generated counterfactual pair failed validation: {report.checks}")
    return pair


def validate_pair(pair: CounterfactualPair, k: int = RECENT_HISTORY_K) -> PairValidationReport:
    """Check exact counterfactual equality and deployment leakage constraints."""

    if k <= 0:
        raise ValueError("k must be positive")
    a, b = pair.left, pair.right
    t = pair.handoff_t
    checks: dict[str, bool] = {
        "same_handoff_t": a.handoff_t == b.handoff_t == t,
        "same_step_count": len(a.observations) == len(b.observations),
        "same_task_id": all(obs.task_id == TASK_ID for obs in (*a.observations, *b.observations)),
        "same_mission": a.mission == b.mission == MISSION,
        "different_hidden_labels": a.hidden_label != b.hidden_label,
        "different_early_observation": any(
            x != y for x, y in zip(a.observations[:t], b.observations[:t], strict=True)
        ),
        "same_current_observation": a.observations[t] == b.observations[t],
        "same_recent_observations": a.observations[t - k : t + 1] == b.observations[t - k : t + 1],
        "same_recent_actions": a.actions[t - k : t] == b.actions[t - k : t],
        "same_pre_handoff_actions": a.actions[:t] == b.actions[:t],
        "different_correct_branch": a.correct_branch != b.correct_branch,
        "different_expert_handoff_action": a.expert_actions[t] != b.expert_actions[t],
        "both_goals_reachable": a.reached_goal and b.reached_goal,
        "no_privileged_agent_fields": all(
            set(getattr(type(obs), "__dataclass_fields__", {}))
            == {"image", "direction", "task_id"}
            for obs in (*a.observations, *b.observations)
        ),
        "no_hidden_label_in_public_text": all(
            label not in a.mission and label not in b.mission
            for label in (a.hidden_label, b.hidden_label)
        ),
    }
    return PairValidationReport(passed=all(checks.values()), checks=checks)


def save_snapshot(snapshot: EnvSnapshot, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def load_snapshot(path: str | Path) -> EnvSnapshot:
    return EnvSnapshot.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
