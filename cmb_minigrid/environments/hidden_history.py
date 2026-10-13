"""Strict counterfactual HiddenHistoryNav environment and pair utilities.

The map deliberately separates an early clue from the handoff. Two worlds use
the same public corridor and the same final eight-step prefix, but one blocks
the left branch and the other blocks the right branch. The differing map cells
are outside the agent's local view at handoff time.
"""

from __future__ import annotations

import json
from collections import deque
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
MAP_FAMILY_IDS = tuple(f"hidden_history_nav_v{i}" for i in range(1, 6))
WIDTH = 19
HEIGHT = 15
HANDOFF_T = 12
RECENT_HISTORY_K = 8
SHARED_PREFIX: tuple[int, ...] = (2,) * HANDOFF_T
_BRANCH_FORWARD_STEPS = 5


@dataclass(frozen=True)
class MapLayout:
    """Public topology shared by the two hidden-label counterfactual worlds."""

    map_family_id: str
    prefix_actions: tuple[int, ...]
    branch_segments: tuple[int, ...] = (_BRANCH_FORWARD_STEPS,)
    start_pos: tuple[int, int] = (1, 7)
    start_dir: int = 0
    clue_pos: tuple[int, int] = (2, 6)
    width: int = WIDTH
    height: int = HEIGHT

    def branch_actions(self, branch: Branch) -> tuple[int, ...]:
        # Both arms have equal lengths; subsequent segments bend away from
        # the public corridor so reaching either endpoint requires the hub.
        first_turn = 0 if branch == "left" else 1
        turns = (first_turn, 1 - first_turn, first_turn)
        return tuple(
            action
            for turn, length in zip(turns, self.branch_segments)
            for action in (turn,) + (2,) * length
        )


MAP_LAYOUTS: dict[str, MapLayout] = {
    # Straight corridor, then a vertical fork (the original Step 2 map).
    "hidden_history_nav_v1": MapLayout("hidden_history_nav_v1", (2,) * 12),
    # A short downward detour before the final horizontal approach.
    "hidden_history_nav_v2": MapLayout("hidden_history_nav_v2", (2, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2)),
    # A vertical public corridor, ending in a horizontal fork.
    "hidden_history_nav_v3": MapLayout(
        "hidden_history_nav_v3", SHARED_PREFIX,
        start_pos=(9, 1), start_dir=1, clue_pos=(10, 2), height=19,
    ),
    # Each arm bends east after five cells.
    "hidden_history_nav_v4": MapLayout("hidden_history_nav_v4", SHARED_PREFIX, (5, 2)),
    # Each arm makes two bends: north/south, east, then north/south again.
    "hidden_history_nav_v5": MapLayout("hidden_history_nav_v5", SHARED_PREFIX, (3, 3, 2)),
}


def get_map_layout(map_family_id: str) -> MapLayout:
    try:
        return MAP_LAYOUTS[map_family_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown map_family_id {map_family_id!r}; expected one of {sorted(MAP_LAYOUTS)}"
        ) from exc


def _hidden_label(value: str) -> HiddenLabel:
    if value not in ("left_blocked", "right_blocked"):
        raise ValueError(f"hidden_label must be left_blocked or right_blocked, got {value!r}")
    return value  # type: ignore[return-value]


def correct_branch(hidden_label: HiddenLabel) -> Branch:
    """Return the open branch for a privileged world label."""

    return "right" if hidden_label == "left_blocked" else "left"


def expert_actions(hidden_label: HiddenLabel) -> tuple[int, ...]:
    """Return the legacy v1 route (kept for callers of the Step 2 API)."""

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
    map_family_id: str = MAP_FAMILY_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "hidden_label": self.hidden_label,
            "agent_pos": list(self.agent_pos),
            "agent_dir": self.agent_dir,
            "step_count": self.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "map_family_id": self.map_family_id,
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
            map_family_id=str(value.get("map_family_id", MAP_FAMILY_ID)),
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
        map_family_id: str = MAP_FAMILY_ID,
        render_mode: str | None = None,
    ):
        self.hidden_label = _hidden_label(hidden_label)
        layout = get_map_layout(map_family_id)
        self.map_family_id = layout.map_family_id
        self._episode_seed = 0
        self._episode_done = True
        mission_space = MissionSpace(mission_func=lambda: MISSION)
        super().__init__(
            mission_space=mission_space,
            width=layout.width,
            height=layout.height,
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

    @property
    def goal_position(self) -> tuple[int, int]:
        return self._goal_position

    def goal_for_hidden_label(self, hidden_label: HiddenLabel | None = None) -> tuple[int, int]:
        """Return the privileged goal coordinate for a world label."""

        label = self.hidden_label if hidden_label is None else _hidden_label(hidden_label)
        return self._branch_endpoints[correct_branch(label)]

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        layout = get_map_layout(self.map_family_id)
        # Fill the interior with walls, then carve the family-specific public
        # prefix and two symmetric branches. The two hidden worlds
        # differ only at the branch endpoints and in the early clue color.
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())

        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        position = layout.start_pos
        direction = layout.start_dir
        public_cells = {position}
        for action in layout.prefix_actions:
            if action == 0:
                direction = (direction - 1) % 4
            elif action == 1:
                direction = (direction + 1) % 4
            elif action == 2:
                dx, dy = directions[direction]
                position = (position[0] + dx, position[1] + dy)
                if not (1 <= position[0] < width - 1 and 1 <= position[1] < height - 1):
                    raise ValueError(f"layout {layout.map_family_id} leaves the map at {position}")
                public_cells.add(position)
            else:
                raise ValueError(f"invalid layout action {action}")
        handoff_position, handoff_direction = position, direction
        branch_endpoints: dict[Branch, tuple[int, int]] = {}
        branch_cells: set[tuple[int, int]] = set()
        for branch in ("left", "right"):
            branch_direction = handoff_direction
            branch_position = handoff_position
            for action in layout.branch_actions(branch):
                if action < 2:
                    branch_direction = (branch_direction + (-1 if action == 0 else 1)) % 4
                else:
                    dx, dy = directions[branch_direction]
                    branch_position = (branch_position[0] + dx, branch_position[1] + dy)
                    if not (1 <= branch_position[0] < width - 1 and 1 <= branch_position[1] < height - 1):
                        raise ValueError(f"layout {layout.map_family_id} branch {branch} leaves the map")
                    if branch_position in public_cells or branch_position in branch_cells:
                        raise ValueError(f"layout {layout.map_family_id} branches overlap a corridor")
                    branch_cells.add(branch_position)
            branch_endpoints[branch] = branch_position
        for cell in public_cells | branch_cells:
            self.grid.set(*cell, None)
        if layout.clue_pos in public_cells | branch_cells:
            raise ValueError(f"layout {layout.map_family_id} clue obstructs a corridor")
        self.grid.set(*layout.clue_pos, Key("red" if self.hidden_label == "left_blocked" else "blue"))

        blocked_endpoint = branch_endpoints["left" if self.hidden_label == "left_blocked" else "right"]
        goal_endpoint = branch_endpoints[correct_branch(self.hidden_label)]
        self.grid.set(*blocked_endpoint, Wall())
        self.grid.set(*goal_endpoint, Goal())
        self._branch_endpoints = branch_endpoints
        self._goal_position = goal_endpoint
        self._handoff_position = handoff_position
        self._handoff_direction = handoff_direction

        self.agent_pos = layout.start_pos
        self.agent_dir = layout.start_dir

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        if "hidden_label" in options:
            self.hidden_label = _hidden_label(str(options["hidden_label"]))
        if "map_family_id" in options:
            self.map_family_id = get_map_layout(str(options["map_family_id"])).map_family_id
        layout = get_map_layout(self.map_family_id)
        self.width, self.height = layout.width, layout.height
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
            map_family_id=self.map_family_id,
        )

    def restore(self, snapshot: EnvSnapshot | Mapping[str, Any]) -> AgentObs:
        if not isinstance(snapshot, EnvSnapshot):
            snapshot = EnvSnapshot.from_dict(snapshot)
        self.reset(
            seed=snapshot.seed,
            options={"hidden_label": snapshot.hidden_label, "map_family_id": snapshot.map_family_id},
        )
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
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
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


def _plan_actions(env: HiddenHistoryNavEnv, goal: tuple[int, int]) -> tuple[int, ...]:
    """Privileged BFS planner over the actual family-specific MiniGrid."""

    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    start = (tuple(int(v) for v in env.agent_pos), int(env.agent_dir))
    queue: deque[tuple[tuple[int, int], int, tuple[int, ...]]] = deque([(start[0], start[1], ())])
    visited = {start}
    while queue:
        pos, direction, path = queue.popleft()
        if pos == goal:
            return path
        for action in (0, 1, 2):
            if action == 0:
                next_pos, next_direction = pos, (direction - 1) % 4
            elif action == 1:
                next_pos, next_direction = pos, (direction + 1) % 4
            else:
                dx, dy = directions[direction]
                candidate = (pos[0] + dx, pos[1] + dy)
                cell = env.grid.get(*candidate)
                can_enter = cell is None or getattr(cell, "can_overlap", lambda: False)()
                next_pos, next_direction = (candidate, direction) if can_enter else (pos, direction)
            state = (next_pos, next_direction)
            if state not in visited:
                visited.add(state)
                queue.append((next_pos, next_direction, path + (action,)))
    raise ContractError(f"no privileged route to goal {goal} in {env.map_family_id}")


def expert_actions_for_env(env: HiddenHistoryNavEnv) -> tuple[int, ...]:
    """Generate the full route, including the family-specific public prefix."""

    prefix = get_map_layout(env.map_family_id).prefix_actions
    # The caller invokes this at reset, so replay the prefix in a temporary
    # environment to obtain the correct handoff pose without mutating `env`.
    actions = list(prefix)
    probe = HiddenHistoryNavEnv(hidden_label=env.hidden_label, map_family_id=env.map_family_id)
    try:
        probe.reset(seed=env._episode_seed, options={"hidden_label": env.hidden_label})
        for action in prefix:
            probe.step(action)
        actions.extend(_plan_actions(probe, probe.goal_position))
    finally:
        probe.close()
    return tuple(actions)


def _run_episode(seed: int, hidden_label: HiddenLabel, map_family_id: str = MAP_FAMILY_ID) -> HiddenHistoryEpisode:
    env = HiddenHistoryNavEnv(hidden_label=hidden_label, map_family_id=map_family_id)
    observations: list[AgentObs] = []
    snapshots: list[EnvSnapshot] = []
    rewards: list[float] = []
    terminated_steps: list[bool] = []
    truncated_steps: list[bool] = []
    terminated = truncated = False
    try:
        obs, _ = env.reset(seed=seed, options={"hidden_label": hidden_label})
        observations.append(obs)
        snapshots.append(env.snapshot())
        actions = expert_actions_for_env(env)
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            snapshots.append(env.snapshot())
            rewards.append(float(reward))
            terminated_steps.append(bool(terminated))
            truncated_steps.append(bool(truncated))
            if terminated or truncated:
                break
    finally:
        env.close()
    return HiddenHistoryEpisode(
        seed=seed,
        hidden_label=hidden_label,
        map_family_id=map_family_id,
        mission=MISSION,
        handoff_t=HANDOFF_T,
        observations=tuple(observations),
        actions=tuple(actions[: len(observations) - 1]),
        expert_actions=actions,
        snapshots=tuple(snapshots),
        rewards=tuple(rewards),
        terminated=tuple(terminated_steps),
        truncated=tuple(truncated_steps),
        reached_goal=bool(terminated and not truncated),
    )


def generate_counterfactual_pair(seed: int, map_family_id: str = MAP_FAMILY_ID) -> CounterfactualPair:
    """Generate two paired worlds with a common public handoff prefix."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    map_family_id = get_map_layout(map_family_id).map_family_id
    pair = CounterfactualPair(
        seed=seed,
        left=_run_episode(seed, "left_blocked", map_family_id),
        right=_run_episode(seed, "right_blocked", map_family_id),
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
    if t < k or any(
        len(episode.observations) <= t or len(episode.actions) <= t
        or len(episode.snapshots) <= t or len(episode.expert_actions) <= t
        for episode in (a, b)
    ):
        return PairValidationReport(False, {"valid_handoff_window": False})
    checks: dict[str, bool] = {
        "same_map_family": a.map_family_id == b.map_family_id,
        "same_seed": a.seed == b.seed == pair.seed,
        "same_handoff_t": a.handoff_t == b.handoff_t == t,
        "same_handoff_pose": (
            a.snapshots[t].agent_pos == b.snapshots[t].agent_pos
            and a.snapshots[t].agent_dir == b.snapshots[t].agent_dir
        ),
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
