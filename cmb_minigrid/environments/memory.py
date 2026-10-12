"""MiniGrid-Memory adapter with the frozen non-privileged AgentObs contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.actions import Actions

from cmb_minigrid.contracts import AgentObs, ContractError

ENV_ID = "MiniGrid-MemoryS13-v0"
TASK_ID = "minigrid_memory_s13"
ACTION_NAMES = ("left", "right", "forward")
_ACTION_MAP = (Actions.left, Actions.right, Actions.forward)


class AgentObsSpace(spaces.Space[AgentObs]):
    """Gymnasium space for the structured, public AgentObs value."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (7, 7, 3),
        task_id: str = TASK_ID,
        seed=None,
    ):
        self.image_shape = image_shape
        self.task_id = task_id
        super().__init__(shape=None, dtype=None, seed=seed)

    def sample(self, mask=None, probability=None) -> AgentObs:
        del mask, probability
        image = self.np_random.integers(0, 256, size=self.image_shape, dtype=np.uint8)
        return AgentObs(image=image, direction=int(self.np_random.integers(0, 4)), task_id=self.task_id)

    def contains(self, value: Any) -> bool:
        return (
            isinstance(value, AgentObs)
            and value.image.shape == self.image_shape
            and value.image.dtype == np.uint8
            and 0 <= value.direction < 4
            and value.task_id == self.task_id
        )


class MiniGridMemoryEnv(gym.Env[AgentObs, int]):
    """Expose MiniGrid Memory with only left/right/forward and AgentObs output."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, *, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render mode: {render_mode!r}")
        self.render_mode = render_mode
        self._env = gym.make(ENV_ID, render_mode=render_mode)
        raw_shape = self._env.observation_space["image"].shape
        self.observation_space = AgentObsSpace(tuple(raw_shape))
        self.action_space = spaces.Discrete(len(_ACTION_MAP))
        self._episode_done = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        raw_obs, _privileged_info = self._env.reset(seed=seed, options=options)
        self._episode_done = False
        return self._to_agent_obs(raw_obs), {}

    def step(self, action: int):
        if self._episode_done:
            raise RuntimeError("reset() must be called before stepping a completed episode")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, 2], got {action!r}")
        raw_obs, reward, terminated, truncated, _privileged_info = self._env.step(
            _ACTION_MAP[int(action)]
        )
        self._episode_done = bool(terminated or truncated)
        return self._to_agent_obs(raw_obs), float(reward), bool(terminated), bool(truncated), {}

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _to_agent_obs(raw_obs: Mapping[str, Any]) -> AgentObs:
        try:
            image = np.array(raw_obs["image"], dtype=np.uint8, copy=True)
            direction = int(raw_obs["direction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"MiniGrid returned an invalid local observation: {exc}") from exc
        return AgentObs(image=image, direction=direction, task_id=TASK_ID)


def make_memory_env(*, render_mode: str | None = None) -> MiniGridMemoryEnv:
    """Create the official MiniGrid Memory S13 task behind the agent interface."""

    return MiniGridMemoryEnv(render_mode=render_mode)
