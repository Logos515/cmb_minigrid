"""Seeded random/scripted rollouts and canonical observation hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmb_minigrid.contracts import AgentObs
from cmb_minigrid.environments.memory import MiniGridMemoryEnv
from cmb_minigrid.reproducibility import set_seed

Policy = Callable[[AgentObs, int], int]
SCRIPTED_ACTIONS = (2, 2, 1, 2, 2, 1, 2, 0)


@dataclass(frozen=True)
class Rollout:
    seed: int
    observations: tuple[AgentObs, ...]
    actions: tuple[int, ...]
    observation_hashes: tuple[str, ...]
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool


def observation_hash(obs: AgentObs) -> str:
    """Hash only deployment-visible fields in a stable byte representation."""

    digest = hashlib.sha256()
    digest.update(obs.image.dtype.str.encode("ascii"))
    digest.update(json.dumps(obs.image.shape, separators=(",", ":")).encode("ascii"))
    digest.update(obs.image.tobytes(order="C"))
    digest.update(int(obs.direction).to_bytes(1, "little", signed=False))
    digest.update(obs.task_id.encode("utf-8"))
    return digest.hexdigest()


def rollout(
    env: MiniGridMemoryEnv,
    *,
    seed: int,
    steps: int = 20,
    actions: tuple[int, ...] | list[int] | None = None,
    render: bool = False,
) -> Rollout:
    """Run seeded random actions or an explicit action sequence.

    The result contains the initial observation plus one observation per
    transition, preserving the ``obs[0:T+1]`` / ``action[0:T]`` contract.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if actions is not None and len(actions) < steps:
        raise ValueError(f"script has {len(actions)} actions but {steps} steps were requested")

    set_seed(seed)
    action_rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    observations = [obs]
    hashes = [observation_hash(obs)]
    executed_actions: list[int] = []
    rewards: list[float] = []
    terminated = truncated = False

    for index in range(steps):
        action = int(actions[index]) if actions is not None else int(action_rng.integers(0, 3))
        obs, reward, terminated, truncated, _ = env.step(action)
        executed_actions.append(action)
        observations.append(obs)
        hashes.append(observation_hash(obs))
        rewards.append(float(reward))
        if render:
            env.render()
        if terminated or truncated:
            break

    return Rollout(
        seed=seed,
        observations=tuple(observations),
        actions=tuple(executed_actions),
        observation_hashes=tuple(hashes),
        rewards=tuple(rewards),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def write_rollout_json(trajectory: Rollout, path: str | Path) -> Path:
    """Write a self-contained sample with no simulator-private fields."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_id": "MiniGrid-MemoryS13-v0",
        "seed": trajectory.seed,
        "actions": list(trajectory.actions),
        "observations": [
            {
                "image": obs.image.tolist(),
                "direction": obs.direction,
                "task_id": obs.task_id,
            }
            for obs in trajectory.observations
        ],
        "observation_hashes": list(trajectory.observation_hashes),
        "rewards": list(trajectory.rewards),
        "terminated": trajectory.terminated,
        "truncated": trajectory.truncated,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
