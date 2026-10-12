"""Immutable interfaces shared by data collection, models, and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class ContractError(ValueError):
    """Raised when a shared experiment contract is violated."""


@dataclass(frozen=True)
class AgentObs:
    """Non-privileged observation available to a deployed agent."""

    image: np.ndarray
    direction: int
    task_id: str

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ContractError(
                "AgentObs.image must have shape (height, width, 3); "
                f"received {tuple(image.shape)}"
            )
        if not np.issubdtype(image.dtype, np.integer):
            raise ContractError(f"AgentObs.image must use an integer dtype, got {image.dtype}")
        if not isinstance(self.direction, (int, np.integer)) or self.direction < 0:
            raise ContractError("AgentObs.direction must be a non-negative integer")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ContractError("AgentObs.task_id must be a non-empty string")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "direction", int(self.direction))


@dataclass(frozen=True)
class RSSMState:
    """Posterior state after its observation has been consumed.

    Batch-first shapes are required: ``deter`` is ``[B, D]`` and ``stoch`` is
    ``[B, G, C]`` for ``G`` categorical groups with ``C`` classes each.
    """

    deter: Any
    stoch: Any
    stoch_logits: Any | None = None

    def __post_init__(self) -> None:
        deter_shape = _shape(self.deter)
        stoch_shape = _shape(self.stoch)
        if len(deter_shape) != 2:
            raise ContractError(f"RSSMState.deter must be [B, D], got {deter_shape}")
        if len(stoch_shape) != 3:
            raise ContractError(f"RSSMState.stoch must be [B, G, C], got {stoch_shape}")
        if deter_shape[0] != stoch_shape[0]:
            raise ContractError("RSSMState.deter and stoch batch sizes must match")
        if any(size <= 0 for size in (*deter_shape, *stoch_shape)):
            raise ContractError("RSSMState dimensions must be positive")
        if self.stoch_logits is not None and _shape(self.stoch_logits) != stoch_shape:
            raise ContractError("RSSMState.stoch_logits must have the same shape as stoch")


@dataclass(frozen=True)
class Trajectory:
    """Trajectory with canonical ``obs[0:T+1]``/``action[0:T]`` layout."""

    observations: Sequence[AgentObs]
    actions: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.observations) != len(self.actions) + 1:
            raise ContractError(
                "A trajectory must contain obs[0:T+1] and action[0:T]; "
                f"received {len(self.observations)} observations and {len(self.actions)} actions"
            )
        if not self.observations:
            raise ContractError("A trajectory must contain at least obs[0]")
        if not all(isinstance(obs, AgentObs) for obs in self.observations):
            raise ContractError("Trajectory.observations must contain AgentObs values")
        if not all(isinstance(action, (int, np.integer)) and action >= 0 for action in self.actions):
            raise ContractError("Trajectory.actions must contain non-negative integer actions")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "actions", tuple(int(action) for action in self.actions))

    @property
    def horizon(self) -> int:
        return len(self.actions)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ContractError("RSSM state values must expose a shape attribute")
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Invalid tensor shape: {shape!r}") from exc
