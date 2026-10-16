"""Small action policy heads trained on frozen RSSM posterior features."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cmb_minigrid.contracts import ContractError


class ActorPolicy(nn.Module):
    """Three-action behavior-cloning policy with a public task-goal token.

    The scalar goal token is fixed for this single-task experiment. It is an
    explicit interface slot, not a hidden-label or correct-branch input.
    """

    action_dim = 3
    goal_dim = 1

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim + self.goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def _goal(self, features: Tensor, goal: Tensor | None) -> Tensor:
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ContractError(f"features must end in [{self.feature_dim}]")
        if goal is None:
            goal = features.new_ones(*features.shape[:-1], self.goal_dim)
        if goal.shape != (*features.shape[:-1], self.goal_dim):
            raise ContractError("goal must match feature leading dimensions and have width 1")
        return goal.to(features)

    def forward(self, features: Tensor, goal: Tensor | None = None) -> Tensor:
        return self.network(torch.cat((features, self._goal(features, goal)), -1))

    @torch.no_grad()
    def act(self, features: Tensor, goal: Tensor | None = None) -> Tensor:
        """Return deterministic public action IDs in ``[0, 3)``."""

        return self(features, goal).argmax(-1)
