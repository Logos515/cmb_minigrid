"""Small categorical RSSM, inspired by DreamerV3 rather than a JAX port.

``state[t]`` has consumed ``obs[t]``. Filtering obs[0] uses a zero previous
action; reward/continue for action[t] are decoded from posterior state[t+1].
Only public image/direction/task_id observations are accepted. The task text
is fixed in this experiment and has no language encoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cmb_minigrid.contracts import ContractError, RSSMState


@dataclass(frozen=True)
class RSSMConfig:
    deter: int = 128
    stoch_groups: int = 16
    stoch_classes: int = 16
    embed_dim: int = 64
    hidden_dim: int = 128
    unimix: float = 0.01
    free_nats: float = 1.0
    dynamics_scale: float = 0.5
    representation_scale: float = 0.1
    reward_scale: float = 1.0
    continue_scale: float = 1.0

    def __post_init__(self):
        for name in ("deter", "stoch_groups", "stoch_classes", "embed_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.embed_dim, self.stoch_groups, self.stoch_classes) != (64, 16, 16):
            raise ValueError("Step 4 requires embed_dim=64 and stoch=16x16")
        if not math.isfinite(self.unimix) or not 0 <= self.unimix < 1:
            raise ValueError("unimix must be finite and in [0,1)")
        for name in ("free_nats", "dynamics_scale", "representation_scale",
                     "reward_scale", "continue_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RSSMConfig:
        return cls(**values)


@dataclass(frozen=True)
class RSSMSequence:
    """T+1 batch-first posterior states, retaining the single-state contract."""

    states: tuple[RSSMState, ...]
    prior_logits: Tensor  # [B,T+1,G,C], computed before observation conditioning.

    @property
    def features(self) -> Tensor:
        return torch.stack([torch.cat((s.deter, s.stoch.flatten(1)), -1)
                            for s in self.states], 1)

    @property
    def posterior_logits(self) -> Tensor:
        return torch.stack([s.stoch_logits for s in self.states], 1)


def categorical_kl(
    posterior_logits: Tensor, prior_logits: Tensor, *, free_nats: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return per-sample dynamics/representation KL, with groups summed first.

    Dynamics: KL(sg(q)||p); representation: KL(q||sg(p)). Free nats clamp
    the aggregate [..] sample KL, not each of its 16 categorical groups.
    Inputs may have leading batch/time axes and must end in [G,C].
    """
    if posterior_logits.shape != prior_logits.shape or posterior_logits.ndim < 2:
        raise ContractError("KL logits must have matching [...,G,C] shapes")
    if not math.isfinite(free_nats) or free_nats < 0:
        raise ValueError("free_nats must be finite and non-negative")
    q = posterior_logits.log_softmax(-1)
    p = prior_logits.log_softmax(-1)
    dynamics = (q.detach().exp() * (q.detach() - p)).sum((-1, -2))
    representation = (q.exp() * (q - p.detach())).sum((-1, -2))
    return dynamics.clamp_min(free_nats), representation.clamp_min(free_nats)


def _integers(tensor: Tensor) -> bool:
    return tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class SymbolicEncoder(nn.Module):
    """Three categorical channel embeddings + small MLP; not an RGB encoder."""

    categories = (11, 6, 3)  # MiniGrid object/color/state vocabularies.

    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, 4) for n in self.categories])
        self.network = nn.Sequential(nn.Flatten(1), nn.Linear(7 * 7 * 12, 128),
                                     nn.SiLU(), nn.Linear(128, 64), nn.LayerNorm(64))

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1:] != (7, 7, 3) or not _integers(image):
            raise ContractError("symbolic image must have integer shape [B,7,7,3]")
        for channel, classes in enumerate(self.categories):
            values = image[..., channel]
            if ((values < 0) | (values >= classes)).any():
                raise ContractError(f"image channel {channel} has invalid categorical IDs")
        embedded = torch.cat([emb(image[..., i].long())
                              for i, emb in enumerate(self.embeddings)], -1)
        return self.network(embedded)


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class RSSM(nn.Module):
    action_dim = 3

    def __init__(self, config: RSSMConfig | None = None):
        super().__init__()
        self.config = config or RSSMConfig()
        c = self.config
        self.stoch_dim = c.stoch_groups * c.stoch_classes
        self.feature_dim = c.deter + self.stoch_dim
        self.encoder = SymbolicEncoder()
        self.gru = nn.GRUCell(self.stoch_dim + self.action_dim, c.deter)
        self.prior = _mlp(c.deter, c.hidden_dim, self.stoch_dim)
        self.posterior = _mlp(c.deter + c.embed_dim + 4, c.hidden_dim, self.stoch_dim)
        self.image_head = _mlp(self.feature_dim, c.hidden_dim,
                               7 * 7 * sum(SymbolicEncoder.categories))
        self.direction_head = _mlp(self.feature_dim, c.hidden_dim, 4)
        self.reward_head = _mlp(self.feature_dim, c.hidden_dim, 1)
        self.continue_head = _mlp(self.feature_dim, c.hidden_dim, 1)

    def initial(self, batch: int) -> RSSMState:
        """Zero deter and category-zero one-hot sentinel, on model device/dtype.

        This sentinel has not consumed obs[0]. Call observe_step with None (or
        an all-zero action vector) once to obtain posterior state[0].
        """
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ContractError("initial batch must be a positive integer")
        c = self.config
        param = next(self.parameters())
        deter = param.new_zeros(batch, c.deter)
        logits = param.new_zeros(batch, c.stoch_groups, c.stoch_classes)
        stoch = F.one_hot(logits.argmax(-1), c.stoch_classes).to(param.dtype)
        return RSSMState(deter, stoch, logits)

    def _check_state(self, state: RSSMState) -> None:
        c = self.config
        if (state.deter.shape[1:] != (c.deter,)
                or state.stoch.shape[1:] != (c.stoch_groups, c.stoch_classes)):
            raise ContractError("RSSM state shape does not match model configuration")

    def features(self, state: RSSMState) -> Tensor:
        self._check_state(state)
        return torch.cat((state.deter, state.stoch.flatten(1)), -1)

    def _action(self, action: Tensor | None, state: RSSMState) -> Tensor:
        batch = state.deter.shape[0]
        if action is None:
            return state.deter.new_zeros(batch, self.action_dim)
        if action.shape == (batch,) and _integers(action):
            if ((action < 0) | (action >= self.action_dim)).any():
                raise ContractError("action IDs must be in [0,3)")
            return F.one_hot(action.long(), self.action_dim).to(state.deter)
        if action.shape == (batch, self.action_dim):
            valid_entries = (action == 0) | (action == 1)
            valid_sums = (action.sum(-1) == 0) | (action.sum(-1) == 1)
            if valid_entries.all() and valid_sums.all():
                return action.to(state.deter)
        raise ContractError("action must be integer [B] or one-hot/zero [B,3]")

    def _transition(self, state: RSSMState, action: Tensor | None) -> Tensor:
        self._check_state(state)
        return self.gru(torch.cat((state.stoch.flatten(1), self._action(action, state)), -1),
                        state.deter)

    def _logits(self, network_output: Tensor) -> Tensor:
        c = self.config
        logits = network_output.reshape(-1, c.stoch_groups, c.stoch_classes)
        log_probs = logits.log_softmax(-1)
        if c.unimix == 0:
            return log_probs
        return torch.logaddexp(log_probs + math.log1p(-c.unimix),
                               torch.full_like(log_probs, math.log(c.unimix / c.stoch_classes)))

    def sample(self, logits: Tensor, *, sample: bool | None = None) -> Tensor:
        """Hard one-hot forward; straight-through probability gradients.

        Training samples categorical groups; eval uses argmax. An explicit
        sample=True/False overrides this behavior for diagnostic use.
        """
        stochastic = self.training if sample is None else sample
        probs = logits.softmax(-1)
        indices = (torch.distributions.Categorical(logits=logits).sample()
                   if stochastic else logits.argmax(-1))
        hard = F.one_hot(indices, self.config.stoch_classes).to(logits.dtype)
        return hard + (probs - probs.detach())

    @staticmethod
    def _check_obs(obs: Mapping[str, Any]) -> None:
        if not isinstance(obs, Mapping) or set(obs) != {"image", "direction", "task_id"}:
            raise ContractError("observation must contain only image, direction, task_id")

    def observe_step(
        self, prev_state: RSSMState, prev_action: Tensor | None, obs: Mapping[str, Any],
        *, sample: bool | None = None,
    ) -> RSSMState:
        """Fuse the next public observation exactly once and return its posterior."""
        self._check_obs(obs)
        deter = self._transition(prev_state, prev_action)
        image, direction = obs["image"], obs["direction"]
        batch = deter.shape[0]
        if (image.shape[0] != batch or direction.shape != (batch,)
                or not _integers(direction) or ((direction < 0) | (direction >= 4)).any()):
            raise ContractError("observation batch/direction must be integer [B] in [0,4)")
        embedding = self.encoder(image)
        direction = F.one_hot(direction.long(), 4).to(embedding)
        logits = self._logits(self.posterior(torch.cat((deter, embedding, direction), -1)))
        return RSSMState(deter, self.sample(logits, sample=sample), logits)

    def imagine_step(
        self, state: RSSMState, action: Tensor, *, sample: bool | None = None,
    ) -> RSSMState:
        """Predict state[t+1] from state[t],action[t] with no observation access."""
        deter = self._transition(state, action)
        logits = self._logits(self.prior(deter))
        return RSSMState(deter, self.sample(logits, sample=sample), logits)

    def observe(
        self, obs: Mapping[str, Any], actions: Tensor, observation_mask: Tensor | None = None,
        *, sample: bool | None = None,
    ) -> RSSMSequence:
        """Filter [B,T+1] observations with [B,T] actions; freeze padded states."""
        self._check_obs(obs)
        image, direction = obs["image"], obs["direction"]
        if (image.ndim != 5 or image.shape[2:] != (7, 7, 3)
                or direction.shape != image.shape[:2]
                or actions.shape != (image.shape[0], image.shape[1] - 1)):
            raise ContractError("sequence shapes must be image[B,T+1,7,7,3], actions[B,T]")
        batch, count = image.shape[:2]
        mask = (torch.ones(batch, count, dtype=torch.bool, device=image.device)
                if observation_mask is None else observation_mask)
        if (mask.shape != (batch, count) or mask.dtype != torch.bool
                or not mask[:, 0].all() or ((~mask[:, :-1]) & mask[:, 1:]).any()):
            raise ContractError("observation mask must be a non-empty prefix per row")
        state = self.initial(batch)
        states, priors = [], []
        for t in range(count):
            valid = mask[:, t]
            step_obs = {
                "image": torch.where(valid[:, None, None, None], image[:, t], 0),
                "direction": torch.where(valid, direction[:, t], 0),
                "task_id": obs["task_id"],  # Fixed public instruction; unused by encoder.
            }
            action = None if t == 0 else torch.where(valid, actions[:, t - 1], 0)
            proposed = self.observe_step(state, action, step_obs, sample=sample)
            priors.append(self._logits(self.prior(proposed.deter)))
            state = RSSMState(
                torch.where(valid[:, None], proposed.deter, state.deter),
                torch.where(valid[:, None, None], proposed.stoch, state.stoch),
                torch.where(valid[:, None, None], proposed.stoch_logits, state.stoch_logits),
            )
            states.append(state)
        return RSSMSequence(tuple(states), torch.stack(priors, 1))

    def decode(self, features: Tensor) -> dict[str, Any]:
        """Decode arbitrary leading batch/time axes of feature vectors."""
        shape = features.shape[:-1]
        image = self.image_head(features).reshape(*shape, 7, 7, sum(SymbolicEncoder.categories))
        return {
            "image_logits": image.split(SymbolicEncoder.categories, -1),
            "direction_logits": self.direction_head(features),
            "reward": self.reward_head(features).squeeze(-1),
            "continue_logits": self.continue_head(features).squeeze(-1),
        }

    def loss(self, batch: Mapping[str, Any], *, sample: bool | None = None) -> dict[str, Tensor]:
        """Masked offline WM loss. No expert label or privileged metadata is read.

        Reconstruction includes obs[0:T+1]; transition heads use state[1:T+1].
        Continue target is ~terminated, so time-limit truncation bootstraps.
        Padding is sanitized before indexing and reduction, including NaNs.
        """
        obs_mask, mask = batch["observation_mask"], batch["transition_mask"]
        if (mask.dtype != torch.bool or mask.shape != batch["actions"].shape
                or obs_mask.shape != (mask.shape[0], mask.shape[1] + 1)
                or not torch.equal(mask, obs_mask[:, 1:])):
            raise ContractError("transition and observation masks must align")
        for name in ("reward", "terminated", "truncated"):
            if batch[name].shape != mask.shape:
                raise ContractError(f"{name} must have transition shape [B,T]")
        if batch["terminated"].dtype != torch.bool or batch["truncated"].dtype != torch.bool:
            raise ContractError("terminated/truncated must be boolean")
        done = (batch["terminated"] | batch["truncated"]) & mask
        if (done[:, :-1] & mask[:, 1:]).any():
            raise ContractError("no valid transitions are allowed after a terminal/truncated step")
        sequence = self.observe(batch["agent_obs"], batch["actions"], obs_mask, sample=sample)
        heads = self.decode(sequence.features)
        image = torch.where(obs_mask[..., None, None, None], batch["agent_obs"]["image"], 0)
        direction = torch.where(obs_mask, batch["agent_obs"]["direction"], 0)
        recon = []
        for channel, logits in enumerate(heads["image_logits"]):
            ce = F.cross_entropy(logits.flatten(0, -2), image[..., channel].long().flatten(),
                                 reduction="none")
            recon.append(ce.reshape(*obs_mask.shape, 7, 7).mean((-1, -2)))
        visual = torch.stack(recon).mean(0)
        direction_loss = F.cross_entropy(heads["direction_logits"].flatten(0, 1),
                                         direction.long().flatten(), reduction="none")
        reconstruction = visual + direction_loss.reshape(obs_mask.shape)
        reward_target = torch.where(mask, batch["reward"], 0)
        reward = (heads["reward"][:, 1:] - reward_target).square()
        continuation = F.binary_cross_entropy_with_logits(
            heads["continue_logits"][:, 1:], (~batch["terminated"]).float(), reduction="none")
        dynamics, representation = categorical_kl(
            sequence.posterior_logits, sequence.prior_logits, free_nats=self.config.free_nats)

        def mean(value, valid):
            return torch.where(valid, value, 0).sum() / valid.sum().clamp_min(1)

        losses = {
            "reconstruction": mean(reconstruction, obs_mask),
            "reward": mean(reward, mask),
            "continue": mean(continuation, mask),
            "dynamics_kl": mean(dynamics, obs_mask),
            "representation_kl": mean(representation, obs_mask),
        }
        c = self.config
        losses["total"] = (losses["reconstruction"] + c.reward_scale * losses["reward"]
                           + c.continue_scale * losses["continue"]
                           + c.dynamics_scale * losses["dynamics_kl"]
                           + c.representation_scale * losses["representation_kl"])
        return losses
