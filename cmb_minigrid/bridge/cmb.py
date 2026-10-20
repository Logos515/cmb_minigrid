"""Common Memory Bridge: Sender (A state -> discrete 4x16 message) and Receiver.

The Sender reads ``features(state_A[t])`` (A's deter and flattened stoch, which
already fused ``obs[t]``). The Receiver fuses the message with frozen B's
64-dim encoding of the *current* observation (plus the public direction
one-hot) and emits B's deter and 16x16 stochastic logits. B's own
``observe_step`` is never called on ``obs[t]``: the output is already
``state_B_hat[t]``. Frozen RSSM-B is passed to ``receive`` instead of being
stored, so it is not part of the bridge's parameters or checkpoint. Gradients
still flow through the frozen B modules (their weights have
``requires_grad=False``) back to the Sender and Receiver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cmb_minigrid.contracts import ContractError, RSSMState
from cmb_minigrid.models import RSSM


@dataclass(frozen=True)
class CMBConfig:
    a_feature_dim: int = 384
    b_deter: int = 256
    b_groups: int = 16
    b_classes: int = 16
    tokens: int = 4
    token_classes: int = 16
    encoder_dim: int = 64
    sender_hidden: int = 128
    receiver_hidden: int = 128

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def message_bits(self) -> float:
        return self.tokens * math.log2(self.token_classes)


@dataclass(frozen=True)
class CMBMessage:
    logits: Tensor  # [B,tokens,classes], normalized log probabilities
    onehot: Tensor  # [B,tokens,classes], hard one-hot forward, straight-through backward

    @property
    def indices(self) -> Tensor:
        """The transmitted payload: ``tokens`` category IDs per sample."""
        return self.onehot.argmax(-1)


def _straight_through(logits: Tensor, sample: bool) -> Tensor:
    probs = logits.softmax(-1)
    index = (torch.distributions.Categorical(logits=logits).sample() if sample
             else logits.argmax(-1))
    hard = F.one_hot(index, logits.shape[-1]).to(logits.dtype)
    return hard + (probs - probs.detach())


class CMBBridge(nn.Module):
    def __init__(self, config: CMBConfig | None = None):
        super().__init__()
        self.config = c = config or CMBConfig()
        self.sender = nn.Sequential(
            nn.Linear(c.a_feature_dim, c.sender_hidden), nn.SiLU(),
            nn.Linear(c.sender_hidden, c.tokens * c.token_classes))
        receiver_in = c.tokens * c.token_classes + c.encoder_dim + 4
        self.receiver = nn.Sequential(
            nn.Linear(receiver_in, c.receiver_hidden), nn.SiLU(),
            nn.Linear(c.receiver_hidden, c.b_deter + c.b_groups * c.b_classes))

    def send(self, features_a: Tensor, *, sample: bool = False) -> CMBMessage:
        c = self.config
        if features_a.ndim != 2 or features_a.shape[1] != c.a_feature_dim:
            raise ContractError(f"features_a must be [B,{c.a_feature_dim}]")
        logits = self.sender(features_a).reshape(-1, c.tokens, c.token_classes).log_softmax(-1)
        return CMBMessage(logits, _straight_through(logits, sample))

    def receive(self, message: CMBMessage, obs: Mapping[str, Any], rssm_b: RSSM,
                *, sample: bool = False) -> RSSMState:
        """Fuse the message with B's encoding of ``obs[t]``; returns state_B_hat[t]."""
        c = self.config
        if (rssm_b.config.deter, rssm_b.config.stoch_groups, rssm_b.config.stoch_classes,
                rssm_b.config.embed_dim) != (c.b_deter, c.b_groups, c.b_classes, c.encoder_dim):
            raise ContractError("CMB config does not match RSSM-B")
        encoding = rssm_b.encoder(obs["image"])
        direction = F.one_hot(obs["direction"].long(), 4).to(encoding)
        hidden = self.receiver(torch.cat(
            (message.onehot.flatten(1).to(encoding), encoding, direction), -1))
        deter = torch.tanh(hidden[:, :c.b_deter])  # B's GRU output lies in [-1,1]
        logits = rssm_b._logits(hidden[:, c.b_deter:])  # B's unimix + log-softmax
        return RSSMState(deter, rssm_b.sample(logits, sample=sample), logits)

    def forward(self, features_a: Tensor, obs: Mapping[str, Any], rssm_b: RSSM,
                *, sample: bool = False) -> tuple[RSSMState, CMBMessage]:
        message = self.send(features_a, sample=sample)
        return self.receive(message, obs, rssm_b, sample=sample), message

    def parameter_counts(self) -> dict[str, int]:
        sender = sum(p.numel() for p in self.sender.parameters())
        receiver = sum(p.numel() for p in self.receiver.parameters())
        return {"sender": sender, "receiver": receiver, "total": sender + receiver}


def save_cmb(bridge: CMBBridge, path, extra: dict[str, Any]) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save({"schema_version": 1, "cmb_config": asdict(bridge.config),
                "model": bridge.state_dict(), **extra}, temporary)
    temporary.replace(target)


def load_cmb(path, device="cpu") -> tuple[CMBBridge, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported CMB checkpoint schema")
    bridge = CMBBridge(CMBConfig(**payload["cmb_config"])).to(device)
    bridge.load_state_dict(payload["model"])
    return bridge.eval(), payload
