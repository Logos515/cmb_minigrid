"""Step 7 B-state initializers: Reset, Replay-N, Linear/MLP A->B and Oracle-B.

Every method returns a batch-1 B ``RSSMState`` that has already fused
``obs[t]`` (or, for the A->B maps, inherits A's fusion of it). The caller
continues with ``continue_from_handoff`` using ``action[t]`` and ``obs[t+1]``;
``obs[t]`` must never be fed to B again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cmb_minigrid.contracts import ContractError, RSSMState
from cmb_minigrid.environments.hidden_history import EnvSnapshot, HiddenHistoryNavEnv
from cmb_minigrid.handoff.dataset import HandoffHistory, HandoffSample, filter_history
from cmb_minigrid.models import RSSM

# --------------------------------------------------------------------------- validity


def validate_b_state(rssm_b: RSSM, state: RSSMState, *, atol: float = 1e-5) -> None:
    """Raise unless ``state`` is a legal B state: bounded deter, one-hot stoch, normalized logits."""
    c = rssm_b.config
    if state.stoch_logits is None:
        raise ContractError("B state needs stoch_logits")
    if state.deter.shape[1:] != (c.deter,) or state.stoch.shape[1:] != (c.stoch_groups, c.stoch_classes):
        raise ContractError("B state shape does not match RSSM-B")
    for name in ("deter", "stoch", "stoch_logits"):
        if not torch.isfinite(getattr(state, name)).all():
            raise ContractError(f"B state {name} is not finite")
    if state.deter.abs().max() > 1 + atol:
        raise ContractError("B deter outside the GRU range [-1,1]")
    if not (((state.stoch == 0) | (state.stoch == 1)).all() and (state.stoch.sum(-1) == 1).all()):
        raise ContractError("B stoch must be exactly one-hot per group")
    if not torch.allclose(state.stoch_logits.logsumexp(-1),
                          torch.zeros_like(state.stoch_logits[..., 0]), atol=atol):
        raise ContractError("B stoch_logits must be normalized log probabilities")


# --------------------------------------------------------------------------- method interface


@dataclass(frozen=True)
class HandoffInput:
    """What an initializer may see. ``oracle_state_b`` is set only for Oracle-B."""

    history: HandoffHistory
    state_a: RSSMState
    oracle_state_b: RSSMState | None = None


@dataclass(frozen=True)
class HandoffInit:
    method: str
    state_b: RSSMState
    t: int
    env_snapshot: dict[str, Any]
    b_fused_frames: tuple[int, ...]  # observation indices fed to B's observe_step


class ReplayInit:
    """Filter only the most recent ``n`` frames through B from ``initial(1)``.

    Frame ``t-n+1`` is the first one fused, with no previous action (the action
    before the window is deliberately not read). ``n=1`` is Reset-B.
    """

    def __init__(self, rssm_b: RSSM, n: int, name: str | None = None):
        self.rssm_b = rssm_b
        self.n = n
        self.name = name or f"replay_{n}"

    def frames(self, t: int) -> tuple[int, ...]:
        return tuple(range(max(0, t + 1 - self.n), t + 1))

    def initialize(self, inp: HandoffInput, *, sample: bool = False, seed: int | None = None):
        window = inp.history.window(self.n)
        return (filter_history(self.rssm_b, window, sample=sample, seed=seed),
                self.frames(inp.history.t))


class ResetInit(ReplayInit):
    def __init__(self, rssm_b: RSSM):
        super().__init__(rssm_b, 1, "reset")


class OracleInit:
    """Full-history B filtering, taken from the stored ``state_b_star`` reference."""

    name = "oracle_b"

    def __init__(self, rssm_b: RSSM):
        self.rssm_b = rssm_b

    def initialize(self, inp: HandoffInput, *, sample: bool = False, seed: int | None = None):
        if inp.oracle_state_b is None:
            raise ContractError("Oracle-B needs the stored state_b_star")
        return inp.oracle_state_b, tuple(range(inp.history.t + 1))


class StateMap(nn.Module):
    """A feature -> B (deter, stoch logits). ``kind`` is 'linear' or 'mlp'.

    deter is squashed by tanh to B's GRU range; logits are raw before
    normalization. The MLP optionally narrows to a ``bottleneck`` width so its
    payload can be matched to a CMB later.
    """

    def __init__(self, kind: str, in_dim: int, deter: int, groups: int = 16, classes: int = 16,
                 hidden: int = 128, bottleneck: int | None = None):
        super().__init__()
        if kind not in {"linear", "mlp"}:
            raise ValueError("kind must be linear or mlp")
        self.kind, self.in_dim, self.deter = kind, in_dim, deter
        self.groups, self.classes = groups, classes
        self.hidden, self.bottleneck = hidden, bottleneck
        out = deter + groups * classes
        if kind == "linear":
            self.net = nn.Linear(in_dim, out)
        elif bottleneck is None:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out))
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, bottleneck),
                nn.Linear(bottleneck, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def config(self) -> dict[str, Any]:
        return {"kind": self.kind, "in_dim": self.in_dim, "deter": self.deter,
                "groups": self.groups, "classes": self.classes, "hidden": self.hidden,
                "bottleneck": self.bottleneck}

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        out = self.net(features)
        deter = torch.tanh(out[..., :self.deter])
        return deter, out[..., self.deter:].reshape(*out.shape[:-1], self.groups, self.classes)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def matched_hidden_dim(target_params: int, *, in_dim: int, out_dim: int,
                       bottleneck: int | None) -> int:
    """Hidden width whose MLP parameter count is closest to ``target_params``."""
    candidates = []
    for h in range(1, 2049):
        if bottleneck is None:
            n = in_dim * h + h + h * out_dim + out_dim
        else:
            n = (in_dim * h + h + h * bottleneck + bottleneck + bottleneck * h + h
                 + h * out_dim + out_dim)
        candidates.append((abs(n - target_params), h))
    return min(candidates)[1]


class MapInit:
    """Linear / MLP A->B: map ``features(state_A[t])`` to a legal B posterior."""

    def __init__(self, name: str, module: StateMap, rssm_a: RSSM, rssm_b: RSSM):
        self.name, self.module, self.rssm_a, self.rssm_b = name, module.eval(), rssm_a, rssm_b

    def initialize(self, inp: HandoffInput, *, sample: bool = False, seed: int | None = None):
        with torch.no_grad():
            deter, raw = self.module(self.rssm_a.features(inp.state_a))
            logits = self.rssm_b._logits(raw)  # same unimix + log-softmax as B's own heads
            if sample and seed is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    stoch = self.rssm_b.sample(logits, sample=True)
            else:
                stoch = self.rssm_b.sample(logits, sample=sample)
        return RSSMState(deter, stoch, logits), ()


# --------------------------------------------------------------------------- running


def initialize_all(
    methods: list, sample: HandoffSample, *, stochastic: bool = False, seed: int | None = None,
    rssm_b: RSSM | None = None,
) -> dict[str, HandoffInit]:
    """Run every method from the same ``HandoffSample`` (same snapshot and obs_t).

    Methods only see ``history`` and ``state_a``; the future is never passed in
    and ``state_b_star`` only to Oracle-B. When ``rssm_b`` is given, each
    returned state is checked for legality.
    """
    results = {}
    for method in methods:
        inp = HandoffInput(sample.history, sample.state_a,
                           sample.state_b_star if isinstance(method, OracleInit) else None)
        state, frames = method.initialize(inp, sample=stochastic, seed=seed)
        if rssm_b is not None:
            validate_b_state(rssm_b, state)
        results[method.name] = HandoffInit(method.name, state, sample.t,
                                           dict(sample.env_snapshot), frames)
    return results


def continue_from_handoff(rssm_b: RSSM, init: HandoffInit | RSSMState, action_t: int,
                          next_obs: dict[str, Any], *, sample: bool = False) -> RSSMState:
    """state[t+1]: fuse ``obs[t+1]`` after ``action[t]``; obs[t] is NOT re-fused."""
    state = init.state_b if isinstance(init, HandoffInit) else init
    action = torch.tensor([action_t], dtype=torch.long, device=state.deter.device)
    with torch.no_grad():
        return rssm_b.observe_step(state, action, next_obs, sample=sample)


def restore_env(sample: HandoffSample) -> tuple[HiddenHistoryNavEnv, Any]:
    """Fresh simulator restored at the sample's snapshot; checks it reproduces ``obs_t``."""
    snapshot = EnvSnapshot.from_dict(sample.env_snapshot)
    env = HiddenHistoryNavEnv(hidden_label=snapshot.hidden_label,
                              map_family_id=snapshot.map_family_id)
    obs = env.restore(snapshot)
    h = sample.history
    if not (torch.equal(torch.from_numpy(obs.image.copy()), h.images[-1])
            and obs.direction == int(h.directions[-1])):
        env.close()
        raise ContractError("restored snapshot does not reproduce obs_t")
    return env, obs


# --------------------------------------------------------------------------- fitting maps


def collect_aligned_states(rssm_a: RSSM, rssm_b: RSSM, records) -> dict[str, Tensor]:
    """state_A[t'] -> state_B*[t'] pairs over every step of the given (train) records.

    Both models consume the same public history, so the pair is aligned.
    """
    from cmb_minigrid.training.rssm import model_batch

    feats, deter, logits = [], [], []
    device = next(rssm_a.parameters()).device
    with torch.no_grad():
        for record in records:
            batch = model_batch([record], device)
            args = (batch["agent_obs"], batch["actions"])
            seq_a = rssm_a.observe(*args, observation_mask=batch["observation_mask"], sample=False)
            seq_b = rssm_b.observe(*args, observation_mask=batch["observation_mask"], sample=False)
            n = len(record.agent_obs)
            feats.append(seq_a.features[0, :n].cpu())
            deter.append(torch.stack([s.deter[0] for s in seq_b.states[:n]]).cpu())
            logits.append(seq_b.posterior_logits[0, :n].cpu())
    return {"features": torch.cat(feats), "deter": torch.cat(deter), "logits": torch.cat(logits)}


def map_loss(module: StateMap, data: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
    """MSE on deter plus KL(B* posterior || mapped categorical), summed over groups."""
    deter, raw = module(data["features"])
    mse = F.mse_loss(deter, data["deter"])
    target = data["logits"].log_softmax(-1)
    kl = (target.exp() * (target - raw.log_softmax(-1))).sum((-1, -2)).mean()
    return mse + kl, {"deter_mse": mse.item(), "stoch_kl": kl.item()}


def fit_state_map(
    module: StateMap, train: dict[str, Tensor], *, steps: int = 500, lr: float = 3e-3,
    weight_decay: float = 0.0, seed: int = 0,
) -> dict[str, float]:
    """Full-batch Adam; deterministic given ``seed``. Returns first/last train loss."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    module.train()
    first = last = float("nan")
    for step in range(steps):
        loss, _ = map_loss(module, train)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite map loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    module.eval()
    return {"first_loss": first, "last_loss": last}


def save_state_map(module: StateMap, path: str | Path, extra: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "config": module.config(), "model": module.state_dict(),
                **extra}, target)
    return target


def load_state_map(path: str | Path) -> tuple[StateMap, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    module = StateMap(**payload["config"])
    module.load_state_dict(payload["model"])
    return module.eval(), payload


# --------------------------------------------------------------------------- comparison


def compare_states(a: RSSMState, b: RSSMState) -> dict[str, float | bool]:
    """Distances between two batch-1 B states (deterministic part and categorical modes)."""
    same = a.stoch.argmax(-1) == b.stoch.argmax(-1)
    return {
        "deter_l2": (a.deter - b.deter).norm().item(),
        "deter_max_abs": (a.deter - b.deter).abs().max().item(),
        "stoch_group_agreement": same.float().mean().item(),
        "stoch_exact": bool(same.all().item()),
    }
