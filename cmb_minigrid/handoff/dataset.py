"""Step 7 handoff dataset: one frozen-A/B filtered snapshot per trajectory.

For a record with handoff time ``t`` both frozen RSSMs call ``observe_step`` on
the same public prefix ``obs[0:t+1]``/``action[0:t]``, giving ``state_A[t]`` and
``state_B_star[t]``. Both states already fused ``obs[t]``. ``HandoffHistory``
holds only the prefix; the ground-truth continuation lives in a separate
``HandoffFuture`` so initializers cannot reach it by construction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cmb_minigrid.contracts import ContractError, RSSMState
from cmb_minigrid.data.dataset import SplitManifest
from cmb_minigrid.data.trajectory import TrajectoryRecord, replay_from_handoff
from cmb_minigrid.models import RSSM

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HandoffHistory:
    """Public prefix through the handoff: obs[0:t+1] and action[0:t]."""

    images: Tensor  # uint8 [t+1,7,7,3]; images[t] is obs_t
    directions: Tensor  # int64 [t+1]
    actions: Tensor  # int64 [t]
    task_id: str

    def __post_init__(self) -> None:
        if self.images.shape[0] < 1 or self.directions.shape != (self.images.shape[0],):
            raise ContractError("history needs matching images/directions with obs_t")
        if self.actions.shape != (self.images.shape[0] - 1,):
            raise ContractError("history requires exactly t actions for t+1 observations")

    @property
    def t(self) -> int:
        return int(self.images.shape[0]) - 1

    def window(self, n: int) -> HandoffHistory:
        """The most recent ``n`` frames (clipped at obs[0]) and the actions between them."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("window size must be a positive integer")
        start = max(0, self.t + 1 - n)
        return HandoffHistory(self.images[start:], self.directions[start:],
                              self.actions[start:], self.task_id)

    @property
    def obs_t(self) -> dict[str, Any]:
        return {"image": self.images[-1:], "direction": self.directions[-1:],
                "task_id": [self.task_id]}


@dataclass(frozen=True)
class HandoffFuture:
    """True continuation after the handoff; supervision/evaluation only."""

    images: Tensor  # uint8 [F,7,7,3] = obs[t+1:]
    directions: Tensor  # int64 [F]
    actions: Tensor  # int64 [F] = action[t:]
    expert_actions: Tensor  # int64 [F]
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor


@dataclass(frozen=True)
class HandoffSample:
    episode_id: str
    pair_id: str
    map_family_id: str
    split: str
    episode_role: str
    seed: int
    t: int
    env_snapshot: dict[str, Any]
    hidden_label: str  # privileged: offline evaluation only
    history: HandoffHistory
    future: HandoffFuture
    state_a: RSSMState  # batch 1, fused through obs_t
    state_b_star: RSSMState  # batch 1, fused through obs_t


# --------------------------------------------------------------------------- filtering


def filter_history(
    model: RSSM, history: HandoffHistory, *, sample: bool = False, seed: int | None = None,
) -> RSSMState:
    """Fuse every frame of ``history`` exactly once with ``observe_step``.

    The first frame is filtered from ``initial(1)`` with no previous action;
    frame ``i`` uses ``history.actions[i-1]``. Returns the batch-1 posterior
    after the last frame. ``sample=True`` with a ``seed`` is reproducible and
    does not disturb the global RNG.
    """
    device = next(model.parameters()).device
    images = history.images.to(device)
    directions = history.directions.to(device)
    actions = history.actions.to(device)

    def run() -> RSSMState:
        state = model.initial(1)
        for i in range(images.shape[0]):
            obs = {"image": images[i:i + 1], "direction": directions[i:i + 1],
                   "task_id": [history.task_id]}
            previous = None if i == 0 else actions[i - 1:i]
            state = model.observe_step(state, previous, obs, sample=sample)
        return state

    with torch.no_grad():
        if sample and seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return run()
        return run()


# --------------------------------------------------------------------------- building


def _history(record: TrajectoryRecord) -> HandoffHistory:
    t = record.handoff_t
    return HandoffHistory(
        images=torch.from_numpy(np.stack([o.image for o in record.agent_obs[:t + 1]])),
        directions=torch.tensor([o.direction for o in record.agent_obs[:t + 1]], dtype=torch.long),
        actions=torch.tensor(record.actions[:t], dtype=torch.long),
        task_id=record.agent_obs[0].task_id,
    )


def _future(record: TrajectoryRecord) -> HandoffFuture:
    t = record.handoff_t
    tail = record.agent_obs[t + 1:]
    images = (torch.from_numpy(np.stack([o.image for o in tail]))
              if tail else torch.zeros((0, 7, 7, 3), dtype=torch.uint8))
    return HandoffFuture(
        images=images,
        directions=torch.tensor([o.direction for o in tail], dtype=torch.long),
        actions=torch.tensor(record.actions[t:], dtype=torch.long),
        expert_actions=torch.tensor(record.expert_actions[t:], dtype=torch.long),
        rewards=torch.tensor(record.rewards[t:], dtype=torch.float32),
        terminated=torch.tensor(record.terminated[t:], dtype=torch.bool),
        truncated=torch.tensor(record.truncated[t:], dtype=torch.bool),
    )


def build_handoff_samples(
    records: list[TrajectoryRecord],
    manifest: SplitManifest,
    rssm_a: RSSM,
    rssm_b: RSSM,
    *,
    verify_snapshots: bool = True,
) -> tuple[list[HandoffSample], dict[str, Any]]:
    """Filter each record's public prefix with frozen A and B; skip no-handoff records."""
    samples: list[HandoffSample] = []
    skipped: list[str] = []
    for record in records:
        if record.env_snapshot is None:
            skipped.append(record.episode_id)
            continue
        if verify_snapshots and not replay_from_handoff(record):
            raise ContractError(f"snapshot cannot rebuild the handoff for {record.episode_id}")
        history = _history(record)
        samples.append(HandoffSample(
            episode_id=record.episode_id,
            pair_id=record.pair_id,
            map_family_id=record.map_family_id,
            split=manifest.assignment[record.episode_id],
            episode_role=record.episode_role,
            seed=record.seed,
            t=record.handoff_t,
            env_snapshot=dict(record.env_snapshot),
            hidden_label=str(record.privileged_state[0]["hidden_label"]),
            history=history,
            future=_future(record),
            state_a=filter_history(rssm_a, history),
            state_b_star=filter_history(rssm_b, history),
        ))
    return samples, {"skipped_without_handoff": skipped}


# --------------------------------------------------------------------------- persistence


def _state_to_dict(state: RSSMState) -> dict[str, Tensor]:
    return {"deter": state.deter[0].cpu(), "stoch": state.stoch[0].cpu(),
            "logits": state.stoch_logits[0].cpu()}


def _state_from_dict(value: dict[str, Tensor]) -> RSSMState:
    return RSSMState(value["deter"][None], value["stoch"][None], value["logits"][None])


def _tensors(obj) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def save_handoff_dataset(
    samples: list[HandoffSample], path: str | Path, *, meta: dict[str, Any] | None = None,
) -> Path:
    rows = []
    for s in samples:
        rows.append({
            "episode_id": s.episode_id, "pair_id": s.pair_id, "map_family_id": s.map_family_id,
            "split": s.split, "episode_role": s.episode_role, "seed": s.seed, "t": s.t,
            "env_snapshot": s.env_snapshot, "hidden_label": s.hidden_label,
            "history": _tensors(s.history),
            # Saved separately as well so consumers need not slice the prefix.
            "obs_t": {"image": s.history.images[-1], "direction": s.history.directions[-1]},
            "future": _tensors(s.future),
            "state_a": _state_to_dict(s.state_a),
            "state_b_star": _state_to_dict(s.state_b_star),
        })
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "meta": meta or {}, "samples": rows}, target)
    return target


def load_handoff_dataset(path: str | Path) -> tuple[list[HandoffSample], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported handoff dataset schema")
    samples = []
    for r in payload["samples"]:
        history = HandoffHistory(**r["history"])
        if not (torch.equal(history.images[-1], r["obs_t"]["image"])
                and int(history.directions[-1]) == int(r["obs_t"]["direction"])):
            raise ContractError(f"obs_t does not match the history tail for {r['episode_id']}")
        samples.append(HandoffSample(
            episode_id=r["episode_id"], pair_id=r["pair_id"], map_family_id=r["map_family_id"],
            split=r["split"], episode_role=r["episode_role"], seed=int(r["seed"]), t=int(r["t"]),
            env_snapshot=r["env_snapshot"], hidden_label=r["hidden_label"], history=history,
            future=HandoffFuture(**r["future"]),
            state_a=_state_from_dict(r["state_a"]),
            state_b_star=_state_from_dict(r["state_b_star"]),
        ))
    return samples, payload["meta"]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
