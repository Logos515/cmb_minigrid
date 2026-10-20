"""Step 9: closed-loop takeover from identical snapshots, History Swap, result tables.

For every (sample, method) the runner restores a *fresh* simulator from the
sample's handoff snapshot, builds B's posterior ``state_B[t]`` (already fused
with ``obs[t]``), lets Actor-B pick ``action[t]`` from that state, steps the real
environment and then fuses ``obs[t+1]`` into B, and so on until the episode ends.
``obs[t]`` is never fused a second time. The privileged BFS planner and hidden
label are used only to score wrong actions, collisions and path efficiency; they
are never inputs to B or the Actor.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from cmb_minigrid.bridge.cmb import CMBMessage
from cmb_minigrid.contracts import RSSMState
from cmb_minigrid.data.trajectory import _shortest_action
from cmb_minigrid.environments.hidden_history import _plan_actions
from cmb_minigrid.environments.rollout import observation_hash
from cmb_minigrid.handoff.dataset import HandoffSample
from cmb_minigrid.handoff.initializers import (
    HandoffInput,
    MapInit,
    OracleInit,
    ReplayInit,
    ResetInit,
    initialize_all,
    restore_env,
)
from cmb_minigrid.models import RSSM
from cmb_minigrid.models.policy import ActorPolicy

FRAME_BYTES = 7 * 7 * 3 + 1  # uint8 image + direction
TIMING_KEYS = ("takeover_ms", "closed_loop_ms")


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


class CMBInit:
    """``state_B_hat[t]`` from the CMB: Sender on ``features(state_A[t])``, Receiver on ``obs_t``."""

    def __init__(self, bridge, rssm_a: RSSM, rssm_b: RSSM, name: str = "cmb"):
        self.bridge, self.rssm_a, self.rssm_b, self.name = bridge.eval(), rssm_a, rssm_b, name

    def initialize(self, inp: HandoffInput, *, sample: bool = False, seed: int | None = None):
        with torch.no_grad():
            state, _ = self.bridge(self.rssm_a.features(inp.state_a), inp.history.obs_t,
                                   self.rssm_b, sample=sample)
        return state, ()


def payload_bytes(method, t: int) -> int:
    """Bytes A's side must hand to B's side at the handoff (``t`` = handoff index)."""
    if isinstance(method, ResetInit):
        return 0
    if isinstance(method, ReplayInit):
        w = min(method.n, t + 1)
        return w * FRAME_BYTES + (w - 1)  # raw frames plus the actions between them
    if isinstance(method, OracleInit):
        return (t + 1) * FRAME_BYTES + t  # unconstrained full public history
    if isinstance(method, MapInit):
        width = method.module.bottleneck or method.module.in_dim
        return width * 4  # float32
    if isinstance(method, CMBInit):
        return math.ceil(method.bridge.config.message_bits / 8)
    raise TypeError(f"unknown method type: {type(method).__name__}")


def _obs_tensor(obs) -> dict[str, Any]:
    return {"image": torch.from_numpy(obs.image.copy()).unsqueeze(0),
            "direction": torch.tensor([obs.direction], dtype=torch.long),
            "task_id": [obs.task_id]}


class HandoffRunner:
    def __init__(self, rssm_a: RSSM, rssm_b: RSSM, policy: ActorPolicy, methods: list,
                 *, stochastic: bool = False, seed: int = 0):
        self.rssm_a, self.rssm_b, self.policy = rssm_a, rssm_b, policy
        self.methods, self.stochastic, self.seed = methods, stochastic, seed

    @contextlib.contextmanager
    def _rng(self, index: int):
        """Per-run RNG scope: deterministic modes never touch it; sampling is seeded."""
        if not self.stochastic:
            yield
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 1_000_003 + index)
            yield

    @torch.no_grad()
    def closed_loop(self, sample: HandoffSample, env, state: RSSMState) -> dict[str, Any]:
        """Autonomous B/Actor control from ``state`` (fused with obs_t) in ``env``."""
        goal = env.goal_for_hidden_label()
        optimal = len(_plan_actions(env, goal))
        actions, wrong, collisions = [], 0, 0
        reward, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            action = int(self.policy.act(self.rssm_b.features(state)).item())
            planned = _shortest_action(env, env.hidden_label)
            before = (tuple(env.agent_pos), env.agent_dir)
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            wrong += int(action != planned)
            collisions += int(action == 2 and (tuple(env.agent_pos), env.agent_dir) == before)
            if not (terminated or truncated):  # obs[t+1] is fused once, after action[t]
                state = self.rssm_b.observe_step(
                    state, torch.tensor([action], dtype=torch.long), _obs_tensor(obs),
                    sample=self.stochastic)
        steps = len(actions)
        success = bool(terminated)
        return {
            "actions": actions, "first_action": actions[0], "steps": steps,
            "success": success, "terminated": bool(terminated), "truncated": bool(truncated),
            "reward": float(reward), "wrong_actions": wrong, "collisions": collisions,
            "optimal_steps": optimal,
            "path_efficiency": (optimal / steps) if success and steps else 0.0,
        }

    def run_sample(self, sample: HandoffSample, method, index: int = 0) -> dict[str, Any]:
        env, obs_t = restore_env(sample)
        try:
            restored_equal = env.snapshot().to_dict() == sample.env_snapshot
            with self._rng(index):
                started = time.perf_counter()
                init = initialize_all([method], sample, stochastic=self.stochastic,
                                      rssm_b=self.rssm_b)[method.name]
                takeover_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                result = self.closed_loop(sample, env, init.state_b)
                loop_ms = (time.perf_counter() - started) * 1000
        finally:
            env.close()
        expert = int(sample.future.expert_actions[0])
        return {
            "episode_id": sample.episode_id, "pair_id": sample.pair_id, "split": sample.split,
            "map_family_id": sample.map_family_id, "episode_role": sample.episode_role,
            "hidden_label": sample.hidden_label, "method": method.name, "t": sample.t,
            "snapshot_sha": digest(sample.env_snapshot), "obs_t_sha": observation_hash(obs_t),
            "snapshot_restored_equal": bool(restored_equal),
            "expert_first_action": expert,
            "first_branch_correct": result["first_action"] == expert,
            **result, "payload_bytes": payload_bytes(method, sample.t),
            "b_observe_steps_at_takeover": len(init.b_fused_frames),
            "takeover_ms": takeover_ms, "closed_loop_ms": loop_ms,
        }

    def run(self, samples: list[HandoffSample]) -> list[dict[str, Any]]:
        return [self.run_sample(s, m, i) for i, s in enumerate(samples) for m in self.methods]


# --------------------------------------------------------------------------- tables


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(rows: list[dict[str, Any]], scope: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    order: dict[tuple, int] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups[key].append(row)
        order.setdefault(key, len(order))
    out = []
    for key in sorted(groups, key=lambda k: (k[:-1], order[k])):
        g = groups[key]
        successes = [r for r in g if r["success"]]
        out.append({
            "scope": scope, **{k: v for k, v in zip(keys, key, strict=True)},
            "n": len(g),
            "success_rate": _mean([r["success"] for r in g]),
            "first_branch_acc": _mean([r["first_branch_correct"] for r in g]),
            "mean_wrong_actions": _mean([r["wrong_actions"] for r in g]),
            "mean_collisions": _mean([r["collisions"] for r in g]),
            "timeout_rate": _mean([r["truncated"] for r in g]),
            "mean_steps": _mean([r["steps"] for r in g]),
            "path_efficiency_all": _mean([r["path_efficiency"] for r in g]),
            "path_efficiency_success": _mean([r["path_efficiency"] for r in successes]),
            "payload_bytes": _mean([r["payload_bytes"] for r in g]),
            "takeover_ms": _mean([r["takeover_ms"] for r in g]),
        })
    return out


def main_tables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per map family, per split and overall rows, each split out by method."""
    return (summarize(rows, "family", ("split", "map_family_id", "method"))
            + summarize(rows, "split", ("split", "method"))
            + summarize(rows, "all", ("method",)))


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def strip_timing(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in TIMING_KEYS} for r in rows]


# --------------------------------------------------------------------------- History Swap


def swap_pairs(samples: list[HandoffSample]) -> list[tuple[HandoffSample, HandoffSample]]:
    """Ordered (own, donor): equal obs_t, different hidden label/early history."""
    by_pair: dict[str, list[HandoffSample]] = defaultdict(list)
    for s in samples:
        if s.episode_role == "expert":
            by_pair[s.pair_id].append(s)
    pairs = []
    for pair_id in sorted(by_pair):
        ends = by_pair[pair_id]
        if len(ends) != 2 or ends[0].hidden_label == ends[1].hidden_label:
            continue
        a, b = ends
        if not (torch.equal(a.history.images[-1], b.history.images[-1])
                and int(a.history.directions[-1]) == int(b.history.directions[-1])):
            raise ValueError(f"pair {pair_id} endpoints do not share obs_t")
        pairs += [(a, b), (b, a)]
    return pairs


def _control_donor(own: HandoffSample, samples: list[HandoffSample]) -> HandoffSample | None:
    """Another pair's endpoint with the same hidden label and family (same cue, other history)."""
    candidates = sorted((s for s in samples if s.episode_role == "expert"
                         and s.pair_id != own.pair_id and s.hidden_label == own.hidden_label
                         and s.map_family_id == own.map_family_id), key=lambda s: s.episode_id)
    return candidates[0] if candidates else None


def run_history_swap(runner: HandoffRunner, cmb: CMBInit, samples: list[HandoffSample]
                     ) -> list[dict[str, Any]]:
    """Swap only the CMB message between endpoints that share ``obs_t``.

    Each condition restores the *own* endpoint's snapshot and passes the *own*
    ``obs_t`` to the Receiver; only the message differs: ``own`` (control), ``swap``
    (message from the paired endpoint with the other early cue), ``zero`` (all-zero
    payload) and ``same_cue`` (message from another pair with the same cue).
    """
    messages: dict[str, Any] = {}

    def message_of(s: HandoffSample):
        if s.episode_id not in messages:
            with torch.no_grad():
                messages[s.episode_id] = cmb.bridge.send(cmb.rssm_a.features(s.state_a))
        return messages[s.episode_id]

    def indices(msg) -> list[int]:
        return [int(i) for i in msg.indices[0]]

    rows = []
    for own, donor in swap_pairs(samples):
        own_msg = message_of(own)
        conditions = [("own", own, own), ("swap", own, donor), ("zero", own, None)]
        control = _control_donor(own, samples)
        if control is not None:
            conditions.append(("same_cue", own, control))
        for condition, base, source in conditions:
            if source is None:  # zeroed CMB: an all-zero payload, no category selected
                msg = CMBMessage(torch.zeros_like(own_msg.logits), torch.zeros_like(own_msg.onehot))
            else:
                msg = message_of(source)
            env, obs_t = restore_env(base)
            snapshot_equal = env.snapshot().to_dict() == base.env_snapshot
            try:
                with torch.no_grad():
                    state = cmb.bridge.receive(msg, base.history.obs_t, cmb.rssm_b)
                result = runner.closed_loop(base, env, state)
            finally:
                env.close()
            rows.append({
                "pair_id": own.pair_id, "split": own.split, "map_family_id": own.map_family_id,
                "own_episode_id": own.episode_id,
                "donor_episode_id": None if source is None else source.episode_id,
                "own_label": own.hidden_label,
                "donor_label": None if source is None else source.hidden_label,
                "condition": condition, "message": None if source is None else indices(msg),
                "message_differs_from_own": True if source is None else indices(msg) != indices(own_msg),
                "snapshot_sha": digest(base.env_snapshot), "obs_t_sha": observation_hash(obs_t),
                "own_expert_action": int(own.future.expert_actions[0]),
                "donor_expert_action": (None if source is None
                                        else int(source.future.expert_actions[0])),
                "first_action": result["first_action"], "success": result["success"],
                "steps": result["steps"], "wrong_actions": result["wrong_actions"],
                "collisions": result["collisions"], "path_efficiency": result["path_efficiency"],
                "snapshot_equal": snapshot_equal,
            })
    own_first = {r["own_episode_id"]: r["first_action"] for r in rows if r["condition"] == "own"}
    for r in rows:
        r["flipped_vs_own"] = r["first_action"] != own_first[r["own_episode_id"]]
        r["follows_message_cue"] = r["first_action"] == r["donor_expert_action"]
    return rows


def summarize_swap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["map_family_id"], r["condition"])].append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        out.append({
            "split": key[0], "map_family_id": key[1], "condition": key[2], "n": len(g),
            "own_branch_correct": _mean([r["first_action"] == r["own_expert_action"] for r in g]),
            "flipped_vs_own": _mean([r["flipped_vs_own"] for r in g]),
            "follows_message_cue": _mean([r["follows_message_cue"] for r in g]),
            "success_rate": _mean([r["success"] for r in g]),
            "path_efficiency": _mean([r["path_efficiency"] for r in g]),
            "message_differs_rate": _mean([r["message_differs_from_own"] for r in g]),
        })
    return out
