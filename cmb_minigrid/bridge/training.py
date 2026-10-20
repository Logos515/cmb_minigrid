"""CMB training (two stages), evaluation against baselines, and overhead accounting.

Frozen: RSSM-A, RSSM-B and Actor-B (``requires_grad=False``). The *student*
branch (CMB -> state_B_hat -> Actor-B / imagine_step / decode) runs with
gradients so they reach the Sender and Receiver. The *teacher* branch, built
from the Oracle ``state_B_star``, is computed under ``torch.no_grad``.

Stage 1: state auxiliary loss to ``state_B_star`` + KL(Actor-B(teacher) ||
Actor-B(student)). Stage 2 adds multi-step distillation: both rollouts apply the
same recorded ``action[t:t+K]`` through ``imagine_step`` and the student's
prior state is pulled to the teacher's (K in ``horizons``). Real future
observations are used only as an evaluation check (image CE of decoded imagined
states) and never as a training target.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from cmb_minigrid.bridge.cmb import CMBBridge, CMBConfig, save_cmb
from cmb_minigrid.contracts import RSSMState
from cmb_minigrid.handoff.dataset import HandoffSample, file_sha256, load_handoff_dataset
from cmb_minigrid.handoff.initializers import (
    MapInit,
    OracleInit,
    ReplayInit,
    ResetInit,
    initialize_all,
    load_state_map,
    validate_b_state,
)
from cmb_minigrid.models import RSSM
from cmb_minigrid.models.policy import ActorPolicy
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.training.policy import load_policy
from cmb_minigrid.training.rssm import load_rssm

HORIZONS = (5, 10)


# --------------------------------------------------------------------------- batching


@dataclass
class HandoffBatch:
    features_a: Tensor  # [N,Fa]
    obs: dict[str, Any]  # obs_t: image [N,7,7,3], direction [N], task_id
    b_star: RSSMState
    actions: Tensor  # [N,K] action[t:t+K], padded with 0
    step_mask: Tensor  # [N,K] bool, true where action[t+k-1] and obs[t+k] exist
    future_images: Tensor  # [N,K,7,7,3] = obs[t+1:t+K+1], padded with 0
    expert_action: Tensor  # [N] expert action at the handoff

    def __len__(self) -> int:
        return self.features_a.shape[0]

    def index(self, idx: Tensor) -> HandoffBatch:
        return HandoffBatch(
            self.features_a[idx],
            {"image": self.obs["image"][idx], "direction": self.obs["direction"][idx],
             "task_id": [self.obs["task_id"][i] for i in idx.tolist()]},
            RSSMState(self.b_star.deter[idx], self.b_star.stoch[idx], self.b_star.stoch_logits[idx]),
            self.actions[idx], self.step_mask[idx], self.future_images[idx], self.expert_action[idx])


def stack_states(states: list[RSSMState]) -> RSSMState:
    return RSSMState(torch.cat([s.deter for s in states]), torch.cat([s.stoch for s in states]),
                     torch.cat([s.stoch_logits for s in states]))


def make_batch(samples: list[HandoffSample], rssm_a: RSSM, device, kmax: int = max(HORIZONS)
               ) -> HandoffBatch:
    n = len(samples)
    actions = torch.zeros(n, kmax, dtype=torch.long)
    mask = torch.zeros(n, kmax, dtype=torch.bool)
    images = torch.zeros(n, kmax, 7, 7, 3, dtype=torch.uint8)
    for i, s in enumerate(samples):
        k = min(kmax, len(s.future.actions))
        actions[i, :k] = s.future.actions[:k]
        mask[i, :k] = True
        images[i, :k] = s.future.images[:k]
    with torch.no_grad():
        features = torch.cat([rssm_a.features(RSSMState(
            s.state_a.deter.to(device), s.state_a.stoch.to(device))) for s in samples])
    return HandoffBatch(
        features_a=features,
        obs={"image": torch.stack([s.history.images[-1] for s in samples]).to(device),
             "direction": torch.stack([s.history.directions[-1] for s in samples]).to(device),
             "task_id": [s.history.task_id for s in samples]},
        b_star=stack_states([RSSMState(s.state_b_star.deter.to(device), s.state_b_star.stoch.to(device),
                                       s.state_b_star.stoch_logits.to(device)) for s in samples]),
        actions=actions.to(device), step_mask=mask.to(device), future_images=images.to(device),
        expert_action=torch.stack([s.future.expert_actions[0] for s in samples]).to(device))


# --------------------------------------------------------------------------- losses


def categorical_kl_sum(target_logits: Tensor, model_logits: Tensor) -> Tensor:
    """KL(target || model) summed over the trailing [G,C] axes."""
    p = target_logits.log_softmax(-1)
    q = model_logits.log_softmax(-1)
    return (p.exp() * (p - q)).sum((-1, -2))


def rollout(rssm_b: RSSM, state: RSSMState, batch: HandoffBatch, steps: int) -> list[RSSMState]:
    """imagine_step with the recorded actions; never reads an observation."""
    out = []
    for k in range(steps):
        state = rssm_b.imagine_step(state, batch.actions[:, k], sample=False)
        out.append(state)
    return out


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_losses(bridge: CMBBridge, batch: HandoffBatch, rssm_b: RSSM, policy: ActorPolicy,
                   *, stage: int, horizons=HORIZONS, state_weight: float = 1.0,
                   action_weight: float = 1.0, multistep_weight: float = 1.0
                   ) -> dict[str, Tensor]:
    student, _ = bridge(batch.features_a, batch.obs, rssm_b)
    with torch.no_grad():  # Oracle teacher branch only
        teacher = RSSMState(batch.b_star.deter.detach(), batch.b_star.stoch.detach(),
                            batch.b_star.stoch_logits.detach())
        teacher_policy = policy(rssm_b.features(teacher)).log_softmax(-1)
    student_policy = policy(rssm_b.features(student)).log_softmax(-1)
    state_aux = (F.mse_loss(student.deter, teacher.deter)
                 + categorical_kl_sum(teacher.stoch_logits, student.stoch_logits).mean())
    action_kl = (teacher_policy.exp() * (teacher_policy - student_policy)).sum(-1).mean()
    total = state_weight * state_aux + action_weight * action_kl
    losses = {"state_aux": state_aux, "action_kl": action_kl}
    if stage == 2:
        kmax = max(horizons)
        with torch.no_grad():
            teacher_states = rollout(rssm_b, teacher, batch, kmax)
        student_states = rollout(rssm_b, student, batch, kmax)
        per_step = torch.stack([
            ((s.deter - t.deter) ** 2).mean(-1) + categorical_kl_sum(t.stoch_logits, s.stoch_logits)
            for s, t in zip(student_states, teacher_states, strict=True)], 1)  # [N,K]
        mask = batch.step_mask.float()
        multistep = torch.stack([masked_mean(per_step[:, :k], mask[:, :k]) for k in horizons]).mean()
        losses["multistep"] = multistep
        total = total + multistep_weight * multistep
    losses["total"] = total
    return losses


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_state(rssm_b: RSSM, policy: ActorPolicy, batch: HandoffBatch, state: RSSMState,
                   horizons=HORIZONS) -> dict[str, float]:
    """Metrics of one candidate ``state_B[t]`` per method; all from the same batch."""
    teacher = batch.b_star
    p = policy(rssm_b.features(teacher)).log_softmax(-1)
    q = policy(rssm_b.features(state)).log_softmax(-1)
    kmax = max(horizons)
    mask = batch.step_mask.float()
    teacher_states = rollout(rssm_b, teacher, batch, kmax)
    states = rollout(rssm_b, state, batch, kmax)
    ce = []
    for k, s in enumerate(states):
        logits = rssm_b.decode(rssm_b.features(s))["image_logits"]
        target = batch.future_images[:, k]
        ce.append(torch.stack([
            F.cross_entropy(logit.flatten(1, 2).transpose(1, 2), target[..., c].long().flatten(1),
                            reduction="none").mean(-1)
            for c, logit in enumerate(logits)]).mean(0))
    ce = torch.stack(ce, 1)
    drift = torch.stack([(s.deter - t.deter).norm(dim=-1)
                         for s, t in zip(states, teacher_states, strict=True)], 1)
    out = {
        "n": len(batch),
        "deter_l2": (state.deter - teacher.deter).norm(dim=-1).mean().item(),
        "stoch_group_agreement": (state.stoch.argmax(-1) == teacher.stoch.argmax(-1))
        .float().mean().item(),
        "action_kl_to_oracle": (p.exp() * (p - q)).sum(-1).mean().item(),
        "action_agree_with_oracle": (q.argmax(-1) == p.argmax(-1)).float().mean().item(),
        "action_acc_vs_expert": (q.argmax(-1) == batch.expert_action).float().mean().item(),
    }
    for k in horizons:
        out[f"future_obs_ce_{k}"] = masked_mean(ce[:, :k], mask[:, :k]).item()
        out[f"future_deter_drift_{k}"] = masked_mean(drift[:, :k], mask[:, :k]).item()
    return out


@torch.no_grad()
def evaluate_methods(
    samples: list[HandoffSample], rssm_a: RSSM, rssm_b: RSSM, policy: ActorPolicy,
    bridge: CMBBridge | None, *, linear_map=None, mlp_map=None, device="cpu",
) -> dict[str, dict[str, float]]:
    """Reset, Replay-4/8, Linear, MLP, CMB and Oracle-B on identical samples/snapshots."""
    batch = make_batch(samples, rssm_a, device)
    methods = [ResetInit(rssm_b), ReplayInit(rssm_b, 4), ReplayInit(rssm_b, 8)]
    if linear_map is not None:
        methods.append(MapInit("linear", linear_map, rssm_a, rssm_b))
    if mlp_map is not None:
        methods.append(MapInit("mlp", mlp_map, rssm_a, rssm_b))
    methods.append(OracleInit(rssm_b))
    per_method: dict[str, list[RSSMState]] = {m.name: [] for m in methods}
    for s in samples:
        for name, init in initialize_all(methods, s, rssm_b=rssm_b).items():
            per_method[name].append(init.state_b)
    results = {name: evaluate_state(rssm_b, policy, batch, stack_states(states))
               for name, states in per_method.items()}
    if bridge is not None:
        was_training = bridge.training
        bridge.eval()
        state, _ = bridge(batch.features_a, batch.obs, rssm_b)
        if was_training:
            bridge.train()
        validate_b_state(rssm_b, state)
        results["cmb"] = evaluate_state(rssm_b, policy, batch, state)
    return results


_BETTER = {"action_kl_to_oracle": "lower", "action_acc_vs_expert": "higher",
           "action_agree_with_oracle": "higher", "future_obs_ce_5": "lower",
           "future_obs_ce_10": "lower", "future_deter_drift_5": "lower",
           "future_deter_drift_10": "lower"}


def compare_to_reset(results: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    """Per metric: CMB, Reset and whether CMB is not worse (ties count as not worse)."""
    out = {}
    for metric, direction in _BETTER.items():
        cmb, reset = results["cmb"][metric], results["reset"][metric]
        out[metric] = {"cmb": cmb, "reset": reset, "better_is": direction,
                       "cmb_not_worse": bool(cmb <= reset if direction == "lower" else cmb >= reset)}
    return out


# --------------------------------------------------------------------------- overhead


def overhead_table(bridge: CMBBridge, linear_map=None, mlp_map=None) -> dict[str, Any]:
    c = bridge.config
    counts = bridge.parameter_counts()
    bits = c.message_bits
    table = {"cmb": {
        **counts, "payload_bits": bits, "payload_bytes_packed": math.ceil(bits / 8),
        "payload_bytes_one_hot_float32": c.tokens * c.token_classes * 4,
        "note": f"{c.tokens} tokens x {c.token_classes} classes, discrete"}}
    for name, m in (("linear", linear_map), ("mlp", mlp_map)):
        if m is None:
            continue
        width = m.bottleneck if m.bottleneck is not None else m.in_dim
        table[name] = {"total": m.num_parameters, "payload_floats": width,
                       "payload_bytes_float32": width * 4,
                       "note": "continuous float payload"
                               + (" (bottleneck)" if m.bottleneck is not None else " (full A feature)")}
    return table


# --------------------------------------------------------------------------- training


def _norm(module) -> float:
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return math.sqrt(sum(g.pow(2).sum().item() for g in grads)) if grads else 0.0


def _freeze(module) -> None:
    module.eval().requires_grad_(False)


def train_cmb(
    *, handoff_path, checkpoint_a, checkpoint_b, policy_b, output_dir, seed: int = 42,
    config: CMBConfig | None = None, stage1_steps: int = 300, stage2_steps: int = 300,
    batch_size: int = 32, lr: float = 1e-3, val_every: int = 25, patience: int = 8,
    linear_map_path=None, mlp_map_path=None, device="cpu", horizons=HORIZONS,
    state_weight: float = 1.0, action_weight: float = 1.0, multistep_weight: float = 1.0,
) -> dict[str, Any]:
    set_seed(seed, deterministic=True)
    torch.set_num_threads(1)
    rssm_a, _ = load_rssm(checkpoint_a, device)
    rssm_b, _ = load_rssm(checkpoint_b, device)
    policy, _ = load_policy(policy_b, device)
    for module in (rssm_a, rssm_b, policy):
        _freeze(module)
    samples, _ = load_handoff_dataset(handoff_path)
    by_split = {k: [s for s in samples if s.split == k] for k in ("train", "val", "test")}
    if not all(by_split.values()):
        raise ValueError("handoff dataset must contain train, val and test samples")
    cfg = config or CMBConfig(a_feature_dim=rssm_a.feature_dim, b_deter=rssm_b.config.deter)
    bridge = CMBBridge(cfg).to(device)
    optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)
    train = make_batch(by_split["train"], rssm_a, device)
    val = make_batch(by_split["val"], rssm_a, device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    meta = {"seed": seed, "handoff_sha256": file_sha256(handoff_path),
            "checkpoint_a_sha256": file_sha256(checkpoint_a),
            "checkpoint_b_sha256": file_sha256(checkpoint_b),
            "policy_b_sha256": file_sha256(policy_b), "horizons": list(horizons),
            "lr": lr, "batch_size": batch_size, "state_weight": state_weight,
            "action_weight": action_weight, "multistep_weight": multistep_weight,
            "stage1_steps": stage1_steps, "stage2_steps": stage2_steps}
    (out / "config.json").write_text(json.dumps({**meta, "cmb": bridge.config.__dict__,
                                                  "stage1_steps": stage1_steps,
                                                  "stage2_steps": stage2_steps}, indent=2) + "\n")
    log = out / "metrics.jsonl"
    generator = torch.Generator().manual_seed(seed + 1)
    stage_reports = {}
    step = 0
    weights = {"state_weight": state_weight, "action_weight": action_weight,
               "multistep_weight": multistep_weight}
    for stage, steps in ((1, stage1_steps), (2, stage2_steps)):
        if steps == 0:  # ablation: stage skipped entirely
            continue
        local = 0
        best, best_step, bad = float("inf"), 0, 0
        best_state = {k: v.clone() for k, v in bridge.state_dict().items()}
        for local in range(1, steps + 1):
            step += 1
            bridge.train()
            idx = torch.randperm(len(train), generator=generator)[:batch_size].to(device)
            losses = compute_losses(bridge, train.index(idx), rssm_b, policy, stage=stage,
                                    horizons=horizons, **weights)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if local == 1:  # frozen components must not accumulate gradients
                for name, module in (("rssm_a", rssm_a), ("rssm_b", rssm_b), ("policy_b", policy)):
                    if any(p.grad is not None for p in module.parameters()):
                        raise RuntimeError(f"{name} received gradients but must stay frozen")
            sender, receiver = _norm(bridge.sender), _norm(bridge.receiver)
            if not (math.isfinite(losses["total"].item()) and sender > 0 and receiver > 0
                    and math.isfinite(sender) and math.isfinite(receiver)):
                raise RuntimeError(
                    f"stage {stage} step {local}: loss/grad invalid "
                    f"(loss={losses['total'].item()}, sender={sender}, receiver={receiver})")
            optimizer.step()
            row = {"event": "train", "stage": stage, "step": step,
                   **{k: v.item() for k, v in losses.items()},
                   "sender_grad_norm": sender, "receiver_grad_norm": receiver}
            with log.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if local % val_every == 0 or local == steps:
                bridge.eval()
                with torch.no_grad():
                    v = compute_losses(bridge, val, rssm_b, policy, stage=stage, horizons=horizons,
                                       **weights)
                value = v["total"].item()
                if value < best - 1e-4:
                    best, best_step, bad = value, step, 0
                    best_state = {k: t.clone() for k, t in bridge.state_dict().items()}
                    save_cmb(bridge, out / f"stage{stage}_best.pt",
                             {"stage": stage, "step": step, "val_total": value, **meta})
                else:
                    bad += 1
                with log.open("a") as handle:
                    handle.write(json.dumps({"event": "val", "stage": stage, "step": step,
                                             **{k: t.item() for k, t in v.items()},
                                             "bad_validations": bad}) + "\n")
                print(f"stage={stage} step={local}/{steps} train={losses['total'].item():.4f} "
                      f"val={value:.4f}", flush=True)
                if bad >= patience:
                    break
        bridge.load_state_dict(best_state)  # stage 2 starts from the best stage-1 weights
        stage_reports[f"stage{stage}"] = {"best_val_total": best, "best_step": best_step,
                                          "steps_run": local}
        extra = {"stage": stage, "step": best_step, "val_total": best, **meta,
                 "optimizer": optimizer.state_dict()}
        save_cmb(bridge, out / "best.pt", extra)  # last executed stage wins
        if stage == 1:
            save_cmb(bridge, out / "stage1_selected.pt", extra)

    linear_map = load_state_map(linear_map_path)[0] if linear_map_path else None
    mlp_map = load_state_map(mlp_map_path)[0] if mlp_map_path else None
    evaluation = {}
    for split in ("val", "test"):
        results = evaluate_methods(by_split[split], rssm_a, rssm_b, policy, bridge,
                                   linear_map=linear_map, mlp_map=mlp_map, device=device)
        evaluation[split] = {"methods": results, "cmb_vs_reset": compare_to_reset(results)}
    report = {"meta": meta, "stages": stage_reports, "evaluation": evaluation,
              "overhead": overhead_table(bridge, linear_map, mlp_map),
              "samples": {k: len(v) for k, v in by_split.items()},
              "checkpoint": str(out / "best.pt")}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
