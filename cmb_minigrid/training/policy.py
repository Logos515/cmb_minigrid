"""Behavior-cloning training and closed-loop evaluation for Step 6."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cmb_minigrid.data.dataset import read_manifest, validate_split_manifest
from cmb_minigrid.data.trajectory import TrajectoryRecord, read_jsonl
from cmb_minigrid.environments.hidden_history import EnvSnapshot, HiddenHistoryNavEnv
from cmb_minigrid.models import RSSM
from cmb_minigrid.models.policy import ActorPolicy
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.training.rssm import load_rssm, model_batch


@dataclass(frozen=True)
class PolicyTrainingConfig:
    max_steps: int = 500
    batch_size: int = 256
    learning_rate: float = 0.001
    val_every: int = 50
    patience: int = 5
    min_delta: float = 0.001
    hidden_dim: int = 128
    num_threads: int = 1
    class_weighting: bool | str = True

    def __post_init__(self):
        for name in ("max_steps", "batch_size", "val_every", "patience", "hidden_dim", "num_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.class_weighting, (bool, str)):
            raise TypeError("class_weighting must be boolean or inverse/sqrt/none")
        if isinstance(self.class_weighting, str) and self.class_weighting not in {
            "inverse", "sqrt", "none",
        }:
            raise ValueError("class_weighting must be inverse, sqrt, or none")


def fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _policy_examples(model: RSSM, records: list[TrajectoryRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract detached full-history posterior features and expert labels."""

    model.eval().requires_grad_(False)
    features, labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for record in records:
            batch = model_batch([record], device)
            sequence = model.observe(
                batch["agent_obs"], batch["actions"],
                observation_mask=batch["observation_mask"], sample=False,
            )
            # The policy dataset is a detached cache. The RSSM is never part
            # of the Actor optimizer and receives no backward pass.
            features.append(sequence.features[0, :record.horizon].detach().cpu())
            labels.append(torch.tensor(record.expert_actions, dtype=torch.long))
    if not features:
        raise ValueError("policy split has no expert records")
    return torch.cat(features), torch.cat(labels)


def _class_weighting_mode(value: bool | str) -> str:
    if value is True:
        return "inverse"
    if value is False:
        return "none"
    return value


def _class_weights(counts: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones_like(counts)
    if mode == "inverse":
        raw = 1.0 / counts
    elif mode == "sqrt":
        raw = 1.0 / counts.sqrt()
    else:
        raise ValueError(f"unknown class weighting mode: {mode}")
    return raw / raw.mean()


def _metrics(policy: ActorPolicy, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    policy.eval()
    with torch.no_grad():
        logits = policy(features)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(-1) == labels).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item(), "n": len(labels)}


def _save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> tuple[ActorPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported policy checkpoint schema")
    config = checkpoint["policy_config"]
    policy = ActorPolicy(config["feature_dim"], config["hidden_dim"]).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, checkpoint


def train_policy(
    *,
    rssm_checkpoint: str | Path,
    input_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str = "cpu",
    mode: str = "formal",
    training: PolicyTrainingConfig | None = None,
) -> dict[str, Any]:
    settings = training or PolicyTrainingConfig()
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    set_seed(seed, deterministic=True)
    torch.set_num_threads(settings.num_threads)
    model, rssm_meta = load_rssm(rssm_checkpoint, device)
    model.requires_grad_(False)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("RSSM must be frozen before Actor training")
    records = read_jsonl(input_path)
    manifest = read_manifest(manifest_path)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise ValueError("invalid policy dataset: " + "; ".join(errors))
    train_records = [r for r in records if r.episode_role == "expert"
                     and manifest.assignment[r.episode_id] == "train"]
    val_records = [r for r in records if r.episode_role == "expert"
                   and manifest.assignment[r.episode_id] == "val"]
    train_x, train_y = _policy_examples(model, train_records)
    val_x, val_y = _policy_examples(model, val_records)
    class_counts = torch.bincount(train_y, minlength=ActorPolicy.action_dim).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"training expert labels omit an action class: {class_counts.tolist()}")
    weighting_mode = _class_weighting_mode(settings.class_weighting)
    class_weights = _class_weights(class_counts, weighting_mode).to(device)
    policy = ActorPolicy(train_x.shape[-1], settings.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    run_dir = Path(output_dir)
    if run_dir.exists():
        raise FileExistsError(f"policy run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    resolved = {"seed": seed, "device": str(device), "mode": mode,
                "rssm_checkpoint": str(Path(rssm_checkpoint).resolve()),
                "rssm_checkpoint_sha256": fingerprint(rssm_checkpoint),
                "rssm_checkpoint_config": rssm_meta.get("config", {}).get("rssm", {}),
                "input": str(Path(input_path).resolve()),
                "input_sha256": fingerprint(input_path),
                "manifest": str(Path(manifest_path).resolve()),
                "manifest_sha256": fingerprint(manifest_path),
                "training": asdict(settings),
                "policy": {"feature_dim": train_x.shape[-1], "hidden_dim": settings.hidden_dim}}
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    max_steps = 100 if mode == "smoke" else settings.max_steps
    best_val = float("inf")
    best_step = 0
    bad = 0
    metrics_path = run_dir / "metrics.jsonl"
    rng = torch.Generator().manual_seed(seed + 1000)

    def payload(step: int) -> dict[str, Any]:
        return {"schema_version": 1, "config": resolved, "policy_config": resolved["policy"],
                "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val": best_val, "best_step": best_step,
                "bad_validations": bad}

    for step in range(1, max_steps + 1):
        policy.train()
        indices = torch.randint(len(train_y), (min(settings.batch_size, len(train_y)),), generator=rng)
        logits = policy(train_x[indices].to(device))
        loss = F.cross_entropy(logits, train_y[indices].to(device), weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 100.0, error_if_nonfinite=True)
        optimizer.step()
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError(f"non-finite policy loss or gradient at step {step}")
        if not any(parameter.grad is not None and torch.any(parameter.grad != 0)
                   for parameter in policy.parameters()):
            raise RuntimeError(f"zero policy gradient at step {step}")
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "train", "step": step, "loss": loss.item(),
                                     "gradient_norm": norm.item()}) + "\n")
        if step % settings.val_every == 0 or step == max_steps:
            train_metrics = _metrics(policy, train_x.to(device), train_y.to(device))
            validation = _metrics(policy, val_x.to(device), val_y.to(device))
            if validation["loss"] < best_val - settings.min_delta:
                best_val, best_step, bad = validation["loss"], step, 0
                _save(run_dir / "best.pt", payload(step))
            else:
                bad += 1
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "validation", "step": step,
                                         "train_metrics": train_metrics,
                                         "metrics": validation,
                                         "bad_validations": bad}) + "\n")
            print(f"step={step} batch_loss={loss.item():.6f} "
                  f"train_loss={train_metrics['loss']:.6f} "
                  f"train_acc={train_metrics['accuracy']:.3f} "
                  f"val_loss={validation['loss']:.6f} "
                  f"val_acc={validation['accuracy']:.3f}", flush=True)
        _save(run_dir / "last.pt", payload(step))
        if mode == "formal" and bad >= settings.patience:
            break
    final_train = _metrics(policy, train_x.to(device), train_y.to(device))
    final = _metrics(policy, val_x.to(device), val_y.to(device))
    if not (run_dir / "best.pt").exists():
        _save(run_dir / "best.pt", payload(step))
    report = {"status": "loss_decreased" if best_step > 0 else "training_failed_no_improvement",
              "steps": step, "best_step": best_step, "best_val": best_val,
              "final_train": final_train, "final_val": final,
              "train_examples": len(train_y), "val_examples": len(val_y),
              "best_checkpoint": str(run_dir / "best.pt"), "last_checkpoint": str(run_dir / "last.pt"),
              "mode": mode, "seed": seed, "class_counts": class_counts.long().tolist(),
              "class_weights": class_weights.cpu().tolist(),
              "class_weighting": settings.class_weighting,
              "weighting_mode": weighting_mode}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _obs_tensor(obs, device):
    return {"image": torch.from_numpy(obs.image.copy()).unsqueeze(0).to(device),
            "direction": torch.tensor([obs.direction], dtype=torch.long, device=device),
            "task_id": [obs.task_id]}


def _first_action_error(predicted: list[int], expert: list[int], handoff_t: int) -> dict[str, Any]:
    """Return the first mismatch against the reference expert action sequence."""

    for t, (actual, target) in enumerate(zip(predicted, expert, strict=False)):
        if actual != target:
            return {
                "first_error_t": t,
                "first_error_predicted_action": int(actual),
                "first_error_expert_action": int(target),
                "first_error_phase": "before_handoff" if t < handoff_t else "at_or_after_handoff",
                "first_error_before_handoff": t < handoff_t,
                "steps_before_first_error": t,
            }
    return {
        "first_error_t": None,
        "first_error_predicted_action": None,
        "first_error_expert_action": None,
        "first_error_phase": "none",
        "first_error_before_handoff": False,
        "steps_before_first_error": len(predicted),
    }


@torch.no_grad()
def run_closed_loop(record: TrajectoryRecord, rssm: RSSM, policy: ActorPolicy,
                    *, mode: str, device: str | torch.device = "cpu") -> dict[str, Any]:
    if mode not in {"a_full_history", "oracle_b_full_history", "reset_b"}:
        raise ValueError("unknown closed-loop mode")
    snapshot = EnvSnapshot.from_dict(record.privileged_state[0])
    env = HiddenHistoryNavEnv(hidden_label=snapshot.hidden_label,
                              map_family_id=record.map_family_id)
    rssm.eval()
    policy.eval()
    try:
        obs, _ = env.reset(seed=record.seed, options={"hidden_label": snapshot.hidden_label,
                                                        "map_family_id": record.map_family_id})
        state = rssm.initial(1)
        previous_action = None
        predicted = []
        handoff_action = None
        done = cut = False
        while not done and not cut and len(predicted) < env.max_steps:
            current = rssm.initial(1) if mode == "reset_b" else state
            state = rssm.observe_step(current, previous_action if mode != "reset_b" else None,
                                      _obs_tensor(obs, device), sample=False)
            action = int(policy.act(rssm.features(state), ).item())
            predicted.append(action)
            if len(predicted) - 1 == record.handoff_t:
                handoff_action = action
            obs, _, done, cut, _ = env.step(action)
            previous_action = torch.tensor([action], dtype=torch.long, device=device)
        expert_handoff = int(record.expert_actions[record.handoff_t])
        error = _first_action_error(predicted, list(record.expert_actions), record.handoff_t)
        return {"episode_id": record.episode_id, "pair_id": record.pair_id,
                "mode": mode, "predicted_actions": predicted,
                "expert_actions_observed": list(record.expert_actions[:len(predicted)]),
                **error,
                "expert_handoff_action": expert_handoff,
                "predicted_handoff_action": handoff_action,
                "branch_correct": handoff_action == expert_handoff,
                "success": bool(done and not cut), "terminated": bool(done),
                "truncated": bool(cut), "steps": len(predicted)}
    finally:
        env.close()


@torch.no_grad()
def run_offline_replay(
    record: TrajectoryRecord,
    rssm: RSSM,
    policy: ActorPolicy,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Score an Actor while replaying a recorded expert trajectory.

    The environment is advanced only with ``record.actions``. Actor actions
    are predictions for comparison and are never used as environment input.
    RSSM filtering uses the same recorded action history used to construct the
    offline behavior-cloning features.
    """

    snapshot = EnvSnapshot.from_dict(record.privileged_state[0])
    env = HiddenHistoryNavEnv(hidden_label=snapshot.hidden_label,
                              map_family_id=record.map_family_id)
    rssm.eval()
    policy.eval()
    predictions: list[int] = []
    targets = [int(action) for action in record.expert_actions]
    replay_actions = [int(action) for action in record.actions]
    step_rows: list[dict[str, Any]] = []
    try:
        obs, _ = env.reset(seed=record.seed, options={
            "hidden_label": snapshot.hidden_label,
            "map_family_id": record.map_family_id,
        })
        state = rssm.initial(1)
        previous_action = None
        for t, target in enumerate(targets):
            if obs != record.agent_obs[t]:
                raise RuntimeError(f"offline replay observation mismatch at t={t}")
            state = rssm.observe_step(
                state,
                previous_action,
                _obs_tensor(obs, device),
                sample=False,
            )
            predicted = int(policy.act(rssm.features(state)).item())
            predictions.append(predicted)
            step_rows.append({
                "t": t,
                "predicted_action": predicted,
                "expert_action": target,
                "replay_action": replay_actions[t],
                "correct": predicted == target,
            })
            obs, _, terminated, truncated, _ = env.step(replay_actions[t])
            if obs != record.agent_obs[t + 1]:
                raise RuntimeError(f"offline replay observation mismatch at t={t + 1}")
            previous_action = torch.tensor([replay_actions[t]], dtype=torch.long, device=device)
            if terminated or truncated:
                if t + 1 != record.horizon:
                    raise RuntimeError(f"offline replay ended early at t={t}")
                break
        if len(predictions) != record.horizon:
            raise RuntimeError("offline replay did not cover the full trajectory")
        correct = [row["correct"] for row in step_rows]
        handoff = step_rows[record.handoff_t]
        error = _first_action_error(predictions, targets, record.handoff_t)
        return {
            "episode_id": record.episode_id,
            "pair_id": record.pair_id,
            "mode": "offline_expert_replay",
            "predicted_actions": predictions,
            "expert_actions": targets,
            "replay_actions": replay_actions,
            "step_records": step_rows,
            "action_accuracy": float(np.mean(correct)),
            "correct_steps": int(sum(correct)),
            "steps": len(correct),
            "handoff_t": record.handoff_t,
            "predicted_handoff_action": handoff["predicted_action"],
            "expert_handoff_action": handoff["expert_action"],
            "handoff_correct": bool(handoff["correct"]),
            **error,
            "actor_actions_replayed": False,
        }
    finally:
        env.close()


def evaluate_offline_replays(
    *,
    rssm_checkpoint: str | Path,
    policy_checkpoint: str | Path,
    input_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate stepwise Actor predictions without changing replay state."""

    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train, val, test, or all")
    records = read_jsonl(input_path)
    manifest = read_manifest(manifest_path)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise ValueError("invalid evaluation dataset: " + "; ".join(errors))
    selected = [record for record in records if record.episode_role == "expert"
                and (split == "all" or manifest.assignment[record.episode_id] == split)]
    if not selected:
        raise ValueError(f"split {split!r} has no expert records")
    rssm, _ = load_rssm(rssm_checkpoint, device)
    policy, _ = load_policy(policy_checkpoint, device)
    rows = [run_offline_replay(record, rssm, policy, device=device) for record in selected]
    total_steps = sum(row["steps"] for row in rows)
    total_correct = sum(row["correct_steps"] for row in rows)
    report = {
        "split": split,
        "episodes": len(rows),
        "steps": total_steps,
        "action_accuracy": total_correct / total_steps,
        "episode_mean_accuracy": float(np.mean([row["action_accuracy"] for row in rows])),
        "handoff_accuracy": float(np.mean([row["handoff_correct"] for row in rows])),
        "first_error_summary": _first_error_summary(rows),
        "actor_actions_replayed": False,
        "trajectories": rows,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "offline_replay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _first_error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate first-error locations without treating missing errors as t=0."""

    errors = [row for row in rows if row["first_error_t"] is not None]
    return {
        "episodes_with_error": len(errors),
        "episodes_without_error": len(rows) - len(errors),
        "first_error_before_handoff": sum(row["first_error_before_handoff"] for row in errors),
        "first_error_at_or_after_handoff": sum(
            not row["first_error_before_handoff"] for row in errors
        ),
        "mean_first_error_t": (
            float(np.mean([row["first_error_t"] for row in errors])) if errors else None
        ),
        "min_first_error_t": min((row["first_error_t"] for row in errors), default=None),
        "max_first_error_t": max((row["first_error_t"] for row in errors), default=None),
    }


def evaluate_closed_loops(*, checkpoint_a, checkpoint_b, policy_a, policy_b,
                          input_path, manifest_path, output_dir, device="cpu"):
    records = read_jsonl(input_path)
    manifest = read_manifest(manifest_path)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise ValueError("invalid evaluation dataset: " + "; ".join(errors))
    test = [r for r in records if r.episode_role == "expert"
            and manifest.assignment[r.episode_id] == "test"]
    if not test:
        raise ValueError("test split has no expert records")
    rssm_a, _ = load_rssm(checkpoint_a, device)
    rssm_b, _ = load_rssm(checkpoint_b, device)
    actor_a, _ = load_policy(policy_a, device)
    actor_b, _ = load_policy(policy_b, device)
    modes = {"a_full_history": (rssm_a, actor_a),
             "oracle_b_full_history": (rssm_b, actor_b),
             "reset_b": (rssm_b, actor_b)}
    report = {"test_episodes": len(test), "modes": {}, "trajectories": []}
    for mode, (rssm, actor) in modes.items():
        rows = [run_closed_loop(record, rssm, actor, mode=mode, device=device) for record in test]
        report["modes"][mode] = {
            "branch_accuracy": float(np.mean([row["branch_correct"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "episodes": len(rows),
            "first_error_summary": _first_error_summary(rows),
        }
        report["trajectories"].extend(rows)
    oracle = report["modes"]["oracle_b_full_history"]
    reset = report["modes"]["reset_b"]
    report["oracle_reset_gap"] = {"branch_accuracy": oracle["branch_accuracy"] - reset["branch_accuracy"],
                                  "success_rate": oracle["success_rate"] - reset["success_rate"]}
    report["bridge_gate"] = ("proceed_only_if_authorized" if
                              max(report["oracle_reset_gap"].values()) >= 0.1 else "hold_diagnosis")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "closed_loop_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
