"""Full-episode RSSM training with resumable sampling and validation stopping."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cmb_minigrid.data.dataset import TrajectoryDataset, collate_trajectories
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.models.rssm import early_key_color, recall_targets
from cmb_minigrid.reproducibility import experiment_id, set_seed
from cmb_minigrid.runtime import select_device


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.001
    grad_clip: float = 100.0
    val_every: int = 50
    patience: int = 5
    min_delta: float = 0.001
    num_threads: int = 1

    def __post_init__(self):
        for name in ("max_steps", "batch_size", "val_every", "patience", "num_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "grad_clip", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "min_delta" and value == 0):
                raise ValueError(f"invalid {name}")


def fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_batch(records, device) -> dict[str, Any]:
    """Explicit allowlist: no expert labels, metadata or privileged state."""
    batch = collate_trajectories(records)
    obs = batch["agent_obs"]
    return {
        "agent_obs": {"image": obs["image"].to(device),
                      "direction": obs["direction"].to(device), "task_id": obs["task_id"]},
        **{name: batch[name].to(device) for name in (
            "actions", "reward", "terminated", "truncated", "observation_mask",
            "transition_mask")},
    }


def total_loss(losses, config: RSSMConfig):
    return (losses["reconstruction"] + config.reward_scale * losses["reward"]
            + config.continue_scale * losses["continue"]
            + config.dynamics_scale * losses["dynamics_kl"]
            + config.representation_scale * losses["representation_kl"]
            + config.recall_scale * losses.get("recall", 0.0)
            + config.visual_recall_scale * losses.get("visual_recall", 0.0))


@torch.no_grad()
def evaluate(model: RSSM, records, batch_size: int) -> dict[str, float]:
    """Weight components by their valid observations/transitions, not batches."""
    if not records:
        raise ValueError("evaluation split is empty")
    model.eval()
    sums, counts = {}, {}
    for start in range(0, len(records), batch_size):
        batch = model_batch(records[start:start + batch_size], next(model.parameters()).device)
        losses = model.loss(batch, sample=False)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise RuntimeError("non-finite validation loss")
        for name, value in losses.items():
            if name == "total":
                continue
            if name == "visual_recall" or name == "visual_recall_accuracy":
                continue
            if name.startswith("visual_recall_t"):
                labels = early_key_color(batch["agent_obs"]["image"][:, 0])
                suffix = name.removeprefix("visual_recall_t").removesuffix("_accuracy")
                t = int(suffix)
                count = int(((labels >= 0) & batch["observation_mask"][:, t]).sum()) \
                    if t < batch["observation_mask"].shape[1] else 0
            elif name in {"recall", "recall_accuracy"}:
                count = int(recall_targets(batch, model.config.recall_t)[1].sum())
            else:
                mask = "transition_mask" if name in {"reward", "continue"} else "observation_mask"
                count = int(batch[mask].sum())
            sums[name] = sums.get(name, 0.0) + value.item() * count
            counts[name] = counts.get(name, 0) + count
    result = {name: value / max(counts[name], 1) for name, value in sums.items()}
    visual_losses = [result[f"visual_recall_t{t}"] for t in model.config.visual_recall_times
                     if f"visual_recall_t{t}" in result]
    visual_accuracies = [result[f"visual_recall_t{t}_accuracy"]
                         for t in model.config.visual_recall_times
                         if f"visual_recall_t{t}_accuracy" in result]
    if visual_losses:
        result["visual_recall"] = sum(visual_losses) / len(visual_losses)
        result["visual_recall_accuracy"] = sum(visual_accuracies) / len(visual_accuracies)
    result["total"] = total_loss(result, model.config)
    return result


class EpisodeSampler:
    """A shuffled epoch order and cursor, saved independently of model RNG."""

    def __init__(self, size: int, seed: int):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order: list[int] = []
        self.cursor = 0

    def next(self, batch_size: int) -> list[int]:
        if self.cursor == len(self.order):
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.cursor = 0
        end = min(self.cursor + batch_size, self.size)
        indices = self.order[self.cursor:end]
        self.cursor = end
        return indices

    def state_dict(self):
        return {"size": self.size, "order": self.order, "cursor": self.cursor,
                "rng": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("resume dataset size changed")
        self.order, self.cursor = state["order"], state["cursor"]
        self.generator.set_state(state["rng"])


def rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), *state[2:]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_rssm(path: str | Path, device="cpu") -> tuple[RSSM, dict[str, Any]]:
    """Load one model solely from its checkpoint; no YAML or peer is needed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported RSSM checkpoint schema")
    model = RSSM(RSSMConfig.from_mapping(checkpoint["config"]["rssm"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _resume_signature(config):
    signature = dict(config)
    signature["rssm"] = asdict(RSSMConfig.from_mapping(signature["rssm"]))
    signature.pop("device", None)
    signature["training"] = dict(signature["training"])
    signature["training"].pop("max_steps")
    return signature


def train_rssm(config: dict[str, Any], output_dir: str | Path, *, resume=None):
    settings = TrainingConfig(**config["training"])
    set_seed(config["seed"], deterministic=config["deterministic"])
    torch.set_num_threads(settings.num_threads)
    device = select_device(config["device"])
    config = config | {"device": str(device)}
    for name in ("input", "manifest"):
        if fingerprint(config[name]) != config[f"{name}_sha256"]:
            raise ValueError(f"{name} changed since configuration resolution")
    train = TrajectoryDataset(config["input"], split="train", manifest_path=config["manifest"])
    val = TrajectoryDataset(config["input"], split="val", manifest_path=config["manifest"])
    if not len(train) or not len(val):
        raise ValueError("Step 5 requires nonempty grouped train and validation splits")
    run_id = experiment_id(config["experiment_name"], config["seed"], config=config)
    run_dir = Path(resume).resolve().parent if resume else Path(output_dir) / run_id
    if not resume and run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}; use --resume or another output-dir")
    model = RSSM(RSSMConfig.from_mapping(config["rssm"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    sampler = EpisodeSampler(len(train), config["seed"])
    step, bad_validations, best_step = 0, 0, 0
    stopped = False
    if resume:
        model, checkpoint = load_rssm(resume, device)
        if _resume_signature(checkpoint["config"]) != _resume_signature(config):
            raise ValueError("resume configuration changed (only max_steps/device may change)")
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.load_state_dict(checkpoint["sampler"])
        step, bad_validations = checkpoint["step"], checkpoint["bad_validations"]
        best, best_step = checkpoint["best_val"], checkpoint["best_step"]
        initial_train, initial_val = checkpoint["initial_train"], checkpoint["initial_val"]
        run_id = checkpoint["experiment_id"]
        stopped = checkpoint["early_stopped"]
        if settings.max_steps <= step:
            raise ValueError("max_steps must exceed the restored step")
        # Append only to the latest run state, so logs and best checkpoint stay consistent.
        latest = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["step"] != latest["step"] or Path(resume).name != "last.pt":
            raise ValueError("resume requires this run's latest last.pt")
        restore_rng(checkpoint["rng"])
    else:
        initial_train = evaluate(model, train.records, settings.batch_size)
        initial_val = evaluate(model, val.records, settings.batch_size)
        best = initial_val["total"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def log(event):
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    def checkpoint_payload():
        return {"schema_version": 1, "config": config, "experiment_id": run_id,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(), "rng": rng_state(), "step": step,
                "best_val": best, "best_step": best_step, "bad_validations": bad_validations,
                "initial_train": initial_train, "initial_val": initial_val,
                "early_stopped": stopped}

    if not resume:
        log({"event": "initial", "step": 0, "train": initial_train, "val": initial_val})
        save_checkpoint(run_dir / "best.pt", checkpoint_payload())
        save_checkpoint(run_dir / "last.pt", checkpoint_payload())
    try:
        while step < settings.max_steps and not stopped:
            indices = sampler.next(settings.batch_size)
            batch = model_batch([train[i] for i in indices], device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(batch)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise RuntimeError(f"non-finite training loss at step {step + 1}")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.grad_clip, error_if_nonfinite=True).item()
            if gradient_norm == 0:
                raise RuntimeError("zero total gradient")
            module_norms = {
                name: math.sqrt(sum(p.grad.square().sum().item() for p in module.parameters()
                                    if p.grad is not None))
                for name, module in model.named_children()
            }
            optimizer.step()
            step += 1
            log({"event": "train", "step": step,
                 "loss": {name: value.item() for name, value in losses.items()},
                 "gradient_norm_before_clip": gradient_norm,
                 "module_gradient_norm_after_clip": module_norms,
                 "valid_observations": int(batch["observation_mask"].sum()),
                 "valid_transitions": int(batch["transition_mask"].sum()),
                 **({"valid_recall_targets": int(recall_targets(batch, model.config.recall_t)[1].sum())}
                    if model.config.recall_scale > 0 else {})})
            if step % settings.val_every == 0 or step == settings.max_steps:
                validation = evaluate(model, val.records, settings.batch_size)
                improved = validation["total"] < best - settings.min_delta
                if improved:
                    best, best_step, bad_validations = validation["total"], step, 0
                else:
                    bad_validations += 1
                # Smoke must execute its requested 100 updates; formal mode stops on val only.
                stopped = config["mode"] == "formal" and bad_validations >= settings.patience
                log({"event": "validation", "step": step, "loss": validation,
                     "best_step": best_step, "bad_validations": bad_validations})
                payload = checkpoint_payload()
                if improved:
                    save_checkpoint(run_dir / "best.pt", payload)
                save_checkpoint(run_dir / "last.pt", payload)
                print(f"step={step} train={losses['total'].item():.6f} "
                      f"val={validation['total']:.6f} grad={gradient_norm:.6f}"
                      + (f" recall={validation['recall']:.6f} "
                         f"recall_acc={validation['recall_accuracy']:.3f}"
                         if model.config.recall_scale > 0 else ""), flush=True)
        final_train = evaluate(model, train.records, settings.batch_size)
        final_val = evaluate(model, val.records, settings.batch_size)
    except (RuntimeError, ValueError) as exc:
        (run_dir / "failure.json").write_text(json.dumps(
            {"status": "training_failed", "step": step, "reason": str(exc)}) + "\n")
        raise
    passed = final_train["total"] < initial_train["total"] and best_step > 0
    report = {"status": "loss_decreased" if passed else "training_failed_no_improvement",
              "experiment_id": run_id, "run_dir": str(run_dir), "device": str(device),
              "seed": config["seed"], "mode": config["mode"], "steps": step,
              "train_episodes": len(train), "val_episodes": len(val),
              "initial_train": initial_train, "final_train": final_train,
              "initial_val": initial_val, "final_val": final_val, "best_val": best,
              "best_step": best_step, "early_stopped": stopped,
              "best_checkpoint": str(run_dir / "best.pt"),
              "last_checkpoint": str(run_dir / "last.pt"),
              "memory_sufficiency": "not_established_by_world_model_loss"}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
