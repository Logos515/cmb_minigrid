"""CPU-first Step 4 interface/backprop/single-batch diagnostic, not a trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cmb_minigrid.config import RuntimeConfig, load_yaml
from cmb_minigrid.data.dataset import TrajectoryDataset, collate_trajectories
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.reproducibility import experiment_id, set_seed
from cmb_minigrid.runtime import select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4 RSSM single-batch acceptance diagnostic")
    parser.add_argument("--config", type=Path, default=Path("configs/rssm_a.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=8, help="prefix transition count")
    parser.add_argument("--overfit-steps", type=int, default=40, help="0 disables diagnostic updates")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.sequence_length <= 0 or args.overfit_steps < 0:
        parser.error("batch-size/sequence-length must be positive; overfit-steps non-negative")
    values = load_yaml(args.config)
    overrides = {"device": args.device}
    if args.seed is not None:
        overrides["seed"] = args.seed
    runtime = RuntimeConfig.from_mapping(values | overrides)
    config = RSSMConfig.from_mapping(values["rssm"])
    set_seed(runtime.seed, deterministic=runtime.deterministic)
    torch.set_num_threads(1)
    device = select_device(runtime.device)
    dataset = TrajectoryDataset(args.input, split="train", manifest_path=args.manifest)
    if len(dataset) < args.batch_size:
        parser.error("train split is smaller than the requested batch")
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_trajectories)))
    horizon = min(args.sequence_length, batch["actions"].shape[1])
    # Transfer only model inputs/targets; exclude expert actions and metadata.
    obs = batch["agent_obs"]
    data = {
        "agent_obs": {
            "image": obs["image"][:, :horizon + 1].to(device),
            "direction": obs["direction"][:, :horizon + 1].to(device),
            "task_id": [row[:horizon + 1] for row in obs["task_id"]],
        },
        "observation_mask": batch["observation_mask"][:, :horizon + 1].to(device),
    }
    for name in ("actions", "reward", "terminated", "truncated", "transition_mask"):
        data[name] = batch[name][:, :horizon].to(device)
    model = RSSM(config).to(device)
    model.train()
    losses = model.loss(data)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("non-finite sampled loss")
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("missing or non-finite gradients")
    grad_norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt().item()
    if grad_norm == 0:
        raise RuntimeError("zero total gradient")
    sampled_loss = losses["total"].item()

    # Deterministic categorical modes reduce diagnostic noise. eval() still
    # permits gradients; this is a bounded memorization check on one train batch.
    model.eval()
    with torch.no_grad():
        initial = {key: value.item() for key, value in model.loss(data).items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(data)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("non-finite overfit loss")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        output = model.observe(data["agent_obs"], data["actions"], data["observation_mask"])
        imagined = model.imagine_step(output.states[-1], torch.zeros(
            args.batch_size, dtype=torch.long, device=device))
        final = {key: value.item() for key, value in model.loss(data).items()}
    if not all(torch.isfinite(torch.tensor(value)) for value in final.values()):
        raise RuntimeError("non-finite final loss")
    passed = args.overfit_steps == 0 or final["total"] < 0.7 * initial["total"]
    resolved = values | vars(runtime) | {"rssm": vars(config), "diagnostic": vars(args)}
    run_id = experiment_id(runtime.experiment_name, runtime.seed, config=resolved)
    log_dir = runtime.output_dir / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": run_id, "seed": runtime.seed, "device": str(device),
        "config": resolved, "split": "train", "train_episodes": len(dataset),
        "parameters": sum(p.numel() for p in model.parameters()),
        "features_shape": list(output.features.shape),
        "posterior_shape": list(output.posterior_logits.shape),
        "imagine_features_shape": list(model.features(imagined).shape),
        "sampled_loss": sampled_loss, "gradient_norm": grad_norm,
        "overfit_steps": args.overfit_steps, "initial": initial, "final": final,
        "overfit_passed": passed, "overfit_ratio": final["total"] / initial["total"],
    }
    report_path = log_dir / "smoke_rssm.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, default=str))
    print(f"report={report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
