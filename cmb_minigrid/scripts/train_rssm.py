"""Configurable Step 5 training of one independently initialized RSSM."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cmb_minigrid.config import RuntimeConfig, load_yaml
from cmb_minigrid.models import RSSMConfig
from cmb_minigrid.training.rssm import TrainingConfig, fingerprint, train_rssm


def build_parser():
    parser = argparse.ArgumentParser(description="Step 5 independent RSSM sequence training")
    parser.add_argument("--config", type=Path, required=True, help="RSSM-A or RSSM-B preset")
    parser.add_argument("--training-config", type=Path, default=Path("configs/train_rssm.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/step5"))
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--max-steps", type=int, help="total updates, including restored steps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=Path, help="restore latest last.pt including optimizer/RNG")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        model_values = load_yaml(args.config)
        values = load_yaml(args.training_config)
        runtime = RuntimeConfig.from_mapping(model_values | {
            name: getattr(args, name) for name in ("seed", "device")
            if getattr(args, name) is not None})
        training = dict(values["training"])
        if args.mode == "smoke":
            training["max_steps"] = 100
        for name in ("max_steps", "batch_size"):
            if getattr(args, name) is not None:
                training[name] = getattr(args, name)
        training = asdict(TrainingConfig(**training))
        resolved = {"experiment_name": runtime.experiment_name, "seed": runtime.seed,
                    "device": runtime.device, "deterministic": runtime.deterministic,
                    "rssm": asdict(RSSMConfig.from_mapping(model_values["rssm"])),
                    "training": training, "mode": args.mode}
        for name in ("input", "manifest"):
            path = Path(getattr(args, name) or values[name]).resolve()
            resolved[name] = str(path)
            resolved[f"{name}_sha256"] = fingerprint(path)
        report = train_rssm(resolved, args.output_dir, resume=args.resume)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "loss_decreased" else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"training_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
