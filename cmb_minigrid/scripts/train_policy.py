"""Train one Actor-B style policy on frozen RSSM posterior features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmb_minigrid.config import load_yaml
from cmb_minigrid.runtime import select_device
from cmb_minigrid.training.policy import PolicyTrainingConfig, train_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 6 behavior cloning on frozen RSSM features")
    parser.add_argument("--rssm-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train_policy.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--class-weighting", choices=("inverse", "sqrt", "none"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        values = load_yaml(args.config)
        train_values = dict(values.get("training", {}))
        for name in ("max_steps", "batch_size"):
            if getattr(args, name) is not None:
                train_values[name] = getattr(args, name)
        if args.class_weighting is not None:
            train_values["class_weighting"] = args.class_weighting
        training = PolicyTrainingConfig(**train_values)
        input_path = Path(args.input or values["input"]).resolve()
        manifest_path = Path(args.manifest or values["manifest"]).resolve()
        device = select_device(args.device or values.get("device", "cpu"))
        seed = args.seed if args.seed is not None else int(values.get("seed", 42))
        report = train_policy(
            rssm_checkpoint=args.rssm_checkpoint,
            input_path=input_path,
            manifest_path=manifest_path,
            output_dir=args.output_dir,
            seed=seed,
            device=str(device),
            mode=args.mode,
            training=training,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "loss_decreased" else 1
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"policy_training_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
