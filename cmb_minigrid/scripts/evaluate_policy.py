"""Evaluate Step 6 full-history and reset closed loops on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmb_minigrid.training.policy import evaluate_closed_loops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 6 closed-loop policy evaluation")
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--policy-a", type=Path, required=True)
    parser.add_argument("--policy-b", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_closed_loops(
            checkpoint_a=args.checkpoint_a,
            checkpoint_b=args.checkpoint_b,
            policy_a=args.policy_a,
            policy_b=args.policy_b,
            input_path=args.input,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            device=args.device,
        )
        print(json.dumps(report, sort_keys=True))
        # A gap is an empirical result, not a pass/fail assertion. Evaluation
        # succeeds when all requested trajectories were evaluated and saved.
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"policy_evaluation_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
