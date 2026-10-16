"""Evaluate Actor behavior cloning on expert trajectories without closed-loop control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmb_minigrid.training.policy import evaluate_offline_replays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stepwise offline Actor replay evaluation")
    parser.add_argument("--rssm-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_offline_replays(
            rssm_checkpoint=args.rssm_checkpoint,
            policy_checkpoint=args.policy_checkpoint,
            input_path=args.input,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            split=args.split,
            device=args.device,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "trajectories"}, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"offline_policy_evaluation_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
