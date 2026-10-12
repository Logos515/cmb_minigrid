"""Generate and validate strict HiddenHistoryNav counterfactual pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cmb_minigrid.environments.hidden_history import (
    HANDOFF_T,
    generate_counterfactual_pair,
    validate_pair,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HiddenHistoryNav counterfactual pair smoke test")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/samples/hidden_history_pairs.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")

    rng = np.random.default_rng(args.seed)
    passed = 0
    reachable = 0
    random_correct = 0
    first_pair = None
    for index in range(args.count):
        pair = generate_counterfactual_pair(args.seed + index)
        report = validate_pair(pair, k=8)
        if not report:
            raise SystemExit(f"pair {index} failed checks: {report.checks}")
        passed += 1
        reachable += int(pair.left.reached_goal and pair.right.reached_goal)
        guessed_branch = "left" if int(rng.integers(0, 2)) == 0 else "right"
        random_correct += int(guessed_branch == pair.left.correct_branch)
        if first_pair is None:
            first_pair = pair

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert first_pair is not None
    payload = {
        "env_id": "HiddenHistoryNav-v0",
        "count": args.count,
        "seed": args.seed,
        "handoff_t": HANDOFF_T,
        "k": 8,
        "validated_pairs": passed,
        "reachable_pairs": reachable,
        "random_no_history_correct": random_correct,
        "random_no_history_accuracy": random_correct / args.count,
        "first_pair": {
            "seed": first_pair.seed,
            "left_hidden_label": first_pair.left.hidden_label,
            "right_hidden_label": first_pair.right.hidden_label,
            "left_correct_branch": first_pair.left.correct_branch,
            "right_correct_branch": first_pair.right.correct_branch,
            "left_handoff_snapshot": first_pair.left.snapshots[HANDOFF_T].to_dict(),
            "right_handoff_snapshot": first_pair.right.snapshots[HANDOFF_T].to_dict(),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("env_id=HiddenHistoryNav-v0")
    print(f"pairs={args.count} validated={passed} reachable={reachable}")
    print(f"handoff_t={HANDOFF_T} recent_k=8")
    print(f"random_no_history_accuracy={random_correct / args.count:.3f} n={args.count}")
    print(f"sample={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
