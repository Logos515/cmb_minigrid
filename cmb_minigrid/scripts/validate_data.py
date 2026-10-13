"""Validate trajectory records, replay determinism, and split grouping."""

from __future__ import annotations

import argparse
from pathlib import Path

from cmb_minigrid.data import (
    read_jsonl,
    read_manifest,
    replay_handoff,
    validate_counterfactual_records,
    validate_expert_actions,
    validate_split_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Step 3 trajectory data")
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--max-records", type=int, default=0, help="validate only the first N records")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.input)
    manifest = read_manifest(args.manifest)
    split_errors = validate_split_manifest(manifest, records=records)
    if split_errors:
        raise SystemExit("split validation failed: " + "; ".join(split_errors))
    pair_errors = validate_counterfactual_records(records)
    if pair_errors:
        raise SystemExit("counterfactual validation failed: " + "; ".join(pair_errors))
    selected = records if args.max_records <= 0 else records[: args.max_records]
    replayed = sum(replay_handoff(record) for record in selected)
    labels = sum(validate_expert_actions(record) for record in selected)
    if replayed != len(selected):
        raise SystemExit(f"replay validation failed: {replayed}/{len(selected)}")
    if labels != len(selected):
        raise SystemExit(f"expert label validation failed: {labels}/{len(selected)}")
    print(
        f"records={len(records)} replayed={replayed} labels={labels} "
        f"pairs={len({record.pair_id for record in records})} "
        "handoff_restore=validated split_errors=0 pair_errors=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
