"""Print compact statistics for a trajectory JSONL dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cmb_minigrid.data import read_jsonl, read_manifest, validate_split_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Step 3 trajectory data")
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.input)
    manifest = read_manifest(args.manifest)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise SystemExit("invalid split manifest: " + "; ".join(errors))
    payload = {
        "records": len(records),
        "pairs": len({record.pair_id for record in records}),
        "map_families": sorted({record.map_family_id for record in records}),
        "roles": dict(Counter(record.episode_role for record in records)),
        "terminated": sum(bool(record.terminated[-1]) for record in records),
        "truncated": sum(bool(record.truncated[-1]) for record in records),
        "horizon_min": min(record.horizon for record in records),
        "horizon_max": max(record.horizon for record in records),
        "split_counts": dict(Counter(manifest.assignment.values())),
        "per_family": {
            family: {
                "pairs": len({r.pair_id for r in records if r.map_family_id == family}),
                "records": sum(r.map_family_id == family for r in records),
                "split": next(manifest.assignment[r.episode_id] for r in records if r.map_family_id == family),
            }
            for family in sorted({r.map_family_id for r in records})
        },
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
