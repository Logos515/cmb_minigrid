"""Collect Step 3 trajectories and a grouped split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cmb_minigrid.config import ConfigError, load_yaml
from cmb_minigrid.data import collect_records, split_records, write_jsonl, write_manifest
from cmb_minigrid.environments.hidden_history import ENV_ID, MAP_FAMILY_ID, MAP_FAMILY_IDS
from cmb_minigrid.reproducibility import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect HiddenHistoryNav trajectory data")
    parser.add_argument("--config", type=Path, help="YAML data collection configuration")
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--pairs", type=int, default=None, help="legacy alias for --pairs-per-family")
    count.add_argument(
        "--pairs-per-family", type=int, default=None, help="number of pairs generated for each map family"
    )
    parser.add_argument("--map-families", nargs="+", choices=MAP_FAMILY_IDS, help="map family IDs to collect")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exploration-rate", type=float, default=None)
    parser.add_argument("--no-exploration", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=None)
    parser.add_argument("--force", action="store_true", help="allow replacing existing output files")
    return parser


def _ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path}; pass --force")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        values: dict[str, Any] = load_yaml(args.config) if args.config else {}
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    pairs = (
        args.pairs_per_family
        if args.pairs_per_family is not None
        else args.pairs
        if args.pairs is not None
        else int(values.get("pairs_per_family", values.get("pairs", 2)))
    )
    map_families = args.map_families if args.map_families is not None else values.get(
        "map_families", [values.get("map_family_id", MAP_FAMILY_ID)]
    )
    if not isinstance(map_families, (list, tuple)) or not map_families:
        raise SystemExit("map_families must be a non-empty list of family IDs")
    if values.get("env_id", ENV_ID) != ENV_ID:
        raise SystemExit(f"only env_id={ENV_ID} is supported by this collector")
    seed = args.seed if args.seed is not None else int(values.get("seed", 42))
    exploration_rate = (
        args.exploration_rate
        if args.exploration_rate is not None
        else float(values.get("exploration_rate", 0.15))
    )
    include_exploration = bool(values.get("include_exploration", True)) and not args.no_exploration
    output = args.output or Path(values.get("output", "data/samples/step3_trajectories.jsonl"))
    manifest_path = args.manifest or Path(values.get("manifest", "data/samples/step3_manifest.json"))
    raw_ratios = args.split_ratios or values.get("split_ratios", (0.8, 0.1, 0.1))
    if not isinstance(raw_ratios, (list, tuple)) or len(raw_ratios) != 3:
        raise SystemExit("split_ratios must contain exactly three numbers")
    ratios = tuple(float(value) for value in raw_ratios)
    if pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if seed < 0:
        raise SystemExit("--seed must be non-negative")
    _ensure_writable(output, args.force)
    _ensure_writable(manifest_path, args.force)
    set_seed(seed)
    try:
        records = collect_records(
            pair_count=pairs,
            seed=seed,
            exploration_rate=exploration_rate,
            include_exploration=include_exploration,
            map_family_ids=tuple(str(value) for value in map_families),
        )
        manifest = split_records(records, seed=seed, ratios=ratios)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_jsonl(records, output)
    write_manifest(manifest, manifest_path)
    failures = sum(bool(record.truncated[-1]) for record in records)
    successes = sum(bool(record.terminated[-1]) for record in records)
    print(
        f"records={len(records)} pairs_per_family={pairs} families={len(map_families)} "
        f"pairs={pairs * len(map_families)} successes={successes} truncated={failures}"
    )
    print(f"jsonl={output}")
    print(f"manifest={manifest_path}")
    print("split_counts=" + ",".join(
        f"{name}={sum(split == name for split in manifest.assignment.values())}"
        for name in ("train", "val", "test")
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
