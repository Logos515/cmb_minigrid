"""Step 7: build the handoff dataset, fit Linear/MLP A->B maps, report init checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from cmb_minigrid.data.dataset import read_manifest, validate_split_manifest
from cmb_minigrid.data.trajectory import read_jsonl
from cmb_minigrid.handoff.dataset import (
    build_handoff_samples,
    file_sha256,
    save_handoff_dataset,
)
from cmb_minigrid.handoff.initializers import (
    MapInit,
    OracleInit,
    ReplayInit,
    ResetInit,
    StateMap,
    collect_aligned_states,
    compare_states,
    fit_state_map,
    initialize_all,
    map_loss,
    matched_hidden_dim,
    save_state_map,
)
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.runtime import select_device
from cmb_minigrid.training.rssm import load_rssm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 7 handoff dataset and baselines")
    p.add_argument("--checkpoint-a", type=Path, required=True)
    p.add_argument("--checkpoint-b", type=Path, required=True)
    p.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay-windows", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--map-steps", type=int, default=500)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-bottleneck", type=int, default=64)
    p.add_argument("--mlp-target-params", type=int, default=None,
                   help="if set, choose the MLP hidden width to match this parameter count")
    return p


def run(args) -> dict:
    set_seed(args.seed, deterministic=True)
    torch.set_num_threads(1)
    device = select_device(args.device)
    rssm_a, _ = load_rssm(args.checkpoint_a, device)
    rssm_b, _ = load_rssm(args.checkpoint_b, device)
    for m in (rssm_a, rssm_b):
        m.eval().requires_grad_(False)
    records = read_jsonl(args.input)
    manifest = read_manifest(args.manifest)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(errors))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)

    samples, build_info = build_handoff_samples(records, manifest, rssm_a, rssm_b)
    meta = {"checkpoint_a_sha256": file_sha256(args.checkpoint_a),
            "checkpoint_b_sha256": file_sha256(args.checkpoint_b),
            "input_sha256": file_sha256(args.input),
            "manifest_sha256": file_sha256(args.manifest), "seed": args.seed}
    save_handoff_dataset(samples, out / "handoff_dataset.pt", meta=meta)

    # Linear / MLP maps are fitted on train-split states only.
    train_records = [r for r in records if manifest.assignment[r.episode_id] == "train"]
    train = collect_aligned_states(rssm_a, rssm_b, train_records)
    in_dim, deter = rssm_a.feature_dim, rssm_b.config.deter
    out_dim = deter + rssm_b.stoch_dim
    hidden = (matched_hidden_dim(args.mlp_target_params, in_dim=in_dim, out_dim=out_dim,
                                 bottleneck=args.mlp_bottleneck)
              if args.mlp_target_params else args.mlp_hidden)
    maps, fit = {}, {}
    for name, kind in (("linear", "linear"), ("mlp", "mlp")):
        module = StateMap(kind, in_dim, deter, hidden=hidden,
                          bottleneck=args.mlp_bottleneck if kind == "mlp" else None)
        fit[name] = fit_state_map(module, train, steps=args.map_steps, seed=args.seed)
        fit[name]["num_parameters"] = module.num_parameters
        _, fit[name]["train_terms"] = map_loss(module, train)
        save_state_map(module, out / f"{name}_map.pt", {"train": fit[name], **meta})
        maps[name] = module

    methods = [ResetInit(rssm_b),
               *[ReplayInit(rssm_b, n) for n in args.replay_windows],
               ReplayInit(rssm_b, 10_000, "replay_full"),
               MapInit("linear", maps["linear"], rssm_a, rssm_b),
               MapInit("mlp", maps["mlp"], rssm_a, rssm_b),
               OracleInit(rssm_b)]

    # Run every method on every sample, twice, and aggregate distances to Oracle-B.
    agg: dict = defaultdict(lambda: defaultdict(list))
    max_replay_full_vs_oracle = 0.0
    repeatable = same_snapshot = True
    for s in samples:
        first = initialize_all(methods, s, rssm_b=rssm_b)
        second = initialize_all(methods, s, rssm_b=rssm_b)
        for name, init in first.items():
            repeatable &= (torch.equal(init.state_b.deter, second[name].state_b.deter)
                           and torch.equal(init.state_b.stoch, second[name].state_b.stoch))
            same_snapshot &= init.env_snapshot == s.env_snapshot and init.t == s.t
            d = compare_states(init.state_b, s.state_b_star)
            for k, v in d.items():
                agg[(s.split, name)][k].append(float(v))
        max_replay_full_vs_oracle = max(
            max_replay_full_vs_oracle,
            compare_states(first["replay_full"].state_b, s.state_b_star)["deter_max_abs"])
    table = {f"{split}/{name}": {k: sum(v) / len(v) for k, v in stats.items()} | {"n": len(
        stats["deter_l2"])} for (split, name), stats in sorted(agg.items())}

    # Counterfactual pairs: identical obs_t and recent history, different A/B states.
    by_pair = defaultdict(list)
    for s in samples:
        if s.episode_role == "expert":
            by_pair[s.pair_id].append(s)
    pair_checks = []
    for pid, ends in sorted(by_pair.items()):
        if len(ends) == 2:
            a, b = ends
            pair_checks.append({
                "pair_id": pid,
                "obs_t_equal": bool(torch.equal(a.history.images[-1], b.history.images[-1])),
                "state_a_distance": (a.state_a.deter - b.state_a.deter).norm().item(),
                "state_b_star_distance": (a.state_b_star.deter - b.state_b_star.deter).norm().item(),
            })

    report = {
        "samples": len(samples),
        "samples_by_split": {k: sum(s.split == k for s in samples) for k in ("train", "val", "test")},
        "skipped_without_handoff": build_info["skipped_without_handoff"],
        "handoff_t": sorted({s.t for s in samples}),
        "methods": [m.name for m in methods],
        "map_fit": fit,
        "mlp": {"hidden": hidden, "bottleneck": args.mlp_bottleneck,
                "target_params": args.mlp_target_params},
        "all_methods_repeatable": bool(repeatable),
        "all_methods_same_snapshot_and_t": bool(same_snapshot),
        "max_abs_replay_full_vs_oracle_deter": max_replay_full_vs_oracle,
        "distance_to_oracle_b_star": table,
        "pair_checks": {"n_pairs": len(pair_checks),
                        "all_obs_t_equal": all(c["obs_t_equal"] for c in pair_checks),
                        "min_state_a_distance": min((c["state_a_distance"] for c in pair_checks),
                                                    default=None),
                        "min_state_b_star_distance": min(
                            (c["state_b_star_distance"] for c in pair_checks), default=None)},
        "meta": meta,
        "dataset": str(out / "handoff_dataset.pt"),
    }
    (out / "handoff_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"build_handoff_failed: {exc}")
        return 1
    print(json.dumps({k: report[k] for k in (
        "samples", "samples_by_split", "handoff_t", "all_methods_repeatable",
        "all_methods_same_snapshot_and_t", "max_abs_replay_full_vs_oracle_deter")}, sort_keys=True))
    print(f"report={args.output_dir / 'handoff_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
