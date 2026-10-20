"""Step 9: closed-loop takeover of every initialization method plus CMB History Swap."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from cmb_minigrid.bridge.cmb import load_cmb
from cmb_minigrid.handoff.dataset import file_sha256, load_handoff_dataset
from cmb_minigrid.handoff.initializers import (
    MapInit,
    OracleInit,
    ReplayInit,
    ResetInit,
    load_state_map,
)
from cmb_minigrid.handoff.runner import (
    CMBInit,
    HandoffRunner,
    main_tables,
    run_history_swap,
    summarize_swap,
    write_csv,
)
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.runtime import select_device
from cmb_minigrid.training.policy import load_policy
from cmb_minigrid.training.rssm import load_rssm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 9 handoff closed loop and History Swap")
    p.add_argument("--handoff-dataset", type=Path, required=True)
    p.add_argument("--checkpoint-a", type=Path, required=True)
    p.add_argument("--checkpoint-b", type=Path, required=True)
    p.add_argument("--policy-b", type=Path, required=True)
    p.add_argument("--cmb-checkpoint", type=Path, nargs="+", required=True)
    p.add_argument("--cmb-names", nargs="+", default=None,
                   help="method names for --cmb-checkpoint (default: cmb or cmb_seed<seed>)")
    p.add_argument("--linear-map", type=Path, required=True)
    p.add_argument("--mlp-map", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--roles", nargs="+", default=["expert"],
                   help="expert only by default: exploration shares prefix/snapshot with expert")
    p.add_argument("--replay-windows", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--limit", type=int, default=None, help="first N samples (smoke runs)")
    p.add_argument("--stochastic", action="store_true", help="sample B states (seeded)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p


def run(args) -> dict:
    set_seed(args.seed, deterministic=True)
    torch.set_num_threads(1)
    device = select_device(args.device)
    rssm_a, _ = load_rssm(args.checkpoint_a, device)
    rssm_b, _ = load_rssm(args.checkpoint_b, device)
    policy, _ = load_policy(args.policy_b, device)
    for m in (rssm_a, rssm_b, policy):
        m.eval().requires_grad_(False)
    samples, meta = load_handoff_dataset(args.handoff_dataset)
    hashes = {"checkpoint_a_sha256": file_sha256(args.checkpoint_a),
              "checkpoint_b_sha256": file_sha256(args.checkpoint_b),
              "policy_b_sha256": file_sha256(args.policy_b)}
    for key in ("checkpoint_a_sha256", "checkpoint_b_sha256"):
        if meta.get(key) != hashes[key]:
            raise ValueError(f"handoff dataset was built with a different {key}")
    methods = [ResetInit(rssm_b), *[ReplayInit(rssm_b, n) for n in args.replay_windows],
               MapInit("linear", load_state_map(args.linear_map)[0], rssm_a, rssm_b),
               MapInit("mlp", load_state_map(args.mlp_map)[0], rssm_a, rssm_b)]
    if args.cmb_names and len(args.cmb_names) != len(args.cmb_checkpoint):
        raise ValueError("--cmb-names must match --cmb-checkpoint in length")
    cmbs = []
    for path in args.cmb_checkpoint:
        bridge, payload = load_cmb(path, device)
        for key in ("checkpoint_a_sha256", "checkpoint_b_sha256", "policy_b_sha256"):
            if payload.get(key) != hashes[key]:
                raise ValueError(f"{path} was trained against a different {key}")
        if args.cmb_names:
            name = args.cmb_names[len(cmbs)]
        else:
            name = "cmb" if len(args.cmb_checkpoint) == 1 else f"cmb_seed{payload['seed']}"
        cmbs.append(CMBInit(bridge, rssm_a, rssm_b, name))
    methods += cmbs + [OracleInit(rssm_b)]

    selected = [s for s in samples if s.split in args.splits and s.episode_role in args.roles]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no samples selected")
    runner = HandoffRunner(rssm_a, rssm_b, policy, methods, stochastic=args.stochastic,
                           seed=args.seed)
    rows = runner.run(selected)
    tables = main_tables(rows)
    swap_rows, swap_summary = [], {}
    for cmb in cmbs:
        part = run_history_swap(runner, cmb, selected)
        for r in part:
            r["cmb"] = cmb.name
        swap_rows += part
        swap_summary[cmb.name] = summarize_swap(part)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    with (out / "handoff_runs.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    write_csv(tables, out / "main_table.csv")
    write_csv(rows, out / "handoff_runs.csv")
    write_csv(swap_rows, out / "history_swap_rows.csv")
    write_csv([{"cmb": k, **row} for k, v in swap_summary.items() for row in v],
              out / "history_swap_summary.csv")
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hashes": hashes, "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "samples": len(selected), "methods": [m.name for m in methods],
        "splits": args.splits, "roles": args.roles, "main_table": tables,
        "history_swap_summary": swap_summary,
        "all_snapshots_restored_equal": all(r["snapshot_restored_equal"] for r in rows),
        "history_swap_integrity": all(r["snapshot_equal"] for r in swap_rows),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"run_handoff_failed: {exc}")
        return 1
    print(f"samples={report['samples']} methods={report['methods']}")
    print(f"all_snapshots_restored_equal={report['all_snapshots_restored_equal']} "
          f"history_swap_integrity={report['history_swap_integrity']}")
    for row in report["main_table"]:
        if row["scope"] == "split":
            print(f"{row['split']:5s} {row['method']:15s} n={row['n']:3d} "
                  f"success={row['success_rate']:.2f} branch={row['first_branch_acc']:.2f} "
                  f"wrong={row['mean_wrong_actions']:.1f} coll={row['mean_collisions']:.1f} "
                  f"eff={row['path_efficiency_all']:.2f} bytes={row['payload_bytes']:.0f}")
    print(f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
