"""Step 10 orchestration: RSSM seed groups -> Actor-B -> handoff -> CMB variants -> closed loop.

Every stage is an existing CLI of this repository, run as a subprocess so the
exact command lines (with exit codes and wall-clock seconds) land in
``<root>/manifest.json``. Finished stages are skipped on re-run (resumable).

A *group* is one independent RSSM-A/RSSM-B training-seed pair plus its own
Actor-B, handoff dataset, Linear/MLP maps and CMB variants. Groups ``g1..gN`` train
new models with ``A seed = 1000+g`` and ``B seed = 2000+g`` using the Step 5/6
recipes; ``g0`` is the pre-existing user checkpoint set (``models/best_a.pt``,
``best_b.pt``, ``policy_b.pt``), reported separately because its training recipe
differs slightly (e.g. early-stopping patience).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MLP_TARGET_PARAMS = 140608  # parameter count of the 4x16 CMB (Step 8)
REPLAY_WINDOWS = [2, 4, 6, 8, 10, 12, 13, 16]
CMB_VARIANTS = {  # name -> extra train_cmb arguments
    "cmb": ["--tokens", "4", "--token-classes", "16"],
    "cap_2x16": ["--tokens", "2", "--token-classes", "16"],
    "cap_8x16": ["--tokens", "8", "--token-classes", "16"],
    "cap_8x32": ["--tokens", "8", "--token-classes", "32"],
    "loss_state_only": ["--action-weight", "0", "--stage2-steps", "0"],
    "loss_state_action": ["--stage2-steps", "0"],
}


@dataclass
class Job:
    group: str
    name: str
    argv: list[str]
    done: Callable[[], bool]


@dataclass
class Context:
    root: Path
    smoke: bool
    groups: list[int]
    legacy: bool
    paths: dict[str, dict[str, Path]] = field(default_factory=dict)

    def gdir(self, group: str) -> Path:
        return self.root / group

    def names(self) -> list[str]:
        return (["g0"] if self.legacy else []) + [f"g{g}" for g in self.groups]


def py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def first(pattern_dir: Path, name: str) -> Path | None:
    found = sorted(pattern_dir.glob(f"*/{name}")) if pattern_dir.exists() else []
    return found[0] if found else None


def phase_rssm(ctx: Context) -> list[Job]:
    jobs = []
    mode = ["--mode", "smoke"] if ctx.smoke else []
    for g in ctx.groups:
        group = f"g{g}"
        for role, cfg, seed in (("a", "rssm_a", 1000 + g), ("b", "rssm_b", 2000 + g)):
            out = ctx.gdir(group) / f"rssm_{role}"
            jobs.append(Job(group, f"rssm_{role}", py(
                "cmb_minigrid.scripts.train_rssm", "--config", f"configs/{cfg}.yaml",
                "--training-config", "configs/train_rssm_cnn.yaml", "--seed", str(seed),
                "--output-dir", str(out), "--device", "cpu", *mode),
                lambda out=out: first(out, "best.pt") is not None))
    return jobs


def resolve(ctx: Context) -> None:
    """Fill ``ctx.paths`` with the checkpoints each group uses."""
    for group in ctx.names():
        d = ctx.gdir(group)
        if group == "g0":
            ctx.paths[group] = {"a": REPO / "models/best_a.pt", "b": REPO / "models/best_b.pt",
                                "policy": REPO / "models/policy_b.pt"}
            continue
        a, b = first(d / "rssm_a", "best.pt"), first(d / "rssm_b", "best.pt")
        if a and b:
            ctx.paths[group] = {"a": a, "b": b, "policy": d / "policy_b" / "best.pt"}


def phase_probe_policy(ctx: Context) -> list[Job]:
    jobs = []
    for group, p in ctx.paths.items():
        d = ctx.gdir(group)
        jobs.append(Job(group, "probe", py(
            "cmb_minigrid.scripts.probe_rssm", "--checkpoint-a", str(p["a"]),
            "--checkpoint-b", str(p["b"]), "--output-dir", str(d / "probe"), "--device", "cpu"),
            lambda d=d: (d / "probe" / "probe_report.json").exists()))
        if group != "g0":
            jobs.append(Job(group, "policy_b", py(
                "cmb_minigrid.scripts.train_policy", "--rssm-checkpoint", str(p["b"]),
                "--config", "configs/train_policy_sqrt.yaml", "--seed", str(2000 + int(group[1:])),
                "--output-dir", str(d / "policy_b"), "--device", "cpu",
                *(["--mode", "smoke"] if ctx.smoke else [])),
                lambda p=p: p["policy"].exists()))
    return jobs


def phase_handoff(ctx: Context) -> list[Job]:
    return [Job(group, "handoff", py(
        "cmb_minigrid.scripts.build_handoff", "--checkpoint-a", str(p["a"]),
        "--checkpoint-b", str(p["b"]), "--output-dir", str(ctx.gdir(group) / "handoff"),
        "--seed", "42", "--mlp-target-params", str(MLP_TARGET_PARAMS),
        *(["--map-steps", "20"] if ctx.smoke else [])),
        lambda group=group: (ctx.gdir(group) / "handoff" / "handoff_report.json").exists())
        for group, p in ctx.paths.items()]


def phase_cmb(ctx: Context) -> list[Job]:
    jobs = []
    for group, p in ctx.paths.items():
        h = ctx.gdir(group) / "handoff"
        for name, extra in CMB_VARIANTS.items():
            out = ctx.gdir(group) / "cmb" / name
            jobs.append(Job(group, f"cmb_{name}", py(
                "cmb_minigrid.scripts.train_cmb", "--handoff-dataset", str(h / "handoff_dataset.pt"),
                "--checkpoint-a", str(p["a"]), "--checkpoint-b", str(p["b"]),
                "--policy-b", str(p["policy"]), "--linear-map", str(h / "linear_map.pt"),
                "--mlp-map", str(h / "mlp_map.pt"), "--output-dir", str(out), "--seed", "42",
                *(["--smoke"] if ctx.smoke else []), *extra),
                lambda out=out: (out / "report.json").exists()))
    return jobs


def phase_closed_loop(ctx: Context) -> list[Job]:
    jobs = []
    for group, p in ctx.paths.items():
        d = ctx.gdir(group)
        jobs.append(Job(group, "closed_loop", py(
            "cmb_minigrid.scripts.run_handoff", "--handoff-dataset",
            str(d / "handoff" / "handoff_dataset.pt"), "--checkpoint-a", str(p["a"]),
            "--checkpoint-b", str(p["b"]), "--policy-b", str(p["policy"]),
            "--cmb-checkpoint", *[str(d / "cmb" / n / "best.pt") for n in CMB_VARIANTS],
            "--cmb-names", *CMB_VARIANTS, "--replay-windows", *map(str, REPLAY_WINDOWS),
            "--linear-map", str(d / "handoff" / "linear_map.pt"),
            "--mlp-map", str(d / "handoff" / "mlp_map.pt"),
            "--output-dir", str(d / "closed_loop")),
            lambda d=d: (d / "closed_loop" / "report.json").exists()))
    return jobs


def execute(job: Job, logs: Path, force: bool) -> dict:
    record = {"group": job.group, "name": job.name,
              "command": " ".join(job.argv).replace(sys.executable, "python")}
    if job.done() and not force:
        return {**record, "status": "skipped_existing"}
    logs.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    started = time.time()
    with (logs / f"{job.group}_{job.name}.log").open("w") as log:
        code = subprocess.run(job.argv, cwd=REPO, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode
    return {**record, "status": "ok" if code == 0 else "failed", "exit_code": code,
            "seconds": round(time.time() - started, 1)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 10 pilot orchestration")
    p.add_argument("--root", type=Path, default=Path("runs/step10"))
    p.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="new RSSM training-seed groups (A seed 1000+g, B seed 2000+g)")
    p.add_argument("--no-legacy", action="store_true", help="skip g0 (existing checkpoints)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true", help="100-update RSSM/policy and 20-update CMB")
    p.add_argument("--force", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = Context(args.root, args.smoke, args.groups, not args.no_legacy)
    logs = ctx.root / "logs"
    records: list[dict] = []
    failed: set[str] = set()
    manifest_path = ctx.root / "manifest.json"
    previous = {}
    if manifest_path.exists():  # keep real timings of stages finished in earlier invocations
        previous = {(j["group"], j["name"]): j for j in json.loads(manifest_path.read_text())["jobs"]
                    if j.get("status") == "ok"}
    started = time.time()

    def run_phase(title: str, jobs: list[Job]) -> None:
        jobs = [j for j in jobs if j.group not in failed]
        print(f"== {title}: {len(jobs)} jobs", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            for record in pool.map(lambda j: execute(j, logs, args.force), jobs):
                if record["status"] == "skipped_existing":
                    record = previous.get((record["group"], record["name"]), record)
                records.append(record)
                print(f"  {record['group']:3s} {record['name']:22s} {record['status']}"
                      f" {record.get('seconds', '')}", flush=True)
                if record["status"] == "failed":
                    failed.add(record["group"])

    run_phase("RSSM-A/B training", phase_rssm(ctx))
    resolve(ctx)
    for group in ctx.names():
        if group not in ctx.paths:
            failed.add(group)
    run_phase("probe + Actor-B", phase_probe_policy(ctx))
    run_phase("handoff dataset + Linear/MLP", phase_handoff(ctx))
    run_phase("CMB variants", phase_cmb(ctx))
    run_phase("closed loop + History Swap", phase_closed_loop(ctx))

    import torch

    manifest = {
        "root": str(ctx.root), "smoke": args.smoke, "groups": ctx.names(),
        "failed_groups": sorted(failed), "wall_seconds": round(time.time() - started, 1),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                     text=True, check=False).stdout.strip(),
        "seeds": {g: {"rssm_a": 1000 + int(g[1:]), "rssm_b": 2000 + int(g[1:]),
                      "policy_b": 2000 + int(g[1:]), "cmb": 42}
                  for g in ctx.names() if g != "g0"},
        "paths": {g: {k: str(v) for k, v in p.items()} for g, p in ctx.paths.items()},
        "jobs": records,
    }
    ctx.root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"failed groups: {sorted(failed) or 'none'}; manifest={ctx.root / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
