"""Smoke-test the official MiniGrid Memory environment and save a rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

from cmb_minigrid.environments.memory import ACTION_NAMES, make_memory_env
from cmb_minigrid.environments.rollout import rollout, write_rollout_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniGrid-MemoryS13 interface smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--policy", choices=("random", "scripted"), default="random")
    parser.add_argument("--output", type=Path, default=None, help="sample JSON output path")
    parser.add_argument("--render", action="store_true", help="render each transition")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise SystemExit("--seed must be a non-negative integer")
    if args.steps < 0:
        raise SystemExit("--steps must be a non-negative integer")
    if args.policy == "scripted":
        from cmb_minigrid.environments.rollout import SCRIPTED_ACTIONS

        actions = tuple(SCRIPTED_ACTIONS[index % len(SCRIPTED_ACTIONS)] for index in range(args.steps))
    else:
        actions = None

    env = make_memory_env(render_mode="human" if args.render else None)
    try:
        trajectory = rollout(
            env,
            seed=args.seed,
            steps=args.steps,
            actions=actions,
            render=args.render,
        )
    finally:
        env.close()

    if len(trajectory.actions) != args.steps:
        raise SystemExit(
            f"episode ended after {len(trajectory.actions)} transitions; "
            f"requested {args.steps}"
        )
    output = args.output or Path("data/samples") / f"minigrid_memory_seed{args.seed}.json"
    write_rollout_json(trajectory, output)
    first = trajectory.observations[0]
    print("env_id=MiniGrid-MemoryS13-v0")
    print(f"actions={','.join(ACTION_NAMES)}")
    print(f"observation=image:{first.image.shape}:{first.image.dtype},direction:int,task_id:str")
    print(f"seed={args.seed} transitions={len(trajectory.actions)}")
    print(f"initial_observation_hash={trajectory.observation_hashes[0]}")
    print(f"final_observation_hash={trajectory.observation_hashes[-1]}")
    print(f"sample={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
