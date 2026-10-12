"""Minimal Step 0 command-line entry point."""

from __future__ import annotations

import argparse

from .config import ConfigError, RuntimeConfig, load_runtime_config
from .runtime import RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CMB x MiniGrid Step 0 runtime scaffold")
    parser.add_argument("--config", help="YAML runtime configuration")
    parser.add_argument("--seed", type=int, help="override the configured random seed")
    parser.add_argument("--device", help="override device: auto, cpu, or cuda[:N]")
    parser.add_argument("--output-dir", help="override the run output directory")
    parser.add_argument("--experiment-name", help="override the experiment name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_runtime_config(args.config)
        overrides = {key: value for key, value in {
            "seed": args.seed, "device": args.device, "output_dir": args.output_dir,
            "experiment_name": args.experiment_name,
        }.items() if value is not None}
        if overrides:
            config = RuntimeConfig.from_mapping(vars(config) | overrides)
        context = RunContext.create(config)
    except (ConfigError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"experiment_id={context.experiment_id}")
    print(f"device={context.device}")
    print(f"log_dir={context.log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
