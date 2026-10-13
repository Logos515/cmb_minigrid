"""Step 5 counterfactual cue probes on independently loaded frozen RSSMs."""

import argparse
import json
from pathlib import Path

from cmb_minigrid.runtime import select_device
from cmb_minigrid.training.probe import run_probes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 5 frozen RSSM-A/B linear cue probes")
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        report = run_probes(args.checkpoint_a, args.checkpoint_b, args.input, args.manifest,
                            args.output_dir, seed=args.seed, device=select_device(args.device),
                            iterations=args.iterations)
        for name, model in report["models"].items():
            print(json.dumps({"model": name, "seed": model["seed"],
                              "probes": model["probes"]}, sort_keys=True))
        print(f"report={args.output_dir / 'probe_report.json'}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"probe_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
