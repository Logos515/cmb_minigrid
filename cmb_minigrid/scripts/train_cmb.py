"""Step 8: train the CMB Bridge (stage 1 state/action, stage 2 multi-step) and evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmb_minigrid.bridge.cmb import CMBConfig
from cmb_minigrid.bridge.training import train_cmb
from cmb_minigrid.config import load_yaml
from cmb_minigrid.runtime import select_device
from cmb_minigrid.training.rssm import load_rssm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 8 CMB Bridge training")
    p.add_argument("--handoff-dataset", type=Path, required=True)
    p.add_argument("--checkpoint-a", type=Path, required=True)
    p.add_argument("--checkpoint-b", type=Path, required=True)
    p.add_argument("--policy-b", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/cmb.yaml"))
    p.add_argument("--linear-map", type=Path, default=None)
    p.add_argument("--mlp-map", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1-steps", type=int, default=300)
    p.add_argument("--stage2-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--tokens", type=int, default=None, help="override configs tokens")
    p.add_argument("--token-classes", type=int, default=None, help="override configs token_classes")
    p.add_argument("--state-weight", type=float, default=1.0)
    p.add_argument("--action-weight", type=float, default=1.0,
                   help="0 removes the Actor-B action-KL term (ablation)")
    p.add_argument("--multistep-weight", type=float, default=1.0)
    p.add_argument("--smoke", action="store_true",
                   help="20 + 20 updates, no early stopping; checks the pipeline only")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = select_device(args.device)
        cmb = load_yaml(args.config).get("cmb", {})
        rssm_a, _ = load_rssm(args.checkpoint_a, device)
        rssm_b, _ = load_rssm(args.checkpoint_b, device)
        config = CMBConfig(
            a_feature_dim=rssm_a.feature_dim, b_deter=rssm_b.config.deter,
            tokens=args.tokens or cmb.get("tokens", 4),
            token_classes=args.token_classes or cmb.get("token_classes", 16),
            encoder_dim=cmb.get("encoder_dim", 64),
            sender_hidden=cmb.get("sender_hidden", 128),
            receiver_hidden=cmb.get("receiver_hidden", 128))
        smoke = args.smoke
        report = train_cmb(
            handoff_path=args.handoff_dataset, checkpoint_a=args.checkpoint_a,
            checkpoint_b=args.checkpoint_b, policy_b=args.policy_b, output_dir=args.output_dir,
            seed=args.seed, config=config,
            stage1_steps=20 if smoke else args.stage1_steps,
            stage2_steps=(20 if args.stage2_steps else 0) if smoke else args.stage2_steps, batch_size=args.batch_size,
            lr=args.lr, val_every=10 if smoke else args.val_every,
            patience=10**9 if smoke else args.patience, linear_map_path=args.linear_map,
            mlp_map_path=args.mlp_map, device=device, state_weight=args.state_weight,
            action_weight=args.action_weight, multistep_weight=args.multistep_weight)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"train_cmb_failed: {exc}")
        return 1
    print(json.dumps({"stages": report["stages"], "overhead": report["overhead"]}, sort_keys=True))
    for split, block in report["evaluation"].items():
        print(split, json.dumps(block["cmb_vs_reset"], sort_keys=True))
    print(f"report={args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
