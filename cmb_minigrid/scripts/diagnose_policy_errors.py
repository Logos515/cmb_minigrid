"""Inspect Actor decisions at selected times under teacher-forced RSSM history."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cmb_minigrid.data.dataset import read_manifest, validate_split_manifest
from cmb_minigrid.data.trajectory import TrajectoryRecord, read_jsonl
from cmb_minigrid.models import RSSM
from cmb_minigrid.models.policy import ActorPolicy
from cmb_minigrid.training.policy import _obs_tensor, load_policy
from cmb_minigrid.training.rssm import load_rssm

ACTION_NAMES = {0: "left", 1: "right", 2: "forward"}
DIAGNOSTIC_TIMES = (8, 9, 10, 11)


def _prediction(policy: ActorPolicy, features: torch.Tensor) -> dict[str, Any]:
    logits = policy(features)
    probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    predicted = int(logits.argmax(-1).item())
    return {
        "predicted_action": predicted,
        "predicted_action_name": ACTION_NAMES[predicted],
        "logits": logits[0].cpu().tolist(),
        "probabilities": probabilities,
        "confidence": float(max(probabilities)),
    }


def _local_observation(record: TrajectoryRecord, t: int) -> dict[str, Any]:
    obs = record.agent_obs[t]
    image = obs.image.tolist()
    return {
        "direction": int(obs.direction),
        "task_id": obs.task_id,
        "image": image,
        "object_channel": [[int(cell[0]) for cell in row] for row in image],
        "color_channel": [[int(cell[1]) for cell in row] for row in image],
        "state_channel": [[int(cell[2]) for cell in row] for row in image],
    }


@torch.no_grad()
def diagnose_record(
    record: TrajectoryRecord,
    rssm_a: RSSM,
    rssm_b: RSSM,
    policy_a: ActorPolicy,
    policy_b: ActorPolicy,
    *,
    times: tuple[int, ...] = DIAGNOSTIC_TIMES,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if any(t < 0 or t > record.handoff_t for t in times):
        raise ValueError("diagnostic times must be within the pre-handoff window")
    max_time = max(times)
    state_a = rssm_a.initial(1)
    state_b = rssm_b.initial(1)
    rows: list[dict[str, Any]] = []
    for t in range(max_time + 1):
        obs = _obs_tensor(record.agent_obs[t], device)
        previous = None if t == 0 else torch.tensor(
            [record.actions[t - 1]], dtype=torch.long, device=device
        )
        state_a = rssm_a.observe_step(state_a, previous, obs, sample=False)
        state_b = rssm_b.observe_step(state_b, previous, obs, sample=False)
        if t not in times:
            continue
        reset_state_b = rssm_b.observe_step(
            rssm_b.initial(1), None, obs, sample=False
        )
        expert = int(record.expert_actions[t])
        a = _prediction(policy_a, rssm_a.features(state_a))
        b = _prediction(policy_b, rssm_b.features(state_b))
        reset = _prediction(policy_b, rssm_b.features(reset_state_b))
        row = {
            "t": t,
            "expert_action": expert,
            "expert_action_name": ACTION_NAMES[expert],
            "local_observation": _local_observation(record, t),
            "a_full_history": {
                **a,
                "correct": a["predicted_action"] == expert,
            },
            "b_full_history": {
                **b,
                "correct": b["predicted_action"] == expert,
            },
            "reset_b": {
                **reset,
                "correct": reset["predicted_action"] == expert,
            },
            "b_full_history_vs_reset_b": {
                "same_prediction": b["predicted_action"] == reset["predicted_action"],
                "full_history_correct_reset_wrong": (
                    b["predicted_action"] == expert and reset["predicted_action"] != expert
                ),
                "full_history_wrong_reset_correct": (
                    b["predicted_action"] != expert and reset["predicted_action"] == expert
                ),
            },
        }
        rows.append(row)
    return {
        "episode_id": record.episode_id,
        "pair_id": record.pair_id,
        "split": None,
        "map_family_id": record.map_family_id,
        "handoff_t": record.handoff_t,
        "times": list(times),
        "rows": rows,
    }


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for row in report["rows"]:
            row = dict(row)
            row["map_family_id"] = report["map_family_id"]
            row["split"] = report["split"]
            cells[(report["map_family_id"], row["t"])].append(row)
    summary: dict[str, Any] = {}
    for (family, t), rows in sorted(cells.items()):
        entry: dict[str, Any] = {
            "split": rows[0]["split"],
            "map_family_id": family,
            "t": t,
            "episodes": len(rows),
        }
        for mode in ("a_full_history", "b_full_history", "reset_b"):
            entry[f"{mode}_accuracy"] = float(np.mean([
                row[mode]["correct"] for row in rows
            ]))
        entry["b_full_history_correct_reset_wrong"] = sum(
            row["b_full_history_vs_reset_b"]["full_history_correct_reset_wrong"]
            for row in rows
        )
        entry["b_full_history_wrong_reset_correct"] = sum(
            row["b_full_history_vs_reset_b"]["full_history_wrong_reset_correct"]
            for row in rows
        )
        entry["a_error_predicted_action_counts"] = {
            ACTION_NAMES[action]: sum(
                not row["a_full_history"]["correct"]
                and row["a_full_history"]["predicted_action"] == action
                for row in rows
            )
            for action in ACTION_NAMES
        }
        entry["b_error_predicted_action_counts"] = {
            ACTION_NAMES[action]: sum(
                not row["b_full_history"]["correct"]
                and row["b_full_history"]["predicted_action"] == action
                for row in rows
            )
            for action in ACTION_NAMES
        }
        summary[f"{family}:t{t}"] = entry
    return summary


def diagnose(
    *,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    policy_a: str | Path,
    policy_b: str | Path,
    input_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    times: tuple[int, ...] = DIAGNOSTIC_TIMES,
    device: str = "cpu",
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    manifest = read_manifest(manifest_path)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise ValueError("invalid diagnostic dataset: " + "; ".join(errors))
    selected = [record for record in records if record.episode_role == "expert"]
    rssm_a, _ = load_rssm(checkpoint_a, device)
    rssm_b, _ = load_rssm(checkpoint_b, device)
    actor_a, _ = load_policy(policy_a, device)
    actor_b, _ = load_policy(policy_b, device)
    reports = []
    for record in selected:
        report = diagnose_record(
            record, rssm_a, rssm_b, actor_a, actor_b, times=times, device=device
        )
        report["split"] = manifest.assignment[record.episode_id]
        reports.append(report)
    output = {
        "times": list(times),
        "episodes": len(reports),
        "reports": reports,
        "summary": _aggregate(reports),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "policy_error_diagnosis.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Actor errors at t=8..11")
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--policy-a", type=Path, required=True)
    parser.add_argument("--policy-b", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/samples/step3_trajectories.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--times", nargs="+", type=int, default=list(DIAGNOSTIC_TIMES))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = diagnose(
            checkpoint_a=args.checkpoint_a,
            checkpoint_b=args.checkpoint_b,
            policy_a=args.policy_a,
            policy_b=args.policy_b,
            input_path=args.input,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            times=tuple(args.times),
            device=args.device,
        )
        print(json.dumps({
            "episodes": report["episodes"],
            "times": report["times"],
            "summary": report["summary"],
        }, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"policy_error_diagnosis_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
