"""Read-only diagnosis of color response lost by deterministic categorical modes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from cmb_minigrid.data.dataset import read_manifest, validate_split_manifest
from cmb_minigrid.data.trajectory import read_jsonl, validate_counterfactual_records
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.runtime import select_device
from cmb_minigrid.training.rssm import fingerprint, load_rssm, model_batch


def tensor_diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = (left.float() - right.float()).detach()
    if not torch.isfinite(delta).all():
        raise ValueError("non-finite diagnostic tensor difference")
    return {
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "l2": float(delta.square().sum().sqrt()),
    }


@torch.no_grad()
def diagnose_model(model, records, device):
    if len(records) != 2 or any(record.horizon < 1 for record in records):
        raise ValueError("diagnosis requires two endpoints containing obs[0] and obs[1]")
    if model.training:
        raise ValueError("diagnosis requires an eval-mode model")
    batch = model_batch(records, device)
    obs = batch["agent_obs"]
    actions = batch["actions"]

    state = model.initial(2)
    result = {}

    for t in (0, 1):
        image = obs["image"][:, t]
        direction = obs["direction"][:, t]

        previous_action = None if t == 0 else actions[:, t - 1]
        if t == 1 and not torch.equal(previous_action[0], previous_action[1]):
            raise ValueError("paired action[0] must match to isolate cue response")
        step_obs = {"image": image, "direction": direction,
                    "task_id": [row[t] for row in obs["task_id"]]}
        captured = {}

        def capture(name, destination=captured):
            def hook(module, inputs, output):
                destination[name] = output.detach()
            return hook

        # Hooks observe the authoritative forward path; they never replace outputs.
        handles = [model.encoder.register_forward_hook(capture("encoder")),
                   model.posterior.register_forward_hook(capture("posterior"))]
        try:
            next_state = model.observe_step(state, previous_action, step_obs, sample=False)
        finally:
            for handle in handles:
                handle.remove()
        embedding = captured["encoder"]
        posterior_raw = captured["posterior"]
        posterior_logits = model._logits(posterior_raw)
        posterior_probabilities = posterior_logits.exp()

        argmax_class = posterior_logits.argmax(dim=-1)
        stochastic = F.one_hot(
            argmax_class,
            model.config.stoch_classes,
        ).to(posterior_logits.dtype)
        deter = model._transition(state, previous_action)
        for actual, expected in ((next_state.deter, deter),
                                 (next_state.stoch, stochastic),
                                 (next_state.stoch_logits, posterior_logits)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        top_two = posterior_probabilities.topk(2, dim=-1).values
        margins = top_two[..., 0] - top_two[..., 1]
        perturbation = (posterior_probabilities[0] - posterior_probabilities[1]).abs().amax(-1)

        result[f"t={t}"] = {
            "observation_difference": {
                "changed_entries_per_channel": (image[0] != image[1]).sum((0, 1)).tolist(),
                "direction_equal": bool(direction[0] == direction[1]),
                "task_id_equal": step_obs["task_id"][0] == step_obs["task_id"][1],
                "previous_action": None if previous_action is None else previous_action.tolist(),
            },
            "encoder_difference": tensor_diff(
                embedding[0],
                embedding[1],
            ),
            "posterior_logits_difference": tensor_diff(
                posterior_logits[0],
                posterior_logits[1],
            ),
            "posterior_probabilities_difference": tensor_diff(
                posterior_probabilities[0],
                posterior_probabilities[1],
            ),
            "categorical_argmax": {
                "changed_group_count": int(
                    (argmax_class[0] != argmax_class[1]).sum()
                ),
                "total_group_count": int(argmax_class.shape[-1]),
                "world_0": argmax_class[0].tolist(),
                "world_1": argmax_class[1].tolist(),
            },
            "stochastic_state_difference": tensor_diff(
                stochastic[0],
                stochastic[1],
            ),
            "deterministic_state_difference": tensor_diff(
                deter[0],
                deter[1],
            ),
            "raw_posterior_difference": tensor_diff(
                posterior_raw[0],
                posterior_raw[1],
            ),
            "probability_margins": {
                "world_0_top1_minus_top2": margins[0].tolist(),
                "world_1_top1_minus_top2": margins[1].tolist(),
                "max_probability_perturbation_per_group": perturbation.tolist(),
                "groups_with_world_0_margin_gt_twice_perturbation": int(
                    (margins[0] > 2 * perturbation).sum()
                ),
            },
            "observe_step_exact_match": True,
        }
        state = next_state

    return result


def select_pair(records, pair_id: str | None):
    grouped = defaultdict(list)

    for record in records:
        if record.episode_role == "expert":
            grouped[record.pair_id].append(record)

    if pair_id is None:
        if not grouped:
            raise ValueError("selected split contains no expert pairs")
        pair_id = min(grouped)

    endpoints = grouped.get(pair_id, [])
    if len(endpoints) != 2:
        raise ValueError(
            f"pair {pair_id!r} must contain exactly two expert endpoints"
        )

    endpoints.sort(
        key=lambda record: record.privileged_state[0]["hidden_label"]
    )
    return endpoints


def interpret(diagnostic):
    early, following = diagnostic["t=0"], diagnostic["t=1"]
    response = (early["encoder_difference"]["max_abs"] > 0
                and early["posterior_probabilities_difference"]["max_abs"] > 0)
    discarded = (response and early["categorical_argmax"]["changed_group_count"] == 0
                 and early["stochastic_state_difference"]["max_abs"] == 0)
    return {
        "encoder_and_posterior_respond_at_t0": response,
        "nonzero_response_discarded_by_argmax_at_t0": discarded,
        "deterministic_states_equal_at_t1":
            following["deterministic_state_difference"]["max_abs"] == 0,
        "scope": "This pair/checkpoint with sample=False only; stored posterior log "
                 "probabilities may differ even when deter+hard-stoch features coincide. "
                 "t=0 deter is computed before seeing obs[0]. No claim about sampled "
                 "filtering, all pairs, or the cause of weak cue learning.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose RSSM encoder/posterior categorical response at t=0,1"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples/step3_trajectories.jsonl"),
    )
    parser.add_argument("--pair-id")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/samples/step3_manifest.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)
    device = select_device(args.device)
    records = read_jsonl(args.input)
    manifest = read_manifest(args.manifest)
    errors = validate_split_manifest(manifest, records=records)
    if errors:
        raise ValueError("invalid split manifest: " + "; ".join(errors))
    pair = select_pair([record for record in records
                        if manifest.assignment[record.episode_id] == args.split], args.pair_id)
    errors = validate_counterfactual_records(pair)
    if errors:
        raise ValueError("invalid counterfactual pair: " + "; ".join(errors))

    model, checkpoint = load_rssm(args.checkpoint, device)
    for field, path in (("input", args.input), ("manifest", args.manifest)):
        if checkpoint["config"][f"{field}_sha256"] != fingerprint(path):
            raise ValueError(f"checkpoint {field} fingerprint differs")
    model.eval()

    with torch.no_grad():
        diagnostic = diagnose_model(model, pair, device)

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": checkpoint["config"]["seed"],
        "device": str(device),
        "seed": args.seed,
        "split": args.split,
        "input": str(args.input),
        "manifest": str(args.manifest),
        "selected_step": checkpoint["step"],
        "pair_id": pair[0].pair_id,
        "hidden_labels": [
            record.privileged_state[0]["hidden_label"]
            for record in pair
        ],
        "results": diagnostic,
        "interpretation": interpret(diagnostic),
    }

    print(json.dumps(report, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
