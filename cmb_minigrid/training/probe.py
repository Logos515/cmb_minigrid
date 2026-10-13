"""Frozen linear cue probes on strict pairs; never part of an RSSM forward path."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cmb_minigrid.data.dataset import (
    read_manifest,
    validate_split_manifest,
)
from cmb_minigrid.data.trajectory import read_jsonl, validate_counterfactual_records
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.training.rssm import fingerprint, load_rssm, model_batch


def observation_features(obs):
    """Raw public categorical observation baseline, including direction."""
    image = torch.cat([F.one_hot(obs["image"][..., c].long(), n).flatten(1)
                       for c, n in enumerate((11, 6, 3))], -1)
    return torch.cat([image, F.one_hot(obs["direction"].long(), 4)], -1).float()


@torch.no_grad()
def extract_pair_features(model, records, *, recent_k=8):
    """Consume only each expert pair's prefix through handoff; no future inputs."""
    if recent_k <= 0:
        raise ValueError("recent_k must be positive")
    model.eval().requires_grad_(False)
    pairs = {}
    for record in records:
        if record.episode_role == "expert":
            pairs.setdefault(record.pair_id, []).append(record)
    features = {name: [] for name in (
        "full_history", "current_state", "recent_state", "early_state",
        "current_observation", "early_observation")}
    labels, pair_ids = [], []
    device = next(model.parameters()).device
    for pair_id, endpoints in sorted(pairs.items()):
        endpoints.sort(key=lambda record: record.privileged_state[0]["hidden_label"])
        if len(endpoints) != 2:
            raise ValueError(f"incomplete expert pair: {pair_id}")
        t = endpoints[0].handoff_t
        if t < recent_k or endpoints[1].handoff_t != t:
            raise ValueError("pair handoff cannot support the requested recent window")
        batch = model_batch(endpoints, device)
        source = batch["agent_obs"]
        obs = {"image": source["image"][:, :t + 1],
               "direction": source["direction"][:, :t + 1],
               "task_id": [row[:t + 1] for row in source["task_id"]]}
        current = {"image": obs["image"][:, t], "direction": obs["direction"][:, t],
                   "task_id": [row[t] for row in obs["task_id"]]}
        early = {"image": obs["image"][:, 0], "direction": obs["direction"][:, 0],
                 "task_id": [row[0] for row in obs["task_id"]]}
        # Equal recent actions/observations must produce identical reset states.
        recent = {"image": obs["image"][:, t - recent_k:],
                  "direction": obs["direction"][:, t - recent_k:],
                  "task_id": [row[t - recent_k:] for row in obs["task_id"]]}
        actions = batch["actions"][:, :t]
        if (not torch.equal(current["image"][0], current["image"][1])
                or not torch.equal(recent["image"][0], recent["image"][1])
                or not torch.equal(recent["direction"][0], recent["direction"][1])
                or not torch.equal(actions[0], actions[1])):
            raise ValueError("counterfactual public prefix/recent equality failed")
        full = model.observe(obs, actions, sample=False)
        reset = model.observe_step(model.initial(2), None, current, sample=False)
        replay = model.observe(recent, actions[:, t - recent_k:], sample=False)
        values = {
            "full_history": model.features(full.states[t]),
            "early_state": model.features(full.states[0]),
            "current_state": model.features(reset),
            "recent_state": model.features(replay.states[-1]),
            "current_observation": observation_features(current),
            "early_observation": observation_features(early),
        }
        for name, value in values.items():
            features[name].append(value.cpu())
        labels.extend(int(record.privileged_state[0]["hidden_label"] == "right_blocked")
                      for record in endpoints)
        pair_ids.append(pair_id)
    if not labels:
        raise ValueError("probe split contains no expert counterfactual pairs")
    return {name: torch.cat(value) for name, value in features.items()}, torch.tensor(labels), pair_ids


def fit_linear_probe(train_x, train_y, *, iterations=100, l2=0.01):
    """Train-only standardization and logistic fit; no validation-based tuning."""
    if iterations <= 0 or l2 <= 0:
        raise ValueError("probe iterations and l2 must be positive")
    if train_x.requires_grad or train_y.requires_grad:
        raise ValueError("probe inputs must be detached from the RSSM")
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-3)
    x = (train_x - mean) / scale
    probe = nn.Linear(x.shape[1], 2)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=iterations,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(x), train_y) + l2 * probe.weight.square().sum()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite probe loss")
        loss.backward()
        return loss

    optimizer.step(closure)
    probe.eval().requires_grad_(False)
    return probe, mean, scale


@torch.no_grad()
def probe_metrics(probe, mean, scale, x, y):
    prediction = probe((x - mean) / scale).argmax(-1)
    correct = prediction == y
    return {"accuracy": correct.float().mean().item(), "n": len(y),
            "both_endpoints_correct": correct.reshape(-1, 2).all(-1).float().mean().item(),
            "mean_pair_feature_distance": (x[::2] - x[1::2]).norm(dim=-1).mean().item()}


def run_probes(checkpoint_a, checkpoint_b, input_path, manifest_path, output_dir,
               *, seed=42, device="cpu", iterations=100):
    set_seed(seed)
    torch.set_num_threads(1)
    records = read_jsonl(input_path)
    manifest = read_manifest(manifest_path)
    errors = validate_split_manifest(manifest, records=records)
    errors += validate_counterfactual_records(records)
    if errors:
        raise ValueError("invalid probe dataset: " + "; ".join(errors))
    split_records = {split: [record for record in records
                            if manifest.assignment[record.episode_id] == split]
                     for split in ("train", "val")}
    models, checkpoints = {}, {}
    for name, path in (("A", checkpoint_a), ("B", checkpoint_b)):
        models[name], checkpoints[name] = load_rssm(path, device)
        for field, source in (("input", input_path), ("manifest", manifest_path)):
            if checkpoints[name]["config"][f"{field}_sha256"] != fingerprint(source):
                raise ValueError(f"{name} checkpoint and probe {field} differ")
    if checkpoints["A"]["config"]["seed"] == checkpoints["B"]["config"]["seed"]:
        raise ValueError("A/B checkpoints must have independent initialization seeds")
    if (models["A"].config.deter, models["B"].config.deter) != (128, 256):
        raise ValueError("expected RSSM-A deter=128 and RSSM-B deter=256")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"seed": seed, "device": str(device), "probe_iterations": iterations,
              "probe_l2": 0.01, "recent_k": 8, "fit_split": "train", "eval_split": "val",
              "models": {}, "test_split_used": False,
              "evidence_boundary": "Cue decoding is diagnostic only; it does not establish "
                                   "sufficient memory, future prediction or closed-loop control."}
    for name, model in models.items():
        train_features, train_y, train_pairs = extract_pair_features(model, split_records["train"])
        val_features, val_y, val_pairs = extract_pair_features(model, split_records["val"])
        model_report = {"checkpoint": str(Path(checkpoint_a if name == "A" else checkpoint_b)),
                        "checkpoint_sha256": fingerprint(checkpoint_a if name == "A" else checkpoint_b),
                        "seed": checkpoints[name]["config"]["seed"],
                        "train_pairs": train_pairs, "val_pairs": val_pairs, "probes": {}}
        for feature_name, train_x in train_features.items():
            probe, mean, scale = fit_linear_probe(train_x, train_y, iterations=iterations)
            train_metrics = probe_metrics(probe, mean, scale, train_x, train_y)
            val_metrics = probe_metrics(probe, mean, scale, val_features[feature_name], val_y)
            model_report["probes"][feature_name] = {"train": train_metrics, "val": val_metrics}
            torch.save({"probe": probe.state_dict(), "mean": mean, "scale": scale,
                        "feature": feature_name, "rssm_checkpoint_sha256": model_report[
                            "checkpoint_sha256"]}, output_dir / f"probe_{name}_{feature_name}.pt")
        report["models"][name] = model_report
    (output_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
