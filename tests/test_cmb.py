"""Step 8 CMB Bridge contract tests."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from cmb_minigrid.bridge import CMBBridge, CMBConfig, load_cmb, save_cmb
from cmb_minigrid.bridge.training import (
    compare_to_reset,
    compute_losses,
    evaluate_methods,
    make_batch,
    overhead_table,
    rollout,
    train_cmb,
)
from cmb_minigrid.data.dataset import split_records
from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.handoff.dataset import build_handoff_samples, save_handoff_dataset
from cmb_minigrid.handoff.initializers import StateMap, validate_b_state
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.models.policy import ActorPolicy

DETER_A, DETER_B = 32, 48


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    torch.manual_seed(0)
    rssm_a = RSSM(RSSMConfig(deter=DETER_A, encoder_type="cnn")).eval().requires_grad_(False)
    rssm_b = RSSM(RSSMConfig(deter=DETER_B, encoder_type="cnn")).eval().requires_grad_(False)
    policy = ActorPolicy(rssm_b.feature_dim, 16).eval().requires_grad_(False)
    records = collect_records(
        pair_count=1, seed=42,
        map_family_ids=["hidden_history_nav_v1", "hidden_history_nav_v2", "hidden_history_nav_v3"],
        include_exploration=True)
    samples, _ = build_handoff_samples(records, split_records(records), rssm_a, rssm_b)
    root = tmp_path_factory.mktemp("cmb")
    paths = {"handoff": save_handoff_dataset(samples, root / "h.pt")}
    for name, model, deter in (("a", rssm_a, DETER_A), ("b", rssm_b, DETER_B)):
        paths[name] = root / f"{name}.pt"
        torch.save({"schema_version": 1, "config": {"rssm": asdict(model.config), "seed": 1},
                    "model": model.state_dict()}, paths[name])
    paths["policy"] = root / "p.pt"
    torch.save({"schema_version": 1, "policy_config": {"feature_dim": rssm_b.feature_dim,
                                                       "hidden_dim": 16},
                "model": policy.state_dict()}, paths["policy"])
    cfg = CMBConfig(a_feature_dim=rssm_a.feature_dim, b_deter=DETER_B, sender_hidden=16,
                    receiver_hidden=16)
    return rssm_a, rssm_b, policy, samples, cfg, paths, root


def batch_of(world, n=8):
    rssm_a, _, _, samples, _, _, _ = world
    return make_batch(samples[:n], rssm_a, "cpu")


def test_state_b_hat_is_legal_and_consumable_by_policy_and_rssm(world):
    _, rssm_b, policy, _, cfg, _, _ = world
    batch = batch_of(world)
    bridge = CMBBridge(cfg).eval()
    state, message = bridge(batch.features_a, batch.obs, rssm_b)
    validate_b_state(rssm_b, state)
    assert message.indices.shape == (8, 4) and message.indices.max() < 16
    assert state.deter.shape == (8, DETER_B) and state.stoch.shape == (8, 16, 16)
    assert policy(rssm_b.features(state)).shape == (8, 3)
    nxt = rssm_b.imagine_step(state, batch.actions[:, 0])
    assert torch.isfinite(nxt.deter).all()
    # Continuing with obs[t+1] is the only observe_step B performs after the handoff.
    obs1 = {"image": batch.future_images[:, 0], "direction": torch.zeros(8, dtype=torch.long),
            "task_id": batch.obs["task_id"]}
    assert torch.isfinite(rssm_b.observe_step(state, batch.actions[:, 0], obs1).deter).all()


def test_cmb_never_calls_observe_step_on_obs_t(world):
    _, rssm_b, _, _, cfg, _, _ = world
    batch = batch_of(world)

    def boom(*args, **kwargs):
        raise AssertionError("observe_step must not be called by the Receiver")

    rssm_b.observe_step, original = boom, rssm_b.observe_step
    try:
        CMBBridge(cfg)(batch.features_a, batch.obs, rssm_b)
    finally:
        rssm_b.observe_step = original
        del rssm_b.observe_step


@pytest.mark.parametrize("stage", [1, 2])
def test_gradients_reach_cmb_through_frozen_student_branch_only(world, stage):
    rssm_a, rssm_b, policy, _, cfg, _, _ = world
    batch = batch_of(world)
    batch.b_star.deter.requires_grad_(True)  # a teacher that wrongly required grad must get none
    bridge = CMBBridge(cfg).train()
    losses = compute_losses(bridge, batch, rssm_b, policy, stage=stage)
    losses["total"].backward()
    for part in (bridge.sender, bridge.receiver):
        norms = [p.grad.norm() for p in part.parameters()]
        assert all(torch.isfinite(n) for n in norms) and sum(norms) > 0
    assert batch.b_star.deter.grad is None  # teacher branch is detached
    for module in (rssm_a, rssm_b, policy):
        assert all(p.grad is None and not p.requires_grad for p in module.parameters())


def test_multistep_term_alone_backpropagates_through_frozen_imagination(world):
    _, rssm_b, policy, _, cfg, _, _ = world
    bridge = CMBBridge(cfg)
    losses = compute_losses(bridge, batch_of(world), rssm_b, policy, stage=2)
    losses["multistep"].backward()
    assert sum(p.grad.norm() for p in bridge.sender.parameters()) > 0
    assert sum(p.grad.norm() for p in bridge.receiver.parameters()) > 0


def test_real_future_observations_are_not_a_training_signal_and_rollouts_ignore_them(world):
    _, rssm_b, policy, _, cfg, _, _ = world
    batch = batch_of(world)
    bridge = CMBBridge(cfg)
    base = compute_losses(bridge, batch, rssm_b, policy, stage=2)["total"]
    batch.future_images = (batch.future_images + 1) % 3
    assert torch.equal(base, compute_losses(bridge, batch, rssm_b, policy, stage=2)["total"])
    states = rollout(rssm_b, batch.b_star, batch, 5)
    assert len(states) == 5


def test_message_matters_and_checkpoint_round_trips(world, tmp_path):
    _, rssm_b, _, _, cfg, _, _ = world
    batch = batch_of(world)
    bridge = CMBBridge(cfg).eval()
    state, message = bridge(batch.features_a, batch.obs, rssm_b)
    swapped = replace(message, onehot=message.onehot.roll(1, dims=0))
    other = bridge.receive(swapped, batch.obs, rssm_b)
    assert not torch.equal(state.deter, other.deter)  # same obs_t, different message
    save_cmb(bridge, tmp_path / "c.pt", {"stage": 1})
    loaded, payload = load_cmb(tmp_path / "c.pt")
    again, _ = loaded(batch.features_a, batch.obs, rssm_b)
    assert payload["stage"] == 1 and torch.equal(state.deter, again.deter)
    assert torch.equal(state.stoch, again.stoch)
    with pytest.raises(Exception, match="match RSSM-B"):
        CMBBridge(replace(cfg, b_deter=DETER_B + 1)).receive(message, batch.obs, rssm_b)


def test_single_batch_overfit_and_seed_reproducibility(world):
    _, rssm_b, policy, _, cfg, _, _ = world
    batch = batch_of(world)

    def run():
        torch.manual_seed(3)
        bridge = CMBBridge(cfg)
        optimizer = torch.optim.Adam(bridge.parameters(), lr=3e-3)
        first = last = None
        for _ in range(40):
            loss = compute_losses(bridge, batch, rssm_b, policy, stage=2)["total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            first = loss.item() if first is None else first
            last = loss.item()
        return first, last

    first, last = run()
    assert last < first
    assert run() == (first, last)


def test_overhead_table_counts_params_and_payload(world):
    _, _, _, _, cfg, _, _ = world
    bridge = CMBBridge(cfg)
    mlp = StateMap("mlp", cfg.a_feature_dim, DETER_B, bottleneck=64)
    table = overhead_table(bridge, None, mlp)
    assert table["cmb"]["total"] == sum(p.numel() for p in bridge.parameters())
    assert table["cmb"]["payload_bits"] == 16 and table["cmb"]["payload_bytes_packed"] == 2
    assert table["mlp"]["payload_bytes_float32"] == 256 and table["mlp"]["total"] == mlp.num_parameters


def test_evaluate_methods_and_end_to_end_training_smoke(world, tmp_path):
    rssm_a, rssm_b, policy, samples, cfg, paths, _ = world
    val = [s for s in samples if s.split == "val"]
    results = evaluate_methods(val, rssm_a, rssm_b, policy, CMBBridge(cfg))
    assert {"reset", "replay_4", "replay_8", "oracle_b", "cmb"} <= set(results)
    assert results["oracle_b"]["action_kl_to_oracle"] == 0 and results["oracle_b"]["deter_l2"] == 0
    verdict = compare_to_reset(results)
    assert all(isinstance(v["cmb_not_worse"], bool) for v in verdict.values())
    report = train_cmb(
        handoff_path=paths["handoff"], checkpoint_a=paths["a"], checkpoint_b=paths["b"],
        policy_b=paths["policy"], output_dir=tmp_path / "run", seed=1, config=cfg,
        stage1_steps=3, stage2_steps=3, batch_size=4, val_every=2)
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "metrics.jsonl").exists()
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert all(r["sender_grad_norm"] > 0 and r["receiver_grad_norm"] > 0
               for r in rows if r["event"] == "train")
    assert set(report["evaluation"]) == {"val", "test"} and "cmb" in report["evaluation"]["test"]["methods"]


def test_loss_ablation_switches(world, tmp_path):
    _, _, _, _, cfg, paths, _ = world
    report = train_cmb(
        handoff_path=paths["handoff"], checkpoint_a=paths["a"], checkpoint_b=paths["b"],
        policy_b=paths["policy"], output_dir=tmp_path / "state_only", seed=1, config=cfg,
        stage1_steps=3, stage2_steps=0, batch_size=4, val_every=2, action_weight=0.0)
    assert set(report["stages"]) == {"stage1"} and (tmp_path / "state_only" / "best.pt").exists()
    assert report["meta"]["action_weight"] == 0.0 and report["meta"]["stage2_steps"] == 0
    rows = [json.loads(line) for line in (tmp_path / "state_only" / "metrics.jsonl").read_text()
            .splitlines()]
    assert all("multistep" not in r for r in rows)
    # with the action weight 0 the total equals the state term alone
    train_rows = [r for r in rows if r["event"] == "train"]
    assert all(abs(r["total"] - r["state_aux"]) < 1e-6 for r in train_rows)
