"""Step 9 closed-loop takeover and History Swap contract tests."""

import csv
import math
from dataclasses import replace

import pytest
import torch

from cmb_minigrid.bridge.cmb import CMBBridge, CMBConfig, CMBMessage
from cmb_minigrid.contracts import RSSMState
from cmb_minigrid.data.dataset import split_records
from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.handoff.dataset import build_handoff_samples
from cmb_minigrid.handoff.initializers import (
    MapInit,
    OracleInit,
    ReplayInit,
    ResetInit,
    StateMap,
    restore_env,
)
from cmb_minigrid.handoff.runner import (
    CMBInit,
    HandoffRunner,
    main_tables,
    payload_bytes,
    run_history_swap,
    strip_timing,
    summarize_swap,
    swap_pairs,
    write_csv,
)
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
def world():
    torch.manual_seed(0)
    rssm_a = RSSM(RSSMConfig(deter=DETER_A, encoder_type="cnn")).eval().requires_grad_(False)
    rssm_b = RSSM(RSSMConfig(deter=DETER_B, encoder_type="cnn")).eval().requires_grad_(False)
    policy = ActorPolicy(rssm_b.feature_dim, 16).eval().requires_grad_(False)
    records = collect_records(
        pair_count=2, seed=42,
        map_family_ids=["hidden_history_nav_v1", "hidden_history_nav_v2"],
        include_exploration=True)
    samples, _ = build_handoff_samples(records, split_records(records), rssm_a, rssm_b)
    cfg = CMBConfig(a_feature_dim=rssm_a.feature_dim, b_deter=DETER_B, sender_hidden=16,
                    receiver_hidden=16)
    torch.manual_seed(1)
    bridge = CMBBridge(cfg).eval()
    expert = [s for s in samples if s.episode_role == "expert"]
    return rssm_a, rssm_b, policy, samples, expert, bridge


def make_methods(world):
    rssm_a, rssm_b, _, _, _, bridge = world
    torch.manual_seed(2)
    return [ResetInit(rssm_b), ReplayInit(rssm_b, 4), ReplayInit(rssm_b, 8),
            ReplayInit(rssm_b, 16),
            MapInit("linear", StateMap("linear", rssm_a.feature_dim, DETER_B), rssm_a, rssm_b),
            MapInit("mlp", StateMap("mlp", rssm_a.feature_dim, DETER_B, hidden=16,
                                    bottleneck=8), rssm_a, rssm_b),
            CMBInit(bridge, rssm_a, rssm_b), OracleInit(rssm_b)]


class SequencePolicy:
    """Test double: ignores features and replays a fixed action list (or a constant)."""

    def __init__(self, actions=None, constant=2):
        self.actions, self.constant, self.i = actions, constant, 0

    def act(self, features):
        action = self.constant if self.actions is None else self.actions[self.i]
        self.i += 1
        return torch.tensor([action])


def spy(model):
    calls = []
    original = model.observe_step

    def wrapped(prev_state, prev_action, obs, **kwargs):
        calls.append((obs["image"].clone(), None if prev_action is None else int(prev_action[0])))
        return original(prev_state, prev_action, obs, **kwargs)

    model.observe_step = wrapped
    return calls, lambda: delattr(model, "observe_step")


def test_every_method_starts_from_the_identical_snapshot_and_obs_t(world):
    rssm_a, rssm_b, policy, _, expert, _ = world
    methods = make_methods(world)
    runner = HandoffRunner(rssm_a, rssm_b, policy, methods)
    rows = [runner.run_sample(expert[0], m) for m in methods]
    assert {r["method"] for r in rows} == {m.name for m in methods}
    assert len({r["snapshot_sha"] for r in rows}) == 1 and len({r["obs_t_sha"] for r in rows}) == 1
    assert all(r["snapshot_restored_equal"] and r["t"] == 12 for r in rows)


def test_obs_t_is_not_fused_again_and_obs_t_plus_1_is_fused_after_each_action(world):
    rssm_a, rssm_b, policy, _, expert, _ = world
    sample = expert[0]
    runner = HandoffRunner(rssm_a, rssm_b, policy, make_methods(world))
    cases = [(OracleInit(rssm_b), 0),  # stored state: B runs no takeover filtering
             (MapInit("linear", StateMap("linear", rssm_a.feature_dim, DETER_B), rssm_a, rssm_b), 0),
             (ReplayInit(rssm_b, 4), 4), (ResetInit(rssm_b), 1)]
    for method, takeover_calls in cases:
        calls, undo = spy(rssm_b)
        try:
            row = runner.run_sample(sample, method)
        finally:
            undo()
        assert row["steps"] >= 2
        # takeover frames + one fusion per non-final post-action observation, nothing else
        assert len(calls) == takeover_calls + row["steps"] - 1, method.name
        # takeover fuses exactly the last N frames of the prefix, ending with obs_t, once
        for k in range(takeover_calls):
            image, prev = calls[k]
            assert torch.equal(image[0], sample.history.images[sample.t + 1 - takeover_calls + k])
            assert (prev is None) == (k == 0)
        # the first fusion after takeover is obs[t+1], reached by the executed action[t]
        env, _ = restore_env(sample)
        try:
            next_obs, *_ = env.step(row["actions"][0])
        finally:
            env.close()
        image, prev = calls[takeover_calls]
        assert torch.equal(image[0], torch.from_numpy(next_obs.image.copy())), method.name
        assert prev == row["actions"][0]
        assert all(prev is not None for _, prev in calls[takeover_calls:]), method.name


def test_same_method_and_seed_reproduce_exactly(world):
    rssm_a, rssm_b, policy, _, expert, _ = world
    methods = make_methods(world)
    for stochastic in (False, True):
        runs = [HandoffRunner(rssm_a, rssm_b, policy, methods, stochastic=stochastic,
                              seed=7).run(expert[:3]) for _ in range(2)]
        assert strip_timing(runs[0]) == strip_timing(runs[1]), stochastic


def test_forward_only_policy_collides_and_times_out_and_expert_policy_is_optimal(world):
    rssm_a, rssm_b, _, _, expert, _ = world
    sample = expert[0]
    method = OracleInit(rssm_b)
    bad = HandoffRunner(rssm_a, rssm_b, SequencePolicy(constant=2), [method]).run_sample(sample, method)
    assert not bad["success"] and bad["truncated"] and bad["steps"] == 64 - sample.t
    assert bad["collisions"] == bad["steps"] and bad["path_efficiency"] == 0.0
    assert not bad["first_branch_correct"] and bad["wrong_actions"] > 0
    script = [int(a) for a in sample.future.expert_actions]
    good = HandoffRunner(rssm_a, rssm_b, SequencePolicy(script), [method]).run_sample(sample, method)
    assert good["success"] and good["wrong_actions"] == 0 and good["collisions"] == 0
    assert good["steps"] == good["optimal_steps"] == len(script) and good["path_efficiency"] == 1.0
    assert good["first_branch_correct"] and good["reward"] > 0


def test_payload_bytes_accounting(world):
    rssm_a, *_ = world
    m = {x.name: x for x in make_methods(world)}
    t = 12
    assert payload_bytes(m["reset"], t) == 0
    assert payload_bytes(m["replay_4"], t) == 4 * 148 + 3
    assert payload_bytes(m["replay_16"], t) == 13 * 148 + 12 == payload_bytes(m["oracle_b"], t)
    assert payload_bytes(m["linear"], t) == rssm_a.feature_dim * 4
    assert payload_bytes(m["mlp"], t) == 8 * 4
    assert payload_bytes(m["cmb"], t) == 2


def test_tables_and_csv_match_rows(world, tmp_path):
    rssm_a, rssm_b, policy, _, expert, _ = world
    methods = make_methods(world)
    rows = HandoffRunner(rssm_a, rssm_b, policy, methods).run(expert)
    tables = main_tables(rows)
    assert {t["scope"] for t in tables} == {"family", "split", "all"}
    overall = {t["method"]: t for t in tables if t["scope"] == "all"}
    for m in methods:
        mine = [r for r in rows if r["method"] == m.name]
        assert overall[m.name]["n"] == len(mine)
        assert math.isclose(overall[m.name]["success_rate"], sum(r["success"] for r in mine) / len(mine))
    path = write_csv(tables, tmp_path / "t.csv")
    with path.open() as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == len(tables) and "success_rate" in loaded[0]


def tag_states(samples):
    """Give every sample a unique synthetic A state (a random-init A carries no cue)."""
    tagged = []
    for i, s in enumerate(samples):
        deter = torch.zeros_like(s.state_a.deter)
        deter[0, 0] = (i + 1) / 100
        tagged.append(replace(s, state_a=RSSMState(deter, s.state_a.stoch, s.state_a.stoch_logits)))
    return tagged


class LookupPolicy:
    """Reads the sample tag out of the B state and returns that sample's expert handoff action."""

    def __init__(self, samples):
        self.actions = [int(s.future.expert_actions[0]) for s in samples]

    def act(self, features):
        index = round(float(features[0, 0]) * 100) - 1
        return torch.tensor([self.actions[index] if 0 <= index < len(self.actions) else 2])


class IndexBridge:
    """Message = tagged sample index; Receiver emits a legal B state carrying only that tag."""

    def __init__(self, rssm_b, samples):
        self.samples, self.rssm_b = samples, rssm_b
        self.received_obs = []

    def eval(self):
        return self

    def send(self, features, sample=False):
        i = round(float(features[0, 0]) * 100) - 1
        onehot = torch.zeros(1, 4, 16)
        onehot[0, 0, i // 16] = 1
        onehot[0, 1, i % 16] = 1
        onehot[0, 2:, 0] = 1
        return CMBMessage(torch.zeros(1, 4, 16), onehot)

    def receive(self, message, obs, rssm_b, sample=False):
        self.received_obs.append((obs["image"].clone(), obs["direction"].clone()))
        i = int(message.indices[0, 0]) * 16 + int(message.indices[0, 1])
        base = self.rssm_b.initial(1)
        deter = base.deter.clone()
        deter[0, 0] = (i + 1) / 100
        return RSSMState(deter, base.stoch, base.stoch_logits)


def test_history_swap_exchanges_only_the_message(world):
    rssm_a, rssm_b, _, _, expert, _ = world
    tagged = tag_states(expert)
    bridge = IndexBridge(rssm_b, tagged)
    runner = HandoffRunner(rssm_a, rssm_b, LookupPolicy(tagged), [])
    rows = run_history_swap(runner, CMBInit(bridge, rssm_a, rssm_b), tagged)
    pairs = swap_pairs(tagged)
    assert len(pairs) == len(tagged)
    by_own: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_own.setdefault(r["own_episode_id"], {})[r["condition"]] = r
    for own, donor in pairs:
        conds = by_own[own.episode_id]
        assert {"own", "swap", "zero", "same_cue"} == set(conds)  # two pairs per family -> control exists
        # environment snapshot and current observation are identical in every condition
        assert len({c["snapshot_sha"] for c in conds.values()}) == 1
        assert len({c["obs_t_sha"] for c in conds.values()}) == 1
        assert all(c["snapshot_equal"] for c in conds.values())
        assert conds["own"]["first_action"] == int(own.future.expert_actions[0])
        assert conds["swap"]["first_action"] == int(donor.future.expert_actions[0])
        assert conds["swap"]["flipped_vs_own"] and conds["swap"]["message_differs_from_own"]
        assert not conds["own"]["flipped_vs_own"]
        assert not conds["same_cue"]["flipped_vs_own"]  # same cue, other history: no change
    # the Receiver saw exactly one obs_t per condition and it is the endpoints' shared obs_t
    assert len(bridge.received_obs) == len(rows)
    shared = {(o.history.images[-1].numpy().tobytes(), int(o.history.directions[-1]))
              for o in tagged}
    assert {(i[0].numpy().tobytes(), int(d[0])) for i, d in bridge.received_obs} <= shared
    summary = {s["condition"]: s for s in summarize_swap(rows) if s["map_family_id"].endswith("v1")}
    assert summary["swap"]["flipped_vs_own"] == 1.0 and summary["own"]["flipped_vs_own"] == 0.0
    assert summary["swap"]["follows_message_cue"] == 1.0


def test_zero_message_condition_keeps_environment_and_obs_t(world):
    rssm_a, rssm_b, policy, _, expert, bridge = world
    rows = run_history_swap(HandoffRunner(rssm_a, rssm_b, policy, []),
                            CMBInit(bridge, rssm_a, rssm_b), expert)
    zero = [r for r in rows if r["condition"] == "zero"]
    assert len(zero) == len(swap_pairs(expert))
    for r in zero:
        own = next(x for x in rows if x["own_episode_id"] == r["own_episode_id"]
                   and x["condition"] == "own")
        assert r["snapshot_sha"] == own["snapshot_sha"] and r["obs_t_sha"] == own["obs_t_sha"]
        assert r["message"] is None and r["snapshot_equal"] and not r["follows_message_cue"]


def test_history_swap_with_a_real_cmb_runs_with_integrity(world):
    rssm_a, rssm_b, policy, _, expert, bridge = world
    cmb = CMBInit(bridge, rssm_a, rssm_b)
    seen = []
    original = bridge.receive

    def recording(message, obs, rssm_b_, **kwargs):
        seen.append(obs["image"].clone())
        return original(message, obs, rssm_b_, **kwargs)

    bridge.receive = recording
    try:
        rows = run_history_swap(HandoffRunner(rssm_a, rssm_b, policy, []), cmb, expert)
    finally:
        del bridge.receive
    assert rows and all(r["snapshot_equal"] for r in rows)
    assert len(seen) == len(rows)
    for own, _ in swap_pairs(expert):  # every condition for an endpoint saw its own obs_t
        n = sum(r["own_episode_id"] == own.episode_id for r in rows)
        assert n >= 2
