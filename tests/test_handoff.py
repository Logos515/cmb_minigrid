"""Step 7 handoff dataset and initializer contract tests."""

from dataclasses import replace

import pytest
import torch

from cmb_minigrid.contracts import ContractError, RSSMState
from cmb_minigrid.data.dataset import split_records
from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.handoff.dataset import (
    build_handoff_samples,
    filter_history,
    load_handoff_dataset,
    save_handoff_dataset,
)
from cmb_minigrid.handoff.initializers import (
    MapInit,
    OracleInit,
    ReplayInit,
    ResetInit,
    StateMap,
    collect_aligned_states,
    continue_from_handoff,
    fit_state_map,
    initialize_all,
    matched_hidden_dim,
    restore_env,
    validate_b_state,
)
from cmb_minigrid.models import RSSM, RSSMConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    rssm_a = RSSM(RSSMConfig(deter=32, encoder_type="cnn")).eval()
    rssm_b = RSSM(RSSMConfig(deter=48, encoder_type="cnn")).eval()
    records = collect_records(
        pair_count=1, seed=42,
        map_family_ids=["hidden_history_nav_v1", "hidden_history_nav_v2", "hidden_history_nav_v3"],
        include_exploration=True)
    manifest = split_records(records)
    samples, info = build_handoff_samples(records, manifest, rssm_a, rssm_b)
    return rssm_a, rssm_b, records, manifest, samples, info


def spy(model):
    """Record (image, direction, prev_action) of every observe_step call."""
    calls = []
    original = model.observe_step

    def wrapped(prev_state, prev_action, obs, **kwargs):
        calls.append((obs["image"].clone(), int(obs["direction"][0]),
                      None if prev_action is None else int(prev_action[0])))
        return original(prev_state, prev_action, obs, **kwargs)

    model.observe_step = wrapped
    return calls, lambda: delattr(model, "observe_step")


def test_samples_cover_every_handoff_record_and_store_required_fields(world):
    _, _, records, _, samples, info = world
    assert len(samples) + len(info["skipped_without_handoff"]) == len(records)
    s = samples[0]
    rec = next(r for r in records if r.episode_id == s.episode_id)
    assert s.t == rec.handoff_t and s.pair_id == rec.pair_id
    assert s.history.images.shape == (s.t + 1, 7, 7, 3) and s.history.actions.shape == (s.t,)
    assert s.future.images.shape[0] == rec.horizon - s.t  # obs[t+1:]
    assert s.future.actions.tolist() == list(rec.actions[s.t:])
    assert s.env_snapshot == rec.env_snapshot
    assert s.state_a.deter.shape == (1, 32) and s.state_b_star.deter.shape == (1, 48)


def test_dataset_round_trip(world, tmp_path):
    _, _, _, _, samples, _ = world
    path = save_handoff_dataset(samples, tmp_path / "h.pt", meta={"x": 1})
    loaded, meta = load_handoff_dataset(path)
    assert meta == {"x": 1} and len(loaded) == len(samples)
    for a, b in zip(samples, loaded, strict=True):
        assert (a.episode_id, a.pair_id, a.split, a.t) == (b.episode_id, b.pair_id, b.split, b.t)
        assert a.env_snapshot == b.env_snapshot
        assert torch.equal(a.state_b_star.deter, b.state_b_star.deter)
        assert torch.equal(a.future.images, b.future.images)
        assert torch.equal(a.history.images, b.history.images)


@pytest.mark.parametrize("n", [1, 4, 8, 16])
def test_replay_reads_only_the_most_recent_n_frames(world, n):
    _, rssm_b, _, _, samples, _ = world
    s = samples[0]
    calls, undo = spy(rssm_b)
    try:
        method = ResetInit(rssm_b) if n == 1 else ReplayInit(rssm_b, n)
        init = initialize_all([method], s)[method.name]
    finally:
        undo()
    start = max(0, s.t + 1 - n)
    assert len(calls) == min(n, s.t + 1) and init.b_fused_frames == tuple(range(start, s.t + 1))
    for offset, (image, direction, prev) in enumerate(calls):
        i = start + offset
        assert torch.equal(image[0], s.history.images[i]) and direction == int(s.history.directions[i])
        # First window frame has no previous action; later ones use action[i-1] only.
        assert prev is None if offset == 0 else prev == int(s.history.actions[i - 1])


def test_nothing_after_the_handoff_reaches_any_initializer(world):
    rssm_a, rssm_b, _, _, samples, _ = world
    s = samples[0]
    f = s.future
    poisoned = replace(s, future=replace(
        f, images=torch.randint(0, 3, f.images.shape, dtype=torch.uint8),
        directions=(f.directions + 1) % 4, actions=(f.actions + 1) % 3))
    maps = [MapInit("linear", StateMap("linear", rssm_a.feature_dim, 48), rssm_a, rssm_b),
            MapInit("mlp", StateMap("mlp", rssm_a.feature_dim, 48, bottleneck=8), rssm_a, rssm_b)]
    methods = [ResetInit(rssm_b), ReplayInit(rssm_b, 4), ReplayInit(rssm_b, 8),
               ReplayInit(rssm_b, 16), *maps, OracleInit(rssm_b)]
    clean = initialize_all(methods, s)
    dirty = initialize_all(methods, poisoned)
    for name in clean:
        assert torch.equal(clean[name].state_b.deter, dirty[name].state_b.deter), name
        assert torch.equal(clean[name].state_b.stoch, dirty[name].state_b.stoch), name


def test_dataset_states_ignore_post_handoff_observations(world):
    rssm_a, rssm_b, records, manifest, samples, _ = world
    record = next(r for r in records if r.env_snapshot is not None)
    t = record.handoff_t
    tail = tuple(replace(o, image=(o.image + 1) % 3) for o in record.agent_obs[t + 1:])
    mutated = replace(record, agent_obs=record.agent_obs[:t + 1] + tail)
    (new,), _ = build_handoff_samples([mutated], manifest, rssm_a, rssm_b, verify_snapshots=False)
    old = next(s for s in samples if s.episode_id == record.episode_id)
    assert torch.equal(new.state_a.deter, old.state_a.deter)
    assert torch.equal(new.state_b_star.deter, old.state_b_star.deter)
    assert torch.equal(new.state_b_star.stoch, old.state_b_star.stoch)


def test_full_replay_matches_oracle_deterministic_and_seeded_sampling(world):
    _, rssm_b, _, _, samples, _ = world
    for s in samples:
        full = ReplayInit(rssm_b, 10_000, "replay_full")
        out = initialize_all([full, ReplayInit(rssm_b, 16), OracleInit(rssm_b)], s)
        for name in ("replay_full", "replay_16"):  # t=12 < 16: window is the full history
            assert torch.equal(out[name].state_b.deter, out["oracle_b"].state_b.deter)
            assert torch.equal(out[name].state_b.stoch, out["oracle_b"].state_b.stoch)
        assert out["oracle_b"].b_fused_frames == tuple(range(s.t + 1))
    s = samples[0]
    a = filter_history(rssm_b, s.history, sample=True, seed=7)
    b = filter_history(rssm_b, s.history.window(10_000), sample=True, seed=7)
    assert torch.equal(a.deter, b.deter) and torch.equal(a.stoch, b.stoch)
    # Seeded sampling does not consume the global RNG.
    before = torch.get_rng_state()
    filter_history(rssm_b, s.history, sample=True, seed=3)
    assert torch.equal(before, torch.get_rng_state())


def test_obs_t_is_fused_exactly_once_and_continuation_matches_full_history(world):
    _, rssm_b, _, _, samples, _ = world
    s = samples[0]
    calls, undo = spy(rssm_b)
    try:
        init = initialize_all([ReplayInit(rssm_b, 8)], s)["replay_8"]
    finally:
        undo()
    assert len(calls) == 8 and init.b_fused_frames == tuple(range(s.t - 7, s.t + 1))
    assert calls[-1][0][0].equal(s.history.images[-1])  # obs_t is the last frame, fused once
    assert len(set(init.b_fused_frames)) == len(init.b_fused_frames)
    # Continue one step with action[t] and obs[t+1]; compare with filtering obs[0:t+2].
    oracle = initialize_all([OracleInit(rssm_b)], s)["oracle_b"]
    next_obs = {"image": s.future.images[:1], "direction": s.future.directions[:1],
                "task_id": [s.history.task_id]}
    continued = continue_from_handoff(rssm_b, oracle, int(s.future.actions[0]), next_obs)
    extended = replace(s.history, images=torch.cat((s.history.images, s.future.images[:1])),
                       directions=torch.cat((s.history.directions, s.future.directions[:1])),
                       actions=torch.cat((s.history.actions, s.future.actions[:1])))
    expected = filter_history(rssm_b, extended)
    assert torch.equal(continued.deter, expected.deter)
    # Sensitivity: fusing obs_t a second time would change the state.
    again = rssm_b.observe_step(oracle.state_b, s.future.actions[:1] * 0, s.history.obs_t)
    assert not torch.equal(again.deter, oracle.state_b.deter)


def test_all_methods_start_from_the_same_snapshot_and_repeat(world):
    rssm_a, rssm_b, _, _, samples, _ = world
    methods = [ResetInit(rssm_b), ReplayInit(rssm_b, 4), ReplayInit(rssm_b, 8),
               ReplayInit(rssm_b, 16),
               MapInit("linear", StateMap("linear", rssm_a.feature_dim, 48), rssm_a, rssm_b),
               MapInit("mlp", StateMap("mlp", rssm_a.feature_dim, 48, hidden=16), rssm_a, rssm_b),
               OracleInit(rssm_b)]
    s = samples[0]
    first = initialize_all(methods, s, rssm_b=rssm_b)
    again = initialize_all(methods, s, rssm_b=rssm_b)
    stochastic = [initialize_all(methods, s, stochastic=True, seed=5, rssm_b=rssm_b)
                  for _ in range(2)]
    assert set(first) == {"reset", "replay_4", "replay_8", "replay_16", "linear", "mlp", "oracle_b"}
    for name, init in first.items():
        assert init.env_snapshot == s.env_snapshot and init.t == s.t
        assert torch.equal(init.state_b.deter, again[name].state_b.deter)
        assert torch.equal(init.state_b.stoch, again[name].state_b.stoch)
        assert torch.equal(stochastic[0][name].state_b.stoch, stochastic[1][name].state_b.stoch)
    env, obs = restore_env(s)
    try:
        for i, action in enumerate(s.future.actions.tolist()):
            obs, *_ = env.step(action)
            assert torch.equal(torch.from_numpy(obs.image.copy()), s.future.images[i])
    finally:
        env.close()


def test_a_to_b_maps_emit_legal_discrete_states(world):
    rssm_a, rssm_b, _, _, samples, _ = world
    for module in (StateMap("linear", rssm_a.feature_dim, 48),
                   StateMap("mlp", rssm_a.feature_dim, 48, bottleneck=8)):
        with torch.no_grad():  # large raw outputs must still give legal one-hot/logits
            for p in module.parameters():
                p.mul_(50)
        method = MapInit(module.kind, module, rssm_a, rssm_b)
        for stochastic in (False, True):
            state = initialize_all([method], samples[0], stochastic=stochastic, seed=1,
                                   rssm_b=rssm_b)[module.kind].state_b
            validate_b_state(rssm_b, state)
            assert state.stoch.sum(-1).eq(1).all()
    with pytest.raises(ContractError):
        raw = RSSMState(torch.zeros(1, 48), torch.rand(1, 16, 16), torch.zeros(1, 16, 16))
        validate_b_state(rssm_b, raw)


def test_state_map_fit_reduces_loss_and_param_matching(world):
    rssm_a, rssm_b, records, _, _, _ = world
    train = collect_aligned_states(rssm_a, rssm_b, records[:4])
    assert train["features"].shape[1] == rssm_a.feature_dim and train["deter"].shape[1] == 48
    module = StateMap("mlp", rssm_a.feature_dim, 48, hidden=32, bottleneck=8)
    stats = fit_state_map(module, train, steps=60, seed=0)
    assert stats["last_loss"] < stats["first_loss"]
    hidden = matched_hidden_dim(module.num_parameters, in_dim=rssm_a.feature_dim, out_dim=48 + 256,
                                bottleneck=8)
    assert hidden == 32
