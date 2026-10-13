"""Step 4 acceptance checks; no pretrained weights or accelerator required."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from cmb_minigrid.config import load_yaml
from cmb_minigrid.contracts import ContractError
from cmb_minigrid.models import RSSM, RSSMConfig, categorical_kl
from cmb_minigrid.reproducibility import set_seed


@pytest.fixture(autouse=True)
def cpu_seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    set_seed(42)
    yield
    torch.set_num_threads(previous)


def batch(batch_size=2, horizon=3):
    image = torch.stack(
        [torch.randint(classes, (batch_size, horizon + 1, 7, 7)) for classes in (11, 6, 3)],
        dim=-1,
    )
    return {
        "agent_obs": {
            "image": image,
            "direction": torch.randint(4, (batch_size, horizon + 1)),
            "task_id": [["hidden_history_nav"] * (horizon + 1) for _ in range(batch_size)],
        },
        "actions": torch.randint(3, (batch_size, horizon)),
        "reward": torch.rand(batch_size, horizon),
        "terminated": torch.zeros(batch_size, horizon, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, horizon, dtype=torch.bool),
        "observation_mask": torch.ones(batch_size, horizon + 1, dtype=torch.bool),
        "transition_mask": torch.ones(batch_size, horizon, dtype=torch.bool),
    }


def obs_at(data, t):
    obs = data["agent_obs"]
    return {"image": obs["image"][:, t], "direction": obs["direction"][:, t],
            "task_id": [row[t] for row in obs["task_id"]]}


@pytest.mark.parametrize("preset,deter,feature_dim", [("a", 128, 384), ("b", 256, 512)])
def test_config_shapes_probabilities_and_backward(preset, deter, feature_dim):
    config = RSSMConfig.from_mapping(load_yaml(f"configs/rssm_{preset}.yaml")["rssm"])
    # Test prior gradient connectivity above the free-nats floor separately
    # from the default floor behavior (covered explicitly below).
    model = RSSM(replace(config, free_nats=0))
    data = batch()
    initial = model.initial(2)
    assert initial.deter.shape == (2, deter)
    assert model.encoder(obs_at(data, 0)["image"]).shape == (2, 64)
    output = model.observe(data["agent_obs"], data["actions"])
    assert output.features.shape == (2, 4, feature_dim)
    assert output.prior_logits.shape == output.posterior_logits.shape == (2, 4, 16, 16)
    for state in (initial, *output.states, model.imagine_step(output.states[-1], data["actions"][:, 0])):
        assert state.stoch.shape == (2, 16, 16)
        assert ((state.stoch == 0) | (state.stoch == 1)).all()
        torch.testing.assert_close(state.stoch.sum(-1), torch.ones(2, 16))
        probs = state.stoch_logits.softmax(-1)
        assert (probs > 0).all()
        torch.testing.assert_close(probs.sum(-1), torch.ones(2, 16))
    heads = model.decode(output.features)
    assert [x.shape for x in heads["image_logits"]] == [
        (2, 4, 7, 7, n) for n in (11, 6, 3)
    ]
    assert heads["reward"].shape == heads["continue_logits"].shape == (2, 4)
    losses = model.loss(data)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    for module in (model.encoder, model.gru, model.prior, model.posterior,
                   model.image_head, model.reward_head, model.continue_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_sequence_matches_single_steps_and_zero_initial_action():
    model = RSSM().eval()
    data = batch()
    output = model.observe(data["agent_obs"], data["actions"])
    state = model.observe_step(model.initial(2), None, obs_at(data, 0))
    torch.testing.assert_close(state.deter, output.states[0].deter)
    zero_action = torch.zeros(2, 3)
    zero = model.observe_step(model.initial(2), zero_action, obs_at(data, 0))
    torch.testing.assert_close(state.deter, zero.deter)
    for t in range(3):
        state = model.observe_step(state, data["actions"][:, t], obs_at(data, t + 1))
        torch.testing.assert_close(state.deter, output.states[t + 1].deter)
        torch.testing.assert_close(state.stoch, output.states[t + 1].stoch)
    left = model.observe_step(model.initial(2), torch.zeros(2, dtype=torch.long), obs_at(data, 0))
    assert not torch.equal(left.deter, output.states[0].deter)


def test_imagination_never_calls_encoder_and_prefix_has_no_future_leak():
    model = RSSM().eval()
    data = batch()
    state = model.observe_step(model.initial(2), None, obs_at(data, 0))
    with patch.object(model.encoder, "forward", side_effect=AssertionError("future observation")):
        imagined = model.imagine_step(state, data["actions"][:, 0])
        imagined_ids = model.imagine_step(state, F.one_hot(data["actions"][:, 0], 3).float())
    torch.testing.assert_close(imagined.stoch, imagined_ids.stoch)
    prior = model.observe_step(state, data["actions"][:, 0], obs_at(data, 1))
    torch.testing.assert_close(imagined.deter, prior.deter)
    before = model.observe(data["agent_obs"], data["actions"])
    torch.testing.assert_close(imagined.stoch_logits, before.prior_logits[:, 1])
    data["agent_obs"]["image"][:, 2:] = 0
    data["agent_obs"]["direction"][:, 2:] = 0
    data["actions"][:, 2:] = 0
    after = model.observe(data["agent_obs"], data["actions"])
    torch.testing.assert_close(before.features[:, :2], after.features[:, :2])


def test_seeded_sampling_deterministic_evaluation_and_straight_through():
    data = batch()
    set_seed(9)
    first = RSSM()
    a = first.observe(data["agent_obs"], data["actions"])
    set_seed(9)
    second = RSSM()
    b = second.observe(data["agent_obs"], data["actions"])
    torch.testing.assert_close(a.features, b.features, rtol=0, atol=0)
    first.eval()
    set_seed(123)
    a = first.observe(data["agent_obs"], data["actions"])
    set_seed(999)
    b = first.observe(data["agent_obs"], data["actions"])
    torch.testing.assert_close(a.features, b.features, rtol=0, atol=0)
    logits = torch.zeros(2, 16, 16, requires_grad=True)
    stoch = first.sample(logits, sample=True)
    (stoch * torch.arange(16)).sum().backward()
    assert logits.grad.abs().sum() > 0
    torch.testing.assert_close(first.sample(logits, sample=False).argmax(-1), logits.argmax(-1))


def test_unimix_is_normalized_and_numerically_stable():
    for mix in (0.0, 0.01):
        model = RSSM(RSSMConfig(unimix=mix))
        raw = torch.full((2, 256), -1000.0)
        raw[:, 0] = 1000.0
        logits = model._logits(raw)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits.exp().sum(-1), torch.ones(2, 16))
        if mix:
            assert (logits.exp() >= mix / 16 - 1e-8).all()


def test_kl_stop_gradients_and_free_nats_after_group_sum():
    q = torch.randn(2, 16, 16, requires_grad=True)
    p = torch.randn(2, 16, 16, requires_grad=True)
    dyn, rep = categorical_kl(q, p, free_nats=0)
    expected = (q.softmax(-1) * (q.log_softmax(-1) - p.log_softmax(-1))).sum((-1, -2))
    torch.testing.assert_close(dyn, expected)
    torch.testing.assert_close(rep, expected)
    q_grad, p_grad = torch.autograd.grad(dyn.sum(), (q, p), allow_unused=True)
    assert q_grad is None and p_grad.abs().sum() > 0
    q_grad, p_grad = torch.autograd.grad(rep.sum(), (q, p), allow_unused=True)
    assert p_grad is None and q_grad.abs().sum() > 0
    q = torch.zeros(2, 16, 16, requires_grad=True)
    p = torch.zeros_like(q, requires_grad=True)
    dyn, rep = categorical_kl(q, p, free_nats=1)
    torch.testing.assert_close(dyn, torch.ones(2))  # 1 per sample, not 16.
    torch.testing.assert_close(rep, torch.ones(2))
    (dyn + rep).sum().backward()
    assert q.grad.count_nonzero() == p.grad.count_nonzero() == 0


def test_reward_continue_are_next_state_targets_and_truncation_bootstraps():
    model = RSSM().eval()
    data = batch()
    data["terminated"][0, -1] = True
    data["truncated"][1, -1] = True
    output = model.observe(data["agent_obs"], data["actions"])
    heads = model.decode(output.features)
    losses = model.loss(data)
    torch.testing.assert_close(losses["reward"], F.mse_loss(heads["reward"][:, 1:], data["reward"]))
    torch.testing.assert_close(losses["continue"], F.binary_cross_entropy_with_logits(
        heads["continue_logits"][:, 1:], (~data["terminated"]).float()))


def test_padding_ignored_in_states_and_every_loss():
    model = RSSM().eval()
    data = batch()
    data["observation_mask"][0, 2:] = False
    data["transition_mask"][0, 1:] = False
    before = model.loss(data)
    data["actions"][0, 1:] = -1
    data["reward"][0, 1:] = float("nan")
    data["terminated"][0, 1:] = True
    data["agent_obs"]["image"][0, 2:] = 255
    data["agent_obs"]["direction"][0, 2:] = -1
    after = model.loss(data)
    for key in before:
        torch.testing.assert_close(before[key], after[key])
    output = model.observe(data["agent_obs"], data["actions"], data["observation_mask"])
    torch.testing.assert_close(output.features[0, 1], output.features[0, 3])


def test_invalid_inputs_fail_without_silent_clamping():
    model = RSSM()
    data = batch()
    obs = obs_at(data, 0)
    with pytest.raises(ContractError, match="observation"):
        model.observe_step(model.initial(2), None, obs | {"hidden_label": 1})
    obs["image"][0, 0, 0, 0] = 11
    with pytest.raises(ContractError, match="categor"):
        model.observe_step(model.initial(2), None, obs)
    with pytest.raises(ContractError, match="action"):
        model.imagine_step(model.initial(2), torch.tensor([-1, 3]))
    with pytest.raises(ContractError, match="shape"):
        RSSM(RSSMConfig(deter=256)).features(model.initial(2))
    data = batch()
    data["terminated"][0, 0] = True
    with pytest.raises(ContractError, match="terminal"):
        model.loss(data)
    data = batch()
    data["observation_mask"][0, 1] = False
    with pytest.raises(ContractError, match="mask"):
        model.loss(data)
    with pytest.raises(ValueError):
        RSSMConfig.from_mapping({"unimix": 1.0})
    with pytest.raises(ValueError):
        RSSMConfig.from_mapping({"embed_dim": 32})


@pytest.mark.parametrize("deter", [128, 256])
def test_single_cpu_batch_overfit(deter):
    # Easy repeated symbolic target, bounded diagnostic, not a Step 5 training run.
    data = batch(horizon=2)
    data["agent_obs"]["image"].zero_()
    data["agent_obs"]["direction"].zero_()
    data["reward"].fill_(0.5)
    model = RSSM(RSSMConfig(deter=deter)).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = model.loss(data)["total"].item()
    for _ in range(40):
        optimizer.zero_grad()
        losses = model.loss(data)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        optimizer.step()
    final = model.loss(data)
    assert final["total"].item() < 0.3 * initial
    assert final["reconstruction"].item() < 0.1
    assert final["reward"].item() < 0.02
