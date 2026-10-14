"""Independent diagnostic checks; no model training or checkpoint dependency."""

from dataclasses import replace

import pytest
import torch

from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.models import RSSM
from cmb_minigrid.reproducibility import set_seed
from cmb_minigrid.scripts.diagnose_categorical_response import (
    diagnose_model,
    interpret,
    select_pair,
    tensor_diff,
)
from cmb_minigrid.training.rssm import model_batch


@pytest.fixture
def pair():
    return select_pair(collect_records(pair_count=1, seed=42,
                                       map_family_ids=["hidden_history_nav_v2"]), None)


def controlled_model(pair, *, flip=False):
    set_seed(42)
    model = RSSM().eval()
    batch = model_batch(pair, "cpu")
    with torch.no_grad():
        embedding = model.encoder(batch["agent_obs"]["image"][:, 0])
        delta = embedding[1] - embedding[0]
        assert delta.norm() > 0
        first, last = model.posterior[0], model.posterior[2]
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(-10)
        first.weight[0, model.config.deter:model.config.deter + 64] = delta / delta.square().sum()
        first.bias[0] = -(embedding.mean(0) * delta / delta.square().sum()).sum()
        bias = last.bias.view(16, 16)
        bias[:, 0] = 0
        # Two outputs can respond without crossing, or straddle a decision boundary.
        bias[:, 1] = 0 if flip else -2
        last.weight.view(16, 16, -1)[:, 1, 0] = 1
    return model


@pytest.mark.parametrize("flip", [False, True])
def test_response_matches_filtering_and_distinguishes_argmax_loss(pair, flip):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = controlled_model(pair, flip=flip)
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = diagnose_model(model, pair, "cpu")
        early = report["t=0"]
        assert early["observation_difference"]["changed_entries_per_channel"] == [0, 1, 0]
        assert early["encoder_difference"]["l2"] > 0
        assert early["raw_posterior_difference"]["l2"] > 0
        assert early["posterior_probabilities_difference"]["l2"] > 0
        assert early["categorical_argmax"]["changed_group_count"] == (16 if flip else 0)
        assert interpret(report)["nonzero_response_discarded_by_argmax_at_t0"] is (not flip)
        assert early["deterministic_state_difference"]["max_abs"] == 0
        assert (report["t=1"]["deterministic_state_difference"]["l2"] > 0) is flip

        batch = model_batch(pair, "cpu")
        with torch.no_grad():
            sequence = model.observe(batch["agent_obs"], batch["actions"],
                                     batch["observation_mask"], sample=False)
        for t in (0, 1):
            state = sequence.states[t]
            assert report[f"t={t}"]["stochastic_state_difference"] == tensor_diff(
                state.stoch[0], state.stoch[1])
            assert report[f"t={t}"]["deterministic_state_difference"] == tensor_diff(
                state.deter[0], state.deter[1])
            assert report[f"t={t}"]["observe_step_exact_match"]
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), rng)
        assert not model.encoder._forward_hooks and not model.posterior._forward_hooks
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch.set_num_threads(old_threads)


def test_zero_posterior_response_not_misclassified_as_argmax_loss(pair):
    model = controlled_model(pair)
    with torch.no_grad():
        model.posterior[2].weight.zero_()
    report = diagnose_model(model, pair, "cpu")
    assert report["t=0"]["encoder_difference"]["l2"] > 0
    assert report["t=0"]["posterior_probabilities_difference"]["l2"] == 0
    assert not interpret(report)["nonzero_response_discarded_by_argmax_at_t0"]


def test_selection_and_action_mismatch_rejected(pair):
    with pytest.raises(ValueError, match="no expert pairs"):
        select_pair([], None)
    with pytest.raises(ValueError, match="exactly two"):
        select_pair(pair, "missing")
    with pytest.raises(ValueError, match="exactly two"):
        select_pair(pair[:1], None)
    changed = ((pair[1].actions[0] + 1) % 3, *pair[1].actions[1:])
    model = controlled_model(pair)
    with pytest.raises(ValueError, match=r"action\[0\]"):
        diagnose_model(model, [pair[0], replace(pair[1], actions=changed)], "cpu")


def test_nonfinite_diff_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        tensor_diff(torch.tensor([float("nan")]), torch.zeros(1))
