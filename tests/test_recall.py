"""Public-cue supervision, masked handoff gradients and inference isolation."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from cmb_minigrid.contracts import AgentObs
from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.models.rssm import early_key_color, recall_targets
from cmb_minigrid.training.rssm import _resume_signature, evaluate, load_rssm, model_batch


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def records():
    return collect_records(pair_count=1, seed=42, include_exploration=False)


def test_public_color_targets_and_missing_ambiguous_cues(records):
    batch = model_batch(records, "cpu")
    image = batch["agent_obs"]["image"][:, 0]
    assert early_key_color(image).tolist() == [0, 1]
    missing = torch.zeros_like(image[:1])
    other = missing.clone()
    other[:, 0, 0, 0] = OBJECT_TO_IDX["key"]
    other[:, 0, 0, 1] = COLOR_TO_IDX["green"]
    ambiguous = image[:1].clone()
    ambiguous[:, 0, 0, 0] = OBJECT_TO_IDX["key"]
    ambiguous[:, 0, 0, 1] = COLOR_TO_IDX["blue"]
    assert early_key_color(torch.cat((missing, other, ambiguous))).tolist() == [-1, -1, -1]
    assert recall_targets(batch, 12)[1].tolist() == [True, True]
    batch["observation_mask"][1, 12:] = False
    assert recall_targets(batch, 12)[1].tolist() == [True, False]
    assert not recall_targets(batch, 100)[1].any()


def test_recall_uses_state12_only_and_backpropagates_through_history(records):
    torch.manual_seed(42)
    model = RSSM(RSSMConfig(recall_scale=5.0)).train()
    batch = model_batch(records, "cpu")
    with patch.object(model, "observe", wraps=model.observe) as observe:
        losses = model.loss(batch, sample=False)
    assert observe.call_count == 1
    expected = model.observe(batch["agent_obs"], batch["actions"], sample=False).features[:, 12]
    targets = early_key_color(batch["agent_obs"]["image"][:, 0])
    torch.testing.assert_close(losses["recall"], torch.nn.functional.cross_entropy(
        model.recall_head(expected), targets))
    base = RSSM(RSSMConfig())
    base.load_state_dict({k: v for k, v in model.state_dict().items()
                          if not k.startswith("recall_head.")})
    torch.testing.assert_close(losses["total"], base.loss(batch, sample=False)["total"]
                               + 5 * losses["recall"])
    losses["recall"].backward()
    for module in (model.encoder, model.posterior, model.gru, model.recall_head):
        assert sum(p.grad.abs().sum().item() for p in module.parameters()
                   if p.grad is not None) > 0
    assert all(p.grad is None for p in model.image_head.parameters())
    assert all(p.grad is None for p in model.prior.parameters())


def test_recall_head_and_targets_never_enter_filtering(records):
    model = RSSM(RSSMConfig(recall_scale=5.0)).eval()
    batch = model_batch(records, "cpu")
    with torch.no_grad():
        before = model.observe(batch["agent_obs"], batch["actions"], sample=False).features
        for parameter in model.recall_head.parameters():
            parameter.fill_(123)
        after = model.observe(batch["agent_obs"], batch["actions"], sample=False).features
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert set(batch) == {"agent_obs", "actions", "reward", "terminated", "truncated",
                          "observation_mask", "transition_mask"}


def test_visual_recall_supervises_encoder_at_t0_and_t1_without_filtering(records):
    model = RSSM(RSSMConfig(visual_recall_scale=3.0)).train()
    batch = model_batch(records, "cpu")
    losses = model.loss(batch, sample=False)
    assert losses["visual_recall_t0_accuracy"].item() in (0.0, 0.5, 1.0)
    assert losses["visual_recall_t1_accuracy"].item() in (0.0, 0.5, 1.0)
    assert losses["total"].item() == pytest.approx(
        losses["reconstruction"].item() + losses["reward"].item()
        + losses["continue"].item() + 0.5 * losses["dynamics_kl"].item()
        + 0.1 * losses["representation_kl"].item()
        + 3.0 * losses["visual_recall"].item(), abs=1e-5)
    losses["visual_recall"].backward()
    assert sum(p.grad.abs().sum().item() for p in model.encoder.parameters()
               if p.grad is not None) > 0
    assert sum(p.grad.abs().sum().item() for p in model.visual_recall_head.parameters()
               if p.grad is not None) > 0
    assert all(p.grad is None for p in model.gru.parameters())
    with torch.no_grad():
        before = model.observe(batch["agent_obs"], batch["actions"], sample=False).features
        for parameter in model.visual_recall_head.parameters():
            parameter.fill_(99)
        after = model.observe(batch["agent_obs"], batch["actions"], sample=False).features
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_visual_recall_mask_and_config_conversion(records):
    model = RSSM(RSSMConfig(visual_recall_scale=1.0, visual_recall_times=(0, 100)))
    batch = model_batch(records, "cpu")
    losses = model.loss(batch, sample=False)
    assert losses["visual_recall_t100"].item() == 0
    assert RSSMConfig.from_mapping({"visual_recall_times": [0, 1]}).visual_recall_times == (0, 1)


def test_recall_empty_targets_and_future_poisoning(records):
    model = RSSM(RSSMConfig(recall_scale=5.0))
    batch = model_batch(records, "cpu")
    first = model.loss(batch, sample=False)
    # Modify all observations/actions strictly after the handoff.
    batch["agent_obs"]["image"][:, 13:] = 0
    batch["agent_obs"]["direction"][:, 13:] = 0
    batch["actions"][:, 12:] = 0
    second = model.loss(batch, sample=False)
    torch.testing.assert_close(first["recall"], second["recall"], rtol=0, atol=0)
    for t in (12, 100):
        no_cue = RSSM(RSSMConfig(recall_scale=5.0, recall_t=t))
        batch["agent_obs"]["image"][:, 0] = 0
        loss = no_cue.loss(batch, sample=False)
        assert loss["recall"].item() == 0
        assert torch.isfinite(loss["total"])
        loss["total"].backward()


def test_validation_weights_recall_by_eligible_episodes(records):
    model = RSSM(RSSMConfig(recall_scale=5.0))
    blank = records[0].agent_obs[0].image.copy()
    blank[:] = 0
    absent = replace(records[0], agent_obs=(AgentObs(blank, 0, records[0].agent_obs[0].task_id),)
                     + records[0].agent_obs[1:])
    mixed = records + [absent]
    assert evaluate(model, mixed, 1) == pytest.approx(evaluate(model, mixed, 3), abs=2e-6)


def test_legacy_checkpoint_without_recall_fields_loads(tmp_path):
    model = RSSM()
    config = {name: value for name, value in vars(model.config).items()
              if not name.startswith("recall_")}
    path = tmp_path / "old.pt"
    torch.save({"schema_version": 1, "config": {"rssm": config},
                "model": model.state_dict()}, path)
    loaded, _ = load_rssm(path)
    assert loaded.config.recall_scale == 0
    assert not hasattr(loaded, "recall_head")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    old_config = {"rssm": config, "training": {"max_steps": 100}}
    new_config = old_config | {"rssm": vars(loaded.config), "training": {"max_steps": 200}}
    assert _resume_signature(old_config) == _resume_signature(new_config)


@pytest.mark.parametrize("values", [{"recall_scale": -1}, {"recall_scale": float("nan")},
                                    {"recall_t": -1}, {"recall_t": True}])
def test_recall_config_validation(values):
    with pytest.raises(ValueError):
        RSSMConfig(**values)
