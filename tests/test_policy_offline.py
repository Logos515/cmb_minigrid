"""Offline Actor behavior-cloning replay tests."""

import pytest
import torch

from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.models.policy import ActorPolicy
from cmb_minigrid.training.policy import run_offline_replay


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def expert_record():
    records = collect_records(
        pair_count=1,
        seed=42,
        map_family_ids=["hidden_history_nav_v1"],
        include_exploration=False,
    )
    return next(record for record in records if record.episode_role == "expert")


def test_offline_replay_uses_expert_actions_for_environment(expert_record):
    rssm = RSSM(RSSMConfig(deter=32, encoder_type="cnn"))
    policy = ActorPolicy(32 + 16 * 16, hidden_dim=8)
    # Force a prediction that differs from the first expert action. The policy
    # output is still only a diagnostic value and must not drive the replay.
    with torch.no_grad():
        policy.network[-1].weight.zero_()
        policy.network[-1].bias[:] = torch.tensor([10.0, 0.0, 0.0])
    row = run_offline_replay(expert_record, rssm, policy, device="cpu")
    assert row["actor_actions_replayed"] is False
    assert row["replay_actions"] == list(expert_record.actions)
    assert row["expert_actions"] == list(expert_record.expert_actions)
    assert len(row["step_records"]) == expert_record.horizon
    assert row["predicted_actions"][0] == 0
    assert row["step_records"][0]["replay_action"] == expert_record.actions[0]


def test_offline_replay_reports_step_and_handoff_accuracy(expert_record):
    rssm = RSSM(RSSMConfig(deter=32, encoder_type="cnn"))
    policy = ActorPolicy(32 + 16 * 16, hidden_dim=8)
    row = run_offline_replay(expert_record, rssm, policy, device="cpu")
    assert 0.0 <= row["action_accuracy"] <= 1.0
    assert 0.0 <= row["handoff_correct"] <= 1.0
    assert row["steps"] == expert_record.horizon
    assert row["step_records"][expert_record.handoff_t]["t"] == expert_record.handoff_t
    assert "first_error_t" in row
    assert row["steps_before_first_error"] <= row["steps"]
