"""Tests for per-time Actor error diagnosis."""

import pytest
import torch

from cmb_minigrid.data.trajectory import collect_records
from cmb_minigrid.models import RSSM, RSSMConfig
from cmb_minigrid.models.policy import ActorPolicy
from cmb_minigrid.scripts.diagnose_policy_errors import diagnose_record


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_diagnosis_contains_probabilities_observation_and_reset_comparison():
    records = collect_records(
        pair_count=1, seed=42,
        map_family_ids=["hidden_history_nav_v1"],
        include_exploration=False,
    )
    record = next(item for item in records if item.episode_role == "expert")
    rssm_a = RSSM(RSSMConfig(deter=32, encoder_type="cnn"))
    rssm_b = RSSM(RSSMConfig(deter=32, encoder_type="cnn"))
    policy_a = ActorPolicy(32 + 16 * 16, hidden_dim=8)
    policy_b = ActorPolicy(32 + 16 * 16, hidden_dim=8)
    report = diagnose_record(
        record, rssm_a, rssm_b, policy_a, policy_b,
        times=(8, 9, 10, 11), device="cpu",
    )
    assert [row["t"] for row in report["rows"]] == [8, 9, 10, 11]
    row = report["rows"][0]
    assert len(row["a_full_history"]["probabilities"]) == 3
    assert len(row["b_full_history"]["probabilities"]) == 3
    assert len(row["reset_b"]["probabilities"]) == 3
    assert len(row["local_observation"]["image"]) == 7
    assert "full_history_wrong_reset_correct" in row["b_full_history_vs_reset_b"]
