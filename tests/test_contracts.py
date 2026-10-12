import numpy as np
import pytest

from cmb_minigrid.contracts import AgentObs, ContractError, RSSMState, Trajectory


def obs(value=0):
    return AgentObs(np.full((7, 7, 3), value, dtype=np.uint8), 0, "memory")


def test_trajectory_uses_obs_plus_one_contract():
    trajectory = Trajectory([obs(), obs(1)], [2])
    assert trajectory.horizon == 1
    assert trajectory.actions == (2,)
    with pytest.raises(ContractError, match=r"obs\[0:T\+1\]"):
        Trajectory([obs()], [2])


def test_agent_obs_rejects_non_local_or_non_integer_image():
    with pytest.raises(ContractError, match="shape"):
        AgentObs(np.zeros((7, 7)), 0, "memory")
    with pytest.raises(ContractError, match="integer"):
        AgentObs(np.zeros((7, 7, 3), dtype=np.float32), 0, "memory")


def test_agent_obs_value_equality_compares_image_contents():
    assert obs() == obs()
    assert obs() != obs(1)


def test_rssm_state_shape_contract():
    state = RSSMState(np.zeros((2, 128)), np.zeros((2, 16, 16)))
    assert state.deter.shape == (2, 128)
    with pytest.raises(ContractError, match="batch"):
        RSSMState(np.zeros((2, 128)), np.zeros((3, 16, 16)))
    with pytest.raises(ContractError, match=r"\[B, D\]"):
        RSSMState(np.zeros((2, 3, 128)), np.zeros((2, 16, 16)))
