import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from environment.gym_wrapper import GymCoopEnv

@pytest.fixture
def game():
    env = GymCoopEnv(level_name="level_3")
    env.reset(seed=42)
    return env

def _hold_slice(obs, agent):
    # holdings are at indices 4-9 for agent 1, 10-15 for agent 2
    if agent == 1:
        return obs[4:10]
    return obs[10:16]

def _pot_oh(obs):
    # pot state one-hot is at indices 60-63
    return obs[60:64]

def _front1(obs):
    # agent 1 front tile features are at 16-25
    return obs[16:26]

def _order_info(obs):
    # order info (time left, target onions, target tomatoes) at 67-69
    return obs[67:70]

def test_obs_shape_and_dtype(game):
    obs, _ = game.reset(seed=42)
    assert obs.shape == (74,)
    assert obs.dtype == np.float32

def test_obs_values_in_range(game):
    obs, _ = game.reset(seed=42)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)

@pytest.mark.parametrize("item,expected_idx", [
    (None, 0),
    ("onion", 1),
    ("tomato", 2),
    ("bowl", 3),
    ("bowl-done-onion-soup", 4),
    ("bowl-burnt-tomato-soup", 5),
])
def test_holding_onehot(game, item, expected_idx):
    game.env.agent1_holding = item
    raw_obs = game.env.get_observation()
    obs = game._get_obs(raw_obs)

    vec = _hold_slice(obs, 1)
    assert vec[expected_idx] == 1.0
    assert sum(vec) == pytest.approx(1.0)

@pytest.mark.parametrize("state,expected_idx", [
    ("idle", 0),
    ("start", 1),
    ("done", 2),
    ("burnt", 3),
])
def test_pot_state_onehot(game, state, expected_idx):
    game.env.pot_state = state
    raw = game.env.get_observation()
    obs = game._get_obs(raw)

    vec = _pot_oh(obs)
    assert vec[expected_idx] == 1.0
    assert sum(vec) == pytest.approx(1.0)

def test_front_tile_pot(game):
    # place agent facing the pot at (5,3)
    game.env.agent1_pos = [5, 2]
    game.env.agent1_dir = (0, 1)
    raw = game.env.get_observation()
    obs = game._get_obs(raw)

    front = _front1(obs)
    assert front[0] == 1.0

def test_front_tile_garbage(game):
    # garbage is at (5,7), stand at (5,6) facing down
    game.env.agent1_pos = [5, 6]
    game.env.agent1_dir = (0, 1)
    raw = game.env.get_observation()
    obs = game._get_obs(raw)

    front = _front1(obs)
    assert front[3] == 1.0

def test_front_tile_onion_dispenser(game):
    # onion dispenser is at (0,1), stand at (1,1) facing left
    game.env.agent1_pos = [1, 1]
    game.env.agent1_dir = (-1, 0)
    raw = game.env.get_observation()
    obs = game._get_obs(raw)

    front = _front1(obs)
    assert front[4] == 1.0

def test_target_order_info(game):
    game.env.pending_orders = []
    game.env.active_orders = [
        {"meal": "onion-tomato-soup", "onions": 1, "tomatoes": 1,
         "start": 0, "deadline": 600, "served": False},
    ]
    game.env.step_count = 0
    raw = game.env.get_observation()
    obs = game._get_obs(raw)

    info = _order_info(obs)
    assert info[1] == pytest.approx(1.0)  # target_on
    assert info[2] == pytest.approx(1.0)  # target_to
    assert 0.0 <= info[0] <= 1.0
