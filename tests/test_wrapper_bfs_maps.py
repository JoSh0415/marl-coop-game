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

def test_reachable_tile_finite(game):
    # agent spawn (2,5) is in the same open component as pot (5,3)
    dm = game._station_dist_maps["P"]
    assert np.isfinite(dm[5, 2])

def test_wall_tile_inf(game):
    # (0,0) is a wall, not walkable
    dm = game._station_dist_maps["P"]
    assert np.isinf(dm[0, 0])

def test_reach_flag_finite(game):
    d, r = game._dist_and_reach((2, 5), "P")
    assert r == 1.0
    assert 0.0 <= d <= 1.0

def test_reach_flag_inf(game):
    d, r = game._dist_and_reach((0, 0), "P")
    assert r == 0.0
    assert d == 1.0

def test_dist_normalised(game):
    dm = game._station_dist_maps["P"]
    for y in range(game.env.grid_height):
        for x in range(game.env.grid_width):
            if np.isfinite(dm[y, x]):
                d_norm, _ = game._dist_and_reach((x, y), "P")
                assert 0.0 <= d_norm <= 1.0

def test_distance_monotonic(game):
    dm = game._station_dist_maps["P"]
    # (5,2) neighbours pot => dist 0, (5,1) one step further => dist 1
    assert dm[2, 5] == pytest.approx(0.0)
    assert dm[1, 5] == pytest.approx(1.0)
    assert dm[2, 5] == dm[1, 5] - 1.0

def test_all_station_maps_present(game):
    for key in ("P", "R", "S", "G", "I", "J"):
        assert key in game._station_dist_maps
        dm = game._station_dist_maps[key]
        assert dm.shape == (game.env.grid_height, game.env.grid_width)
