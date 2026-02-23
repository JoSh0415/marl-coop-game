import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from environment.env import CoopEnv

LEVEL = [
    "###S#######",
    "#    #    #",
    "#    #    #",
    "#    #    P",
    "I    #    #",
    "# B  #  A #",
    "#    #    G",
    "##R#####J##",
]

@pytest.fixture
def game():
    env = CoopEnv(LEVEL)
    env.reset(seed=0)
    env.pending_orders = []
    env.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": False},
    ]
    return env

def test_pickup_onion(game):
    game.agent1_pos = [1, 4]
    game.agent1_dir = (-1, 0)
    game.agent1_holding = None

    game.step(5, 0)

    assert game.agent1_holding == "onion"

def test_cannot_pickup_if_holding(game):
    game.agent1_pos = [1, 4]
    game.agent1_dir = (-1, 0)
    game.agent1_holding = "bowl"

    game.step(5, 0)

    assert game.agent1_holding == "bowl"

def test_pickup_tomato(game):
    # face tomato dispenser J at (8, 7)
    game.agent1_pos = [8, 6]
    game.agent1_dir = (0, 1)
    game.agent1_holding = None

    game.step(5, 0)

    assert game.agent1_holding == "tomato"

def test_pickup_bowl(game):
    # bowl dispenser R is at (2, 7)
    game.agent1_pos = [2, 6]
    game.agent1_dir = (0, 1)
    game.agent1_holding = None

    game.step(5, 0)

    assert game.agent1_holding == "bowl"

def test_place_on_counter(game):
    key = (5, 2)
    game.agent1_pos = [4, 2]
    game.agent1_dir = (1, 0)
    game.agent1_holding = "onion"

    game.step(5, 0)

    assert game.agent1_holding is None
    assert game.wall_items.get(key) == "onion"

def test_pickup_from_counter(game):
    key = (5, 2)
    game.wall_items[key] = "tomato"
    game.agent1_pos = [4, 2]
    game.agent1_dir = (1, 0)
    game.agent1_holding = None

    game.step(5, 0)

    assert game.agent1_holding == "tomato"
    assert key not in game.wall_items

def test_garbage_clears_holding(game):
    # garbage G is at (10, 6)
    game.agent1_pos = [9, 6]
    game.agent1_dir = (1, 0)
    game.agent1_holding = "onion"

    game.step(5, 0)

    assert game.agent1_holding is None

def test_handoff_counter_bridging(game):
    # column 5 wall tiles border both left and right components
    assert game._is_handoff_counter((5, 1)) is True
    assert game._is_handoff_counter((5, 3)) is True
    assert game._is_handoff_counter((5, 6)) is True

def test_handoff_counter_non_bridging(game):
    assert game._is_handoff_counter((0, 1)) is False
    assert game._is_handoff_counter((10, 1)) is False

def test_handoff_counters_populated(game):
    assert len(game.handoff_counters) > 0
    for (x, y) in game.handoff_counters:
        assert game.level[y][x] == "#"

def test_handoff_reward(game):
    key = (5, 2)
    game.agent1_pos = [4, 2]
    game.agent1_dir = (1, 0)
    game.agent1_holding = "bowl-done-onion-soup"
    game.handoffs_rewarded = 0

    _, reward, _, _ = game.step(5, 0)

    # step penalty + handoff bonus
    assert reward == pytest.approx(-0.01 + 2.0, abs=1e-6)
    assert game.handoffs_rewarded == 1
    assert game.wall_items.get(key) == "bowl-done-onion-soup"

def test_handoff_reward_capped(game):
    game.agent1_pos = [4, 2]
    game.agent1_dir = (1, 0)
    game.handoffs_rewarded = 3
    game.agent1_holding = "bowl-done-onion-soup"

    _, reward, _, _ = game.step(5, 0)

    assert reward == pytest.approx(-0.01, abs=1e-6)
    assert game.handoffs_rewarded == 3
