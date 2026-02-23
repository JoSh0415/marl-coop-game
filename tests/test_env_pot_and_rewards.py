import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from environment.env import CoopEnv, COOK_TIME, BURN_TIME

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

def _make_env(orders=None):
    env = CoopEnv(LEVEL)
    env.reset(seed=0)
    env.pending_orders = []
    if orders is None:
        orders = [
            {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
             "start": 0, "deadline": 9999, "served": False},
        ]
    env.active_orders = list(orders)
    return env

def test_invalid_pot_add_not_idle():
    env = _make_env()
    env.pot_state = "start"
    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "onion"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(-0.01)
    assert env.agent1_holding == "onion"

def test_invalid_pot_add_duplicate():
    env = _make_env()
    env.pot_onions = 1
    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "onion"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(-0.01)

def test_invalid_pot_add_no_matching_order():
    # only tomato-soup is wanted, but we try to add onion
    env = _make_env(orders=[
        {"meal": "tomato-soup", "onions": 0, "tomatoes": 1,
         "start": 0, "deadline": 9999, "served": False},
    ])
    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "onion"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(-0.01)

def test_valid_ingredient_add():
    env = _make_env()
    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "onion"
    env.soups_collected = 0

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(1.0)
    assert env.pot_onions == 1
    assert env.agent1_holding is None

def test_ingredient_add_no_reward_over_budget():
    env = _make_env()
    env.soups_collected = 3
    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "onion"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(0.0)
    assert env.pot_onions == 1

def test_pot_idle_to_start():
    # onion-soup needs 1 onion 0 tomatoes, so recipe is complete
    env = _make_env()
    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "onion"

    env.handle_interact(agent=1)

    assert env.pot_state == "start"
    assert env.pot_timer == 0

def test_pot_start_to_done():
    env = _make_env()
    env.pot_state = "start"
    env.pot_timer = COOK_TIME - 1
    env.pot_onions = 1
    env.pot_recipe = "onion-soup"

    env.step(0, 0)

    assert env.pot_state == "done"

def test_pot_done_to_burnt():
    env = _make_env()
    env.pot_state = "done"
    env.pot_timer = BURN_TIME - 1
    env.pot_onions = 1
    env.pot_recipe = "onion-soup"

    _, reward, _, _ = env.step(0, 0)

    # burn penalty + step penalty
    assert env.pot_state == "burnt"
    assert reward <= -3.0

def test_pickup_done_soup():
    env = _make_env()
    env.pot_state = "done"
    env.pot_timer = COOK_TIME
    env.pot_onions = 1
    env.pot_tomatoes = 0
    env.pot_recipe = "onion-soup"
    env.soups_collected = 0

    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "bowl"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(2.0)
    assert env.agent1_holding == "bowl-done-onion-soup"
    assert env.soups_collected == 1
    assert env.pot_state == "idle"

def test_pickup_done_soup_over_budget():
    env = _make_env()
    env.pot_state = "done"
    env.pot_timer = COOK_TIME
    env.pot_onions = 1
    env.pot_tomatoes = 0
    env.pot_recipe = "onion-soup"
    env.soups_collected = 3

    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "bowl"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(0.0)
    assert env.soups_collected == 4

def test_pickup_burnt_soup():
    env = _make_env()
    env.pot_state = "burnt"
    env.pot_timer = BURN_TIME
    env.pot_onions = 1
    env.pot_recipe = "onion-soup"
    env.soups_collected = 0

    env.agent1_pos = [9, 3]
    env.agent1_dir = (1, 0)
    env.agent1_holding = "bowl"

    r = env.handle_interact(agent=1)

    assert r == pytest.approx(-3.0)
    assert env.soups_collected == 1

def test_cook_done_reward_gated():
    env = _make_env()
    env.pot_state = "start"
    env.pot_timer = COOK_TIME - 1
    env.pot_onions = 1
    env.pot_recipe = "onion-soup"
    env.soups_collected = 3

    _, reward, _, _ = env.step(0, 0)

    assert env.pot_state == "done"
    assert reward == pytest.approx(-0.01)

def test_perfect_terminal_bonus():
    env = _make_env(orders=[
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": True},
        {"meal": "tomato-soup", "onions": 0, "tomatoes": 1,
         "start": 0, "deadline": 9999, "served": True},
        {"meal": "onion-tomato-soup", "onions": 1, "tomatoes": 1,
         "start": 0, "deadline": 9999, "served": True},
    ])
    env.score = 3
    env.step_count = env.max_steps - 1

    _, reward, done, info = env.step(0, 0)

    assert done is True
    assert reward == pytest.approx(-0.01 + 10.0, abs=1e-4)

def test_no_bonus_if_failures():
    env = _make_env(orders=[
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": True},
        {"meal": "tomato-soup", "onions": 0, "tomatoes": 1,
         "start": 0, "deadline": 9999, "served": True},
        {"meal": "onion-tomato-soup", "onions": 1, "tomatoes": 1,
         "start": 0, "deadline": 9999, "served": True},
    ])
    env.score = 3
    env.failed_orders = [{"meal": "x"}]
    env.step_count = env.max_steps - 1

    _, reward, done, _ = env.step(0, 0)

    assert done is True
    assert reward == pytest.approx(-0.01, abs=1e-4)
