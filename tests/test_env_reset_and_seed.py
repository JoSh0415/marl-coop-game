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

def _snapshot(env):
    return (
        tuple(env.agent1_pos),
        tuple(env.agent2_pos),
        env.pot_state,
        env.pot_onions,
        env.pot_tomatoes,
        env.score,
        env.step_count,
        tuple((o["meal"], o["start"]) for o in env.pending_orders),
    )

def test_same_seed_same_state():
    env = CoopEnv(LEVEL)
    env.reset(seed=99)
    snap_a = _snapshot(env)

    env.reset(seed=99)
    snap_b = _snapshot(env)

    assert snap_a == snap_b

def test_same_seed_same_orders():
    env = CoopEnv(LEVEL)
    env.reset(seed=77)
    orders_a = [(o["meal"], o["start"]) for o in env.pending_orders]

    env.reset(seed=77)
    orders_b = [(o["meal"], o["start"]) for o in env.pending_orders]

    assert orders_a == orders_b

def test_different_seeds_vary_orders():
    env = CoopEnv(LEVEL)
    seen = set()
    for s in range(50):
        env.reset(seed=s)
        key = tuple(o["meal"] for o in env.pending_orders)
        seen.add(key)

    # 3 meals ^ 3 slots = 27 possible combos, 50 seeds should hit >1
    assert len(seen) > 1

def test_reset_clears_pot_and_score():
    env = CoopEnv(LEVEL)
    env.reset(seed=10)

    # dirty the state
    env.pot_state = "done"
    env.pot_onions = 1
    env.score = 2

    env.reset(seed=10)

    assert env.pot_state == "idle"
    assert env.pot_onions == 0
    assert env.pot_tomatoes == 0
    assert env.score == 0
    assert env.step_count == 0

def test_reset_clears_wall_items():
    env = CoopEnv(LEVEL)
    env.reset(seed=10)

    env.wall_items[(5, 1)] = "onion"
    env.reset(seed=10)

    assert len(env.wall_items) == 0

def test_reset_clears_holdings():
    env = CoopEnv(LEVEL)
    env.reset(seed=10)

    env.agent1_holding = "bowl"
    env.agent2_holding = "tomato"
    env.reset(seed=10)

    assert env.agent1_holding == None
    assert env.agent2_holding == None

def test_reset_restores_positions():
    env = CoopEnv(LEVEL)
    env.reset(seed=10)
    orig_a1 = list(env.agent1_pos)
    orig_a2 = list(env.agent2_pos)

    env.agent1_pos = [1, 1]
    env.agent2_pos = [2, 2]
    env.reset(seed=10)

    assert env.agent1_pos == orig_a1
    assert env.agent2_pos == orig_a2
