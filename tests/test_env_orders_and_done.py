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

def _make_env():
    env = CoopEnv(LEVEL)
    env.reset(seed=0)
    return env

def test_order_activates_at_correct_step():
    env = _make_env()
    env.pending_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 5},
    ]
    env.active_orders = []

    # order check runs before step_count increments
    for _ in range(5):
        env.step(0, 0)

    assert len(env.active_orders) == 0
    assert len(env.pending_orders) == 1

    env.step(0, 0)  # step_count==5 at check time

    assert len(env.active_orders) == 1
    assert env.active_orders[0]["meal"] == "onion-soup"
    assert len(env.pending_orders) == 0

def test_order_gets_deadline():
    env = _make_env()
    env.pending_orders = [
        {"meal": "tomato-soup", "onions": 0, "tomatoes": 1, "start": 0},
    ]
    env.active_orders = []

    env.step(0, 0)

    order = env.active_orders[0]
    assert order["deadline"] == 0 + env.order_time
    assert order["served"] is False

def test_order_expiry(env=None):
    env = _make_env()
    env.pending_orders = []
    env.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 10, "served": False},
    ]

    # past deadline
    env.step_count = 11

    _, reward, _, _ = env.step(0, 0)

    assert len(env.failed_orders) == 1
    assert reward < -1.5

def test_serve_correct_order():
    env = _make_env()
    env.pending_orders = []
    env.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": False},
    ]
    env.agent1_pos = [3, 1]
    env.agent1_dir = (0, -1)
    env.agent1_holding = "bowl-done-onion-soup"

    _, reward, _, _ = env.step(5, 0)

    assert env.active_orders[0]["served"] == True
    assert env.score == 1
    assert len(env.completed_orders) == 1
    assert reward > 15.0

def test_done_on_max_steps():
    env = CoopEnv(LEVEL, max_steps=3)
    env.reset(seed=0)
    
    # need at least one unserved order so it doesn't finish immediately
    env.pending_orders = []
    env.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": False},
    ]

    for _ in range(2):
        _, _, done, _ = env.step(0, 0)
        assert done is False

    _, _, done, _ = env.step(0, 0)
    assert done is True

def test_done_when_all_served():
    env = _make_env()
    env.pending_orders = []
    env.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": True},
    ]

    _, _, done, _ = env.step(0, 0)

    assert done is True

def test_not_done_if_pending():
    env = _make_env()
    env.pending_orders = [
        {"meal": "tomato-soup", "onions": 0, "tomatoes": 1, "start": 999},
    ]
    env.active_orders = []

    _, _, done, _ = env.step(0, 0)

    assert done is False

def test_not_done_if_unserved():
    env = _make_env()
    env.pending_orders = []
    env.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0,
         "start": 0, "deadline": 9999, "served": False},
    ]

    _, _, done, _ = env.step(0, 0)

    assert done is False
