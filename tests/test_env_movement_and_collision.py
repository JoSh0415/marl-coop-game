import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from environment.env import CoopEnv, action_to_delta

LEVEL = [
    "#####S#####",
    "I         #",
    "#         #",
    "#    P    #",
    "#    #    #",
    "# A  R  B #",
    "#         J",
    "#####G#####",
]

@pytest.fixture
def game():
    env = CoopEnv(LEVEL)
    env.reset(seed=0)
    return env

def test_wall_blocks_movement(game):
    # agent 1 starts at (1,5), move left into wall at (0,5)
    game.agent1_pos = [1, 5]
    game.agent1_dir = (-1, 0)
    old_pos = list(game.agent1_pos)

    game.step(3, 0)  # left

    assert game.agent1_pos == old_pos

def test_station_blocks_movement(game):
    # pot is at (5, 3), try to walk onto it
    game.agent1_pos = [5, 4]
    game.agent1_dir = (0, -1)

    game.step(1, 0)  # up

    assert game.agent1_pos == [5, 4]

def test_agents_cannot_overlap(game):
    # place them adjacent, a1 moves right into a2's cell
    game.agent1_pos = [3, 2]
    game.agent2_pos = [4, 2]

    game.step(4, 0)  # a1 right, a2 stay

    assert game.agent1_pos == [3, 2]

def test_agent2_blocked_by_agent1(game):
    # a1 moves right into empty cell, a2 tries to move into that same cell
    game.agent1_pos = [3, 2]
    game.agent2_pos = [5, 2]

    game.step(4, 3)  # a1 right, a2 left

    assert game.agent1_pos == [4, 2]
    assert game.agent2_pos == [5, 2]

def test_swap(game):
    game.agent1_pos = [3, 2]
    game.agent2_pos = [4, 2]

    game.step(4, 3)  # a1 right, a2 left => swap

    assert game.agent1_pos == [4, 2]
    assert game.agent2_pos == [3, 2]

def test_direction_updates_on_move(game):
    game.agent1_dir = (0, -1)

    game.step(4, 0)  # right

    assert game.agent1_dir == (1, 0)

def test_direction_unchanged_on_stay(game):
    game.agent1_dir = (0, -1)

    game.step(0, 0)

    assert game.agent1_dir == (0, -1)

def test_direction_unchanged_on_interact(game):
    game.agent1_dir = (0, -1)

    game.step(5, 0)

    assert game.agent1_dir == (0, -1)

def test_action_to_delta():
    assert action_to_delta(0) == (0, 0)
    assert action_to_delta(1) == (0, -1)
    assert action_to_delta(2) == (0, 1)
