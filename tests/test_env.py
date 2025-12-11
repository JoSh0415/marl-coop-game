import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
pygame.display.set_mode((1, 1))

import pytest
from environment.env import CoopEnv, find_char, COOK_TIME, BURN_TIME

LEVEL1 = [
    "###S#######",
    "#    #    #",
    "#    #    #",
    "#    #    P",
    "I    #    #",
    "# B  #  A #",
    "#    #    G",
    "##R#####J##",
]

ORDERS1 = [
    {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0},
    {"meal": "tomato-soup", "onions": 0, "tomatoes": 1, "start": 1500},
    {"meal": "onion-tomato-soup", "onions": 1, "tomatoes": 1, "start": 3000}
]

@pytest.fixture
def game():
    env = CoopEnv(LEVEL1, ORDERS1)
    return env

def test_recipes(game):
    assert game._counts_to_recipe(1, 0) == "onion-soup"
    assert game._counts_to_recipe(0, 1) == "tomato-soup"
    assert game._counts_to_recipe(1, 1) == "onion-tomato-soup"
    
    assert game._counts_to_recipe(2, 0) == None
    assert game._counts_to_recipe(0, 2) == None
    
    assert game._recipe_to_counts("onion-soup") == (1, 0)
    assert game._recipe_to_counts("tomato-soup") == (0, 1)
    assert game._recipe_to_counts("onion-tomato-soup") == (1, 1)
    assert game._recipe_to_counts("invalid-recipe") == (None, None)

def test_pickup_onion(game):
    game.agent1_pos = [1, 4]
    game.agent1_dir = (-1, 0)
    obs, reward, done, info = game.step(5, 0)
    assert game.agent1_holding == "onion"

def test_pickup_tomato(game):
    game.agent1_pos = [8, 6]
    game.agent1_dir = (0, 1)
    game.step(5, 0)
    assert game.agent1_holding == "tomato"

def test_pickup_bowl(game):
    game.agent1_pos = [2, 6]
    game.agent1_dir = (0, 1)
    game.step(5, 0)
    assert game.agent1_holding == "bowl"

def test_put_ingredient_in_pot(game):
    game.agent1_pos = [9, 3]
    game.agent1_dir = (1, 0)
    game.agent1_holding = "onion"
    assert game.pot_onions == 0
    assert game.pot_state == "idle"
    
    game.step(5, 0)
    
    assert game.pot_onions == 1
    assert game.agent1_holding == None
    assert game.pot_state == "start"

def test_pot_cooking_transition(game):
    game.pot_state = "start"
    game.pot_timer = COOK_TIME - 1
    game.pot_onions = 1
    game.pot_recipe = "onion-soup"
    
    game.step(0, 0)
    
    assert game.pot_state == "done"

def test_pot_burning_transition(game):
    game.pot_state = "done"
    game.pot_timer = BURN_TIME - 1
    game.pot_onions = 1
    game.pot_recipe = "onion-soup"
    
    game.step(0, 0)
    
    assert game.pot_state == "burnt"

def test_fill_bowl_from_finished_pot(game):
    game.pot_state = "done"
    game.pot_onions = 1
    game.pot_tomatoes = 0
    game.pot_recipe = "onion-soup"
    game.pot_timer = COOK_TIME
    
    game.agent1_pos = [9, 3]
    game.agent1_dir = (1, 0)
    game.agent1_holding = "bowl"
    
    game.step(5, 0)
    
    assert game.agent1_holding == "bowl-done-onion-soup"
    assert game.pot_state == "idle"
    assert game.pot_onions == 0
    assert game.pot_tomatoes == 0

def test_serve_correct_order(game):
    game.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0, "deadline": 1000, "served": False}
    ]
    game.agent1_pos = [3, 1]
    game.agent1_dir = (0, -1)
    game.agent1_holding = "bowl-done-onion-soup"
    initial_score = game.score
    
    obs, reward, done, info = game.step(5, 0)
    
    assert game.score == initial_score + 1
    assert reward > 0
    assert game.active_orders[0]["served"] == True
    assert len(game.completed_orders) == 1

def test_serve_wrong_soup(game):
    game.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0, "deadline": 1000, "served": False}
    ]
    game.agent1_pos = [3, 1]
    game.agent1_dir = (0, -1)
    game.agent1_holding = "bowl-done-tomato-soup"
    
    score_before = game.score
    obs, reward, done, info = game.step(5, 0)
    
    assert game.score == score_before
    assert reward < 0
    assert game.active_orders[0]["served"] == False

def test_serve_undercooked_soup(game):
    game.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0, "deadline": 1000, "served": False}
    ]
    game.agent1_pos = [3, 1]
    game.agent1_dir = (0, -1)
    game.agent1_holding = "bowl-start-onion-soup"
    
    old_score = game.score
    obs, reward, done, info = game.step(5, 0)
    
    assert game.score == old_score
    assert reward < 0

def test_serve_burnt_soup(game):
    game.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0, "deadline": 1000, "served": False}
    ]
    game.agent1_pos = [3, 1]
    game.agent1_dir = (0, -1)
    game.agent1_holding = "bowl-burnt-onion-soup"
    initial_score = game.score
    obs, reward, done, info = game.step(5, 0)
    
    assert game.score == initial_score
    assert reward < 0

def test_trash_item(game):
    game.agent1_pos = [9, 6]
    game.agent1_dir = (1, 0)
    game.agent1_holding = "onion"
    
    game.step(5, 0)
    
    assert game.agent1_holding is None

def test_order_expiry(game):
    game.pending_orders = []
    game.active_orders = [
        {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0, "deadline": 10, "served": False}
    ]
    game.step_count = 11
    
    obs, reward, done, info = game.step(0, 0)
    
    assert len(game.failed_orders) == 1
    assert reward < -0.5

def test_base_time_penalty(game):
    obs, reward, done, info = game.step(0, 0)
    
    assert reward < 0

def test_done_flag(game):
    env = CoopEnv(LEVEL1, ORDERS1, max_steps=5)
    
    for i in range(4):
        obs, reward, done, info = env.step(0, 0)
        assert done == False
    
    obs, reward, done, info = env.step(0, 0)
    assert done == True
