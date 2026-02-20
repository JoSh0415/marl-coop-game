import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from environment.gym_wrapper import GymCoopEnv

def test_gym_compliance():
    env = GymCoopEnv()
    
    check_env(env)

def test_obs_shape():
    env = GymCoopEnv()
    obs, info = env.reset()
    
    assert obs.shape == (74,)
    assert isinstance(obs, np.ndarray)
    
    print(f"Observation: {obs}")

def test_step_interaction():
    env = GymCoopEnv()
    env.reset()
    
    action = env.action_space.sample()
    
    obs, reward, term, trunc, info = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(term, bool)