import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from collections import deque
from environment.gym_wrapper import GymCoopEnv

def test_seed_splits_disjoint():
    # keep validation and test rollouts fully non-overlapping
    val_seeds = set(range(0, 500))
    test_seeds = set(range(10000, 12500))
    assert val_seeds.isdisjoint(test_seeds)

def test_perfect_definition():
    score = 3
    failures = 0
    perfect = (score == 3 and failures == 0)
    assert perfect is True

    assert (2 == 3 and 0 == 0) is False
    assert (3 == 3 and 1 == 0) is False

def test_frame_stack_shape():
    env = GymCoopEnv(level_name="level_3")
    obs, _ = env.reset(seed=0)

    stack_size = 4
    frames = deque(maxlen=stack_size)
    for _ in range(stack_size):
        frames.append(obs.copy())

    # 4 stacked 74-dim frames
    stacked = np.concatenate(list(frames), axis=0)
    assert stacked.shape == (74 * 4,)
    assert stacked.dtype == np.float32

def _run_short_episode(seed, steps=20):
    env = GymCoopEnv(level_name="level_3")
    obs, _ = env.reset(seed=seed)
    rewards = []
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(np.array([0, 0]))
        rewards.append(float(r))
        if term or trunc:
            break
    return rewards

def test_eval_determinism():
    # same seed should produce the exact same trajectory
    rewards_a = _run_short_episode(seed=42)
    rewards_b = _run_short_episode(seed=42)
    assert rewards_a == rewards_b
