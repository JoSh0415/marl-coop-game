import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

import gymnasium as gym
from stable_baselines3 import PPO
from environment.gym_wrapper import GymCoopEnv

def train_baseline():
    EXPERIMENT_NAME = "ppo_baseline_independent"
    
    models_dir = f"models/{EXPERIMENT_NAME}"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = GymCoopEnv()
    env.reset()

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
    )

    TIMESTEPS = 1000000
        
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=EXPERIMENT_NAME)

    save_path = f"{models_dir}/{TIMESTEPS}_steps"
    model.save(save_path)

if __name__ == "__main__":
    train_baseline()