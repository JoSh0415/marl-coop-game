import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1)) # Minimal dummy display for training

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from environment.gym_wrapper import GymCoopEnv

def train_shared():
    EXPERIMENT_NAME = "ppo_shared" # Model name
    
    models_dir = f"models/{EXPERIMENT_NAME}"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Fixed starting seed for reproducibility
    TRAIN_SEED = 12345

    # Create the gym environment
    env = make_vec_env(
        GymCoopEnv,
        n_envs=8,
        seed=TRAIN_SEED,
    )
    # Frame stacking for temporal context
    env = VecFrameStack(env, n_stack=4)

    # Create the initial PPO model with the game environment
    # Hyperparameters chosen for stability in Multi-Agent sparse reward tasks:
    # - n_steps=4096 allows for longer trajectories before updates
    # - ent_coef=0.01 encourages exploration
    # - net_arch=[512, 512, 256] for a deeper policy network to capture complex strategies
    model = PPO(
        "MlpPolicy", 
        env, 
        seed=TRAIN_SEED,
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=4096,
        batch_size=512,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=[512, 512, 256])
    )

    # Number of training timesteps (10 million)
    TIMESTEPS = 10000000
    
    # Save a checkpoint every 500k steps (adjusted for 8 parallel envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000/8,
        save_path=f"{models_dir}/",
        name_prefix="ppo_model"
    )

    # Train the model with the checkpoint callback 
    # to save intermediate models
    model.learn(
        total_timesteps=TIMESTEPS, 
        tb_log_name=EXPERIMENT_NAME,
        callback=checkpoint_callback
    )

    # Save the final model
    save_path = f"{models_dir}/{TIMESTEPS}_steps"
    model.save(save_path)

if __name__ == "__main__":
    train_shared()