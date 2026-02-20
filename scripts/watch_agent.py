import sys
import os
import time
import pygame
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.gym_wrapper import GymCoopEnv
from collections import deque

# Number of frames to stack for the observation
STACK_N = 4
stack = deque(maxlen=STACK_N)

# Handle VecFrameStack correctly by stacking the last STACK_N observations
def stacked_obs():
    return np.concatenate(list(stack), axis=0).astype(np.float32)

# Watch the trained agent in action
def watch():
    # Initialize Pygame and set up the display
    pygame.init()
    
    SCREEN_WIDTH = 660
    SCREEN_HEIGHT = 530
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MARL Coop Agent Watcher")

    # Create the environment with rendering enabled
    env = GymCoopEnv(render=True)
    obs, info = env.reset()

    # Clear the stack and fill it with the initial observation
    stack.clear()
    for _ in range(STACK_N):
        stack.append(obs.copy())

    # Load the trained PPO model
    model_path = "models/ppo_shared/ppo_shared_level_1_final.zip"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = PPO.load(model_path)

    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        
        # Get the action from the model using the stacked observation
        # Use deterministic=True to get the best action according to the policy
        action, _states = model.predict(stacked_obs(), deterministic=True)

        # Take a step in the environment using the action
        obs, reward, terminated, truncated, info = env.step(action)
        stack.append(obs.copy())
        
        # Render the environment
        env.env.render(screen)
        pygame.display.flip()
        
        time.sleep(0.1) 
        
        # Check if the episode has ended
        if terminated or truncated:
            print(f"Episode finished. Final Score: {env.env.score}")
            obs, info = env.reset()
            stack.clear()
            for _ in range(STACK_N):
                stack.append(obs.copy())
            time.sleep(1.0)

    pygame.quit()

if __name__ == "__main__":
    watch()