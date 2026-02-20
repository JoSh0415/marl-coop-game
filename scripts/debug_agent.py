import sys
import os
import time
import pygame 
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.gym_wrapper import GymCoopEnv

# Number of frames to stack for the observation
STACK_N = 4
stack = deque(maxlen=STACK_N)

# Function: Get stacked observations
def stacked_obs():
    return np.concatenate(list(stack), axis=0).astype(np.float32)

# Colors for the UI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
DARK_BG = (20, 20, 30)

# Function: Draw text on the screen
def draw_text(screen, text, x, y, size=20, color=WHITE):
    font = pygame.font.SysFont("consolas", size, bold=True)
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

# Main function to run the debug agent
def debug_agent():
    # Initialize Pygame and set up the display
    pygame.init()
    
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MARL Surgical Debugger")

    # Set seed for reproducibility
    SEED = 12345

    # Create the environment with rendering enabled
    env = GymCoopEnv("level_3", render=True)
    obs, info = env.reset(seed=SEED)
    
    # Initialize the observation stack
    stack.clear()
    for _ in range(STACK_N):
        stack.append(obs.copy())

    # Load the trained PPO model
    model_path = "models/ppo_shared_level_1/ppo_shared_level_1_best.zip" 
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = PPO.load(model_path)

    # Control variables
    running = True
    paused = False
    step_once = False
    
    total_reward = 0.0
    last_reward = 0.0
    
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT:
                    if paused: 
                        step_once = True

        # Update environment
        if not paused or step_once:
            # Predict action using the model
            # deterministic=True for consistent behavior during debugging
            action, _states = model.predict(stacked_obs(), deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            stack.append(obs.copy())
            
            last_reward = reward
            total_reward += reward
            step_once = False # Reset step flag

            # Reset if episode ends
            if terminated or truncated:
                print(f"Episode done. Score: {env.env.score} and Total Reward: {total_reward:.2f}")
                obs, info = env.reset(seed=SEED)
                stack.clear()
                for _ in range(STACK_N):
                    stack.append(obs.copy())
                total_reward = 0
                last_reward = 0
        
        screen.fill(DARK_BG)
        
        # Draw the game environment
        env.env.render(screen)

        # Draw Debug Information Panel
        base_x = 680
        y = 20
        gap = 25
        
        draw_text(screen, "DEBUG INFO", base_x, y, color=(200, 200, 0))
        y += gap * 1.5
        
        # Step count and score
        draw_text(screen, f"Step: {env.env.step_count}", base_x, y)
        y += gap
        draw_text(screen, f"Score: {env.env.score}", base_x, y)
        y += gap
        
        # Reward tracking
        reward_color = GREEN if last_reward > 0 else (RED if last_reward < 0 else WHITE)
        draw_text(screen, f"Last Rw: {last_reward:.4f}", base_x, y, color=reward_color)
        y += gap
        draw_text(screen, f"Total Rw: {total_reward:.2f}", base_x, y)
        y += gap * 1.5

        # Pot status
        draw_text(screen, f"Pot State: {env.env.pot_state}", base_x, y)
        y += gap
        draw_text(screen, f"Pot Timer: {env.env.pot_timer}", base_x, y)
        y += gap * 1.5
        
        # Agent holding status
        draw_text(screen, f"A1 Hold: {env.env.agent1_holding}", base_x, y)
        y += gap
        draw_text(screen, f"A2 Hold: {env.env.agent2_holding}", base_x, y)
        y += gap * 1.5

        # Order info
        active = [o for o in env.env.active_orders if not o.get("served", False)]
        draw_text(screen, f"Active Orders: {len(active)}", base_x, y)
        y += gap

        draw_text(screen, f"Orders Pending: {len(env.env.pending_orders)}", base_x, SCREEN_HEIGHT - 3 * gap)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    debug_agent()