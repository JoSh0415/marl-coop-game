import sys
import os
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from environment.gym_wrapper_rllib_decentralised import GymCoopEnvRLlibDecentralised
from ray.tune.registry import register_env

# Colors for the UI
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
DARK_BG = (20, 20, 30)

# Function: Draw text on the screen
def draw_text(screen, text, x, y, size=20, color=WHITE):
    font = pygame.font.SysFont("consolas", size, bold=True)
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

# Function: Create the RLlib environment
def env_creator(env_config):
    return GymCoopEnvRLlibDecentralised(env_config)

# Main function to run the RLlib debug agent
def debug_agent():
    # Initialize Pygame and set up the display
    pygame.init()

    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MARL Surgical Debugger")

    # Register environment and start Ray
    register_env("marl_coop_decentralised", env_creator)
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Set seed for reproducibility
    SEED = 10021

    # Create the environment
    env = GymCoopEnvRLlibDecentralised(
        {
            "level_name": "level_1",
            "stack_n": 4,
            "render": True,
        }
    )
    obs, info = env.reset(seed=SEED)

    # Load the trained PPO model
    checkpoint_dir = os.path.abspath("models/ppo_decentralised_level_1/checkpoints/checkpoint_9000000")

    if not os.path.exists(checkpoint_dir):
        print(f"Error: checkpoint not found at {checkpoint_dir}")
        return

    algo = Algorithm.from_checkpoint(checkpoint_dir)

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

        if not paused or step_once:
            
            # Predict actions
            action_1 = algo.compute_single_action(
                obs["agent_1"],
                policy_id="agent_1_policy",
                explore=False,
            )
            action_2 = algo.compute_single_action(
                obs["agent_2"],
                policy_id="agent_2_policy",
                explore=False,
            )

            obs, rewards, terminateds, truncateds, info = env.step(
                {
                    "agent_1": int(action_1),
                    "agent_2": int(action_2),
                }
            )

            # Track rewards for agent 1
            last_reward = float(rewards["agent_1"])
            total_reward += float(rewards["agent_1"])
            step_once = False

            terminated = bool(terminateds["__all__"])
            truncated = bool(truncateds["__all__"])

            # Reset if episode ends
            if terminated or truncated:
                print(f"Episode done. Score: {env.env.score} and Total Reward: {total_reward:.2f}")
                obs, info = env.reset(seed=SEED)
                total_reward = 0.0
                last_reward = 0.0

        screen.fill(DARK_BG)

        # Draw the game environment
        env.env.render(screen)

        # Draw Debug Information Panel
        base_x = 680
        y = 20
        gap = 25

        draw_text(screen, "DEBUG INFO", base_x, y, color=(200, 200, 0))
        y += int(gap * 1.5)

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
        y += int(gap * 1.5)

        # Pot status
        draw_text(screen, f"Pot State: {env.env.pot_state}", base_x, y)
        y += gap
        draw_text(screen, f"Pot Timer: {env.env.pot_timer}", base_x, y)
        y += int(gap * 1.5)

        # Agent holding status
        draw_text(screen, f"A1 Hold: {env.env.agent1_holding}", base_x, y)
        y += gap
        draw_text(screen, f"A2 Hold: {env.env.agent2_holding}", base_x, y)
        y += int(gap * 1.5)

        # Order info
        active = [o for o in env.env.active_orders if not o.get("served", False)]
        draw_text(screen, f"Active Orders: {len(active)}", base_x, y)
        y += gap

        draw_text(screen, f"Orders Pending: {len(env.env.pending_orders)}", base_x, SCREEN_HEIGHT - 3 * gap)

        pygame.display.flip()
        clock.tick(30)

    ray.shutdown()
    pygame.quit()


if __name__ == "__main__":
    debug_agent()