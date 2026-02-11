import os
import sys
import json
import csv
import argparse
from collections import deque
import numpy as np
import pygame
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.display.set_mode((1, 1))

from environment.gym_wrapper import GymCoopEnv

# Function: Parse soup information from the holding string
def get_soup_info(holding_str):
    parts = holding_str.split("-", 2)
    if len(parts) != 3:
        return "bowl", "unknown", "invalid"
    return parts[0], parts[1], parts[2]

# Function: Check if a served soup matches any active order
def is_wanted(env, recipe):
    onions, tomatoes = env._recipe_to_counts(recipe)
    if onions is None:
        return False
    
    for order in env.active_orders:
        if not order.get("served", False):
            if order["onions"] == onions and order["tomatoes"] == tomatoes:
                return True
    return False

# Function: Calculate statistics for an event across all results
def calculate_stats(results, key):
    values = []
    for r in results:
        values.append(r[key])
    
    return {
        "mean": float(np.mean(values)),
        "sum": int(np.sum(values)),
        "max": float(np.max(values))
    }

# Function: Run a single episode and collect detailed results
def run_episode(model, level_name, seed, deterministic, stack_size, max_steps):

    # Create the environment and reset it
    gym_env = GymCoopEnv(level_name)
    obs, info = gym_env.reset(seed=seed)
    
    # Initialize the frame stack
    frames = deque(maxlen=stack_size)
    for _ in range(stack_size):
        frames.append(obs.copy())
    
    # Get the raw environment for detailed state access
    # and initialize tracking variables
    raw_env = gym_env.env
    total_reward = 0.0
    steps = 0
    
    w_serve = 0
    nd_serve = 0
    w_pickup = 0
    b_pickup = 0
    w_add = 0
    
    # Track previous holding states for both agents
    prev_h1 = raw_env.agent1_holding
    prev_h2 = raw_env.agent2_holding
    
    done = False
    
    # Main loop for the episode
    while not done:
        if max_steps is not None and steps >= max_steps:
            break
        
        # Prepare the stacked observation and run the model prediction
        obs_stack = np.concatenate(list(frames), axis=0).astype(np.float32)
        action, _ = model.predict(obs_stack, deterministic=deterministic)
        
        a1 = int(action[0])
        a2 = int(action[1])
        
        # Check Agent 1 serve errors
        if a1 == 5:
            tile, _ = raw_env.tile_in_front(raw_env.agent1_pos, raw_env.agent1_dir)
            if tile == "S":
                h = raw_env.agent1_holding
                if isinstance(h, str) and h.startswith("bowl-"):
                    _, state, recipe = get_soup_info(h)
                    if state != "done":
                        nd_serve += 1
                    elif not is_wanted(raw_env, recipe):
                        w_serve += 1

        # Check Agent 2 serve errors
        if a2 == 5:
            tile, _ = raw_env.tile_in_front(raw_env.agent2_pos, raw_env.agent2_dir)
            if tile == "S":
                h = raw_env.agent2_holding
                if isinstance(h, str) and h.startswith("bowl-"):
                    _, state, recipe = get_soup_info(h)
                    if state != "done":
                        nd_serve += 1
                    elif not is_wanted(raw_env, recipe):
                        w_serve += 1
                        
        # Check Agent 1 pot errors
        if a1 == 5:
            h = raw_env.agent1_holding
            if h in ["onion", "tomato"]:
                tile, _ = raw_env.tile_in_front(raw_env.agent1_pos, raw_env.agent1_dir)
                if tile == "P" and raw_env.pot_state == "idle":
                    if not raw_env._ingredient_useful_for_pot(h):
                        w_add += 1

        # Check Agent 2 pot errors
        if a2 == 5:
            h = raw_env.agent2_holding
            if h in ["onion", "tomato"]:
                tile, _ = raw_env.tile_in_front(raw_env.agent2_pos, raw_env.agent2_dir)
                if tile == "P" and raw_env.pot_state == "idle":
                    if not raw_env._ingredient_useful_for_pot(h):
                        w_add += 1

        # Take the step in the environment and update the frame stack
        obs, reward, terminated, truncated, info = gym_env.step(np.array([a1, a2], dtype=np.int64))
        frames.append(obs.copy())
        
        # Update reward and step count
        total_reward += float(reward)
        steps += 1
        
        if terminated or truncated:
            done = True
            
        # Check Agent 1 pickups
        curr_h1 = raw_env.agent1_holding
        if prev_h1 == "bowl" and isinstance(curr_h1, str) and curr_h1.startswith("bowl-"):
            _, state, recipe = get_soup_info(curr_h1)
            if state == "done":
                if not is_wanted(raw_env, recipe):
                    w_pickup += 1
            elif state == "burnt":
                b_pickup += 1
        
        # Check Agent 2 pickups
        curr_h2 = raw_env.agent2_holding
        if prev_h2 == "bowl" and isinstance(curr_h2, str) and curr_h2.startswith("bowl-"):
            _, state, recipe = get_soup_info(curr_h2)
            if state == "done":
                if not is_wanted(raw_env, recipe):
                    w_pickup += 1
            elif state == "burnt":
                b_pickup += 1
                
        prev_h1 = curr_h1
        prev_h2 = curr_h2
    
    # Group the results for this episode
    score = int(raw_env.score)
    failures = len(raw_env.failed_orders)
    perfect = False
    if score == 3 and failures == 0:
        perfect = True
        
    result = {
        "level": level_name,
        "seed": seed,
        "deterministic": deterministic,
        "steps": steps,
        "score": score,
        "failed_orders": failures,
        "perfect": perfect,
        "total_reward": total_reward,
        "wrong_serve_attempts": w_serve,
        "not_done_serve_attempts": nd_serve,
        "wrong_done_soup_pickups": w_pickup,
        "burnt_soup_pickups": b_pickup,
        "wrong_pot_adds": w_add
    }
    
    return result

if __name__ == "__main__":

    # Parse command line arguments to configure the evaluation run
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", nargs="+", default=["level_1"])
    parser.add_argument("--stack-n", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--out-dir", type=str, default="eval_results")
    parser.add_argument("--max-steps-cap", type=int, default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print("Model not found")
        sys.exit(1)
    
    # Load the model
    model = PPO.load(args.model)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Run evaluation across specified levels and episodes
    for level in args.levels:

        # Run all episodes and collect results        
        all_results = []
        for i in range(args.episodes):
            # Use a different seed for each episode to get a range of outcomes
            current_seed = args.seed + i
            res = run_episode(model, level, current_seed, args.deterministic, args.stack_n, args.max_steps_cap)
            all_results.append(res)
        
        # Collect the score and perfect completion statistics
        perfect_count = 0
        score_sum = 0
        for r in all_results:
            if r["perfect"]:
                perfect_count += 1
            score_sum += r["score"]
            
        perfect_rate = perfect_count / len(all_results)
        avg_score = score_sum / len(all_results)
        scores = [r["score"] for r in all_results]
        
        # Create a summary of the full evaluation results
        summary = {
            "n_episodes": len(all_results),
            "perfect_rate": perfect_rate,
            "score_mean": avg_score,
            "score_min": min(scores),
            "score_max": max(scores),
            "events": {
                "wrong_serve_attempts": calculate_stats(all_results, "wrong_serve_attempts"),
                "not_done_serve_attempts": calculate_stats(all_results, "not_done_serve_attempts"),
                "wrong_done_soup_pickups": calculate_stats(all_results, "wrong_done_soup_pickups"),
                "burnt_soup_pickups": calculate_stats(all_results, "burnt_soup_pickups"),
                "wrong_pot_adds": calculate_stats(all_results, "wrong_pot_adds")
            }
        }
        
        # Print the summary
        print(json.dumps(summary, indent=2))
        
        base_name = os.path.basename(args.model).replace(".zip", "")
        csv_file = os.path.join(args.out_dir, f"eval_{base_name}.csv")
        json_file = os.path.join(args.out_dir, f"eval_{base_name}.summary.json")
        
        # Save the detailed results to CSV and the summary to JSON
        with open(csv_file, "w", newline="") as f:
            if len(all_results) > 0:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                for r in all_results:
                    writer.writerow(r)
                    
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)