import os
import sys
import json
import csv
import argparse
from collections import deque, Counter
import numpy as np
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.display.set_mode((1, 1))

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

from environment.gym_wrapper_rllib_decentralised import GymCoopEnvRLlibDecentralised

# Function: Convert action index to movement delta (dx, dy)
def action_to_delta(action):
    if action == 0 or action == 5:
        return (0, 0)
    elif action == 1:
        return (0, -1)
    elif action == 2:
        return (0, 1)
    elif action == 3:
        return (-1, 0)
    elif action == 4:
        return (1, 0)
    return (0, 0)

# Function: Parse soup information from the holding string
def get_soup_info(holding_str):
    parts = holding_str.split("-", 2)
    if len(parts) != 3:
        return "bowl", "unknown", "invalid"
    return parts[0], parts[1], parts[2]

# Function: Check if a soup recipe matches any active unserved order
def is_wanted(env, recipe):
    onions, tomatoes = env._recipe_to_counts(recipe)
    if onions is None:
        return False

    for order in env.active_orders:
        if not order.get("served", False):
            if order["onions"] == onions and order["tomatoes"] == tomatoes:
                return True
    return False

# Function: Calculate statistics for a numeric key across all results
def calculate_stats(results, key):
    values = []
    for r in results:
        values.append(float(r.get(key, 0.0)))

    if len(values) == 0:
        return {"mean": 0.0, "sum": 0, "max": 0.0}

    return {
        "mean": float(np.mean(values)),
        "sum": int(np.sum(values)),
        "max": float(np.max(values))
    }

# Function: Calculate rate for a boolean key across all results
def calculate_rate(results, key):
    if len(results) == 0:
        return 0.0
    return float(np.mean([1.0 if r.get(key, False) else 0.0 for r in results]))

# Function: Run a single episode and collect detailed results
def run_episode(algo, level_name, seed, deterministic, stack_n, max_steps_cap):

    # Create the environment and reset it
    gym_env = GymCoopEnvRLlibDecentralised(
        {
            "level_name": level_name,
            "stack_n": stack_n,
            "render": False,
        }
    )

    obs, info = gym_env.reset(seed=seed)

    # Get the raw environment for detailed state access
    raw_env = gym_env.env
    total_reward = 0.0
    steps = 0

    # Event counters
    w_serve = 0
    nd_serve = 0
    w_pickup = 0
    b_pickup = 0
    w_add = 0
    wrong_pot_add_seeds = []

    # Extra diagnostics
    collision_attempts = 0
    stuck_penalty_steps = 0
    both_idle_steps = 0

    # Track previous holding states for both agents
    prev_h1 = raw_env.agent1_holding
    prev_h2 = raw_env.agent2_holding

    done = False
    terminated = False
    truncated = False

    # Main loop for the episode
    while not done:
        if max_steps_cap is not None and steps >= max_steps_cap:
            truncated = True
            break

        # Run the policy inference from the current observations
        action_1 = algo.compute_single_action(
            obs["agent_1"],
            policy_id="agent_1_policy",
            explore=not deterministic,
        )
        action_2 = algo.compute_single_action(
            obs["agent_2"],
            policy_id="agent_2_policy",
            explore=not deterministic,
        )

        a1 = int(action_1)
        a2 = int(action_2)

        # Collision attempt count
        dx1, dy1 = action_to_delta(a1)
        dx2, dy2 = action_to_delta(a2)
        cand1 = (raw_env.agent1_pos[0] + dx1, raw_env.agent1_pos[1] + dy1)
        cand2 = (raw_env.agent2_pos[0] + dx2, raw_env.agent2_pos[1] + dy2)
        a1_pos = tuple(raw_env.agent1_pos)
        a2_pos = tuple(raw_env.agent2_pos)

        a1_hits_a2 = (cand1 == a2_pos)
        a2_hits_a1 = (cand2 == a1_pos)
        if a1_hits_a2 or a2_hits_a1:
            collision_attempts += 1

        # Count both-idle steps
        if a1 == 0 and a2 == 0:
            both_idle_steps += 1

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
                    new_on = raw_env.pot_onions + (1 if h == "onion" else 0)
                    new_to = raw_env.pot_tomatoes + (1 if h == "tomato" else 0)
                    if raw_env._get_target_order_for_pot_contents(new_on, new_to) is None:
                        w_add += 1
                        wrong_pot_add_seeds.append(seed)

        # Check Agent 2 pot errors
        if a2 == 5:
            h = raw_env.agent2_holding
            if h in ["onion", "tomato"]:
                tile, _ = raw_env.tile_in_front(raw_env.agent2_pos, raw_env.agent2_dir)
                if tile == "P" and raw_env.pot_state == "idle":
                    new_on = raw_env.pot_onions + (1 if h == "onion" else 0)
                    new_to = raw_env.pot_tomatoes + (1 if h == "tomato" else 0)
                    if raw_env._get_target_order_for_pot_contents(new_on, new_to) is None:
                        w_add += 1
                        wrong_pot_add_seeds.append(seed)

        # Take the step in the environment
        obs, rewards, terms, truncs, info = gym_env.step(
            {
                "agent_1": a1,
                "agent_2": a2,
            }
        )

        # Shared team reward is duplicated for both agents, so only count one copy
        total_reward += float(rewards["agent_1"])
        steps += 1

        terminated = bool(terms["__all__"])
        truncated = bool(truncs["__all__"])
        if terminated or truncated:
            done = True

        # Count stuck penalty steps
        if hasattr(raw_env, "stuck_steps") and raw_env.stuck_steps >= 80:
            stuck_penalty_steps += 1

        # Check pickups after the step
        curr_h1 = raw_env.agent1_holding
        if prev_h1 == "bowl" and isinstance(curr_h1, str) and curr_h1.startswith("bowl-"):
            _, state, recipe = get_soup_info(curr_h1)
            if state == "done":
                if not is_wanted(raw_env, recipe):
                    w_pickup += 1
            elif state == "burnt":
                b_pickup += 1

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

    # Episode outcome
    score = int(raw_env.score)
    failures = int(len(raw_env.failed_orders))

    # How many orders remain unserved at end
    unserved_active = 0
    for o in raw_env.active_orders:
        if not o.get("served", False):
            unserved_active += 1

    pending_left = int(len(raw_env.pending_orders))
    completed = int(len(raw_env.completed_orders))

    # Perfect definition
    perfect = (score == 3 and failures == 0)

    # End reason
    end_reason = "terminated"
    if truncated:
        end_reason = "truncated"
    if max_steps_cap is not None and steps >= max_steps_cap:
        end_reason = "cap_truncated"

    result = {
        "level": level_name,
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "steps": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "end_reason": end_reason,

        "score": score,
        "failed_orders": failures,
        "unserved_active_end": int(unserved_active),
        "pending_left_end": int(pending_left),
        "completed_orders": completed,

        "perfect": bool(perfect),
        "total_reward": float(total_reward),

        "wrong_serve_attempts": int(w_serve),
        "not_done_serve_attempts": int(nd_serve),
        "wrong_done_soup_pickups": int(w_pickup),
        "burnt_soup_pickups": int(b_pickup),
        "wrong_pot_adds": int(w_add),

        "collision_attempts": int(collision_attempts),
        "stuck_penalty_steps": int(stuck_penalty_steps),
        "both_idle_steps": int(both_idle_steps),
    }

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", nargs="+", default=["level_3"])
    parser.add_argument("--stack-n", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--out-dir", type=str, default="eval_results")
    parser.add_argument("--max-steps-cap", type=int, default=None)

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint not found")
        sys.exit(1)

    # Register env so restore works
    def env_creator(env_config):
        return GymCoopEnvRLlibDecentralised(env_config)

    register_env("marl_coop_decentralised", env_creator)

    # Start Ray and restore the trained algorithm
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    algo = Algorithm.from_checkpoint(checkpoint_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    for level in args.levels:
        # Run evaluations
        all_results = []
        for i in range(args.episodes):
            current_seed = args.seed + i
            res = run_episode(algo, level, current_seed, args.deterministic, args.stack_n, args.max_steps_cap)
            all_results.append(res)

        # Aggregate summary metrics
        scores = [r["score"] for r in all_results]
        perfect_rate = calculate_rate(all_results, "perfect")
        trunc_rate = calculate_rate(all_results, "truncated")

        reason_counts = Counter([r["end_reason"] for r in all_results])
        wrong_pot_add_seeds = list(set(r["seed"] for r in all_results if r["wrong_pot_adds"] > 0))

        summary = {
            "level": level,
            "n_episodes": len(all_results),

            "perfect_rate": float(perfect_rate),
            "truncated_rate": float(trunc_rate),

            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_min": int(min(scores)) if scores else 0,
            "score_max": int(max(scores)) if scores else 0,

            "failed_orders": calculate_stats(all_results, "failed_orders"),
            "unserved_active_end": calculate_stats(all_results, "unserved_active_end"),
            "steps": calculate_stats(all_results, "steps"),
            "total_reward": calculate_stats(all_results, "total_reward"),

            "end_reasons": dict(reason_counts),

            "events": {
                "wrong_serve_attempts": calculate_stats(all_results, "wrong_serve_attempts"),
                "not_done_serve_attempts": calculate_stats(all_results, "not_done_serve_attempts"),
                "wrong_done_soup_pickups": calculate_stats(all_results, "wrong_done_soup_pickups"),
                "burnt_soup_pickups": calculate_stats(all_results, "burnt_soup_pickups"),
                "wrong_pot_adds": calculate_stats(all_results, "wrong_pot_adds"),
                "wrong_pot_add_seeds": wrong_pot_add_seeds,

                "collision_attempts": calculate_stats(all_results, "collision_attempts"),
                "stuck_penalty_steps": calculate_stats(all_results, "stuck_penalty_steps"),
                "both_idle_steps": calculate_stats(all_results, "both_idle_steps"),
            }
        }

        print(json.dumps(summary, indent=2))

        # Save per-episode CSV and summary JSON
        base_name = os.path.basename(checkpoint_dir.rstrip("/"))
        csv_file = os.path.join(args.out_dir, f"eval_{base_name}_{level}.csv")
        json_file = os.path.join(args.out_dir, f"eval_{base_name}_{level}.summary.json")

        with open(csv_file, "w", newline="") as f:
            if len(all_results) > 0:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                for r in all_results:
                    writer.writerow(r)

        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)

    ray.shutdown()