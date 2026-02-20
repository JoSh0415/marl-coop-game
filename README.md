# Multi-Agent Cooperative Cooking Environment (MARL Project)

A custom two-agent grid-world environment (inspired by Overcooked) built for my Final Year Project.  
The aim is to evaluate how different multi-agent learning setups affect coordination, task sharing, and reliability across increasingly difficult layouts.

This repo contains the environment, training scripts, and evaluation code used to compare three PPO-based approaches under the same levels + reward design.

---

## What this project is testing

Two agents must cooperate to:
- collect ingredients,
- cook soups in pots,
- serve completed orders before time runs out,
- avoid getting stuck (collisions / blocking / wasted actions).

The main focus is **collaboration**, so agents share the overall task outcome and must learn to split roles naturally.

---

## Algorithms being compared

All experiments use PPO as the base optimiser, but with different multi-agent setups:

1. **Independent PPO**
   - Each agent has its **own policy** and learns from its own trajectory.
   - Trained in the same environment, with the same reward structure, but without sharing policy parameters.

2. **Shared Policy PPO (shared policy, shared reward)**
   - Both agents use the **same policy network** (parameter sharing).
   - Training is based on a **joint objective** (shared return), so the policy is encouraged to learn cooperative behaviour.

3. **Shared Policy PPO + Communication**
   - Same shared-policy setup as (2), but agents also exchange a small learned message (or communication vector).
   - The goal is to test whether explicit comms improves coordination compared to just sharing policy + reward.

---

## Repository layout

- `environment/` — core game logic (`env.py`) and Gym wrapper (`gym_wrapper.py`)
- `agents/` — training code for each algorithm setup
- `scripts/` — evaluation + debugging scripts
- `models/` — saved checkpoints / trained agents
- `eval_results/` — outputs from evaluation runs (JSON, summaries, etc.)
- `tests/` — unit tests for environment behaviour

---

## Environment details

The environment is a grid map with counters and stations. Agents can move and interact to pick up, drop, cook, and serve.

### Action space

| Index | Action     | Description |
|------:|------------|-------------|
| 0     | Stay       | Do nothing |
| 1     | Up         | Move up |
| 2     | Down       | Move down |
| 3     | Left       | Move left |
| 4     | Right      | Move right |
| 5     | Interact   | Pick up / drop / add ingredient / serve |

### Observation space

The wrapper returns a fixed-size feature vector. It includes relative positions, what tile is in front, inventory encodings, pot/order state, and BFS distances to key stations.


### Grid legend

| Char | Object      | Function |
|:----:|-------------|----------|
| ` `  | Floor       | Walkable |
| `#`  | Counter     | Blocks movement; items can be placed |
| `P`  | Pot         | Cooking station |
| `S`  | Serving     | Deliver finished soup |
| `I`  | Onion box   | Dispenses onions |
| `J`  | Tomato box  | Dispenses tomatoes |
| `R`  | Bowl rack   | Dispenses bowls |
| `G`  | Garbage     | Deletes held item |

### Recipes

- **Onion soup:** 1 onion
- **Tomato soup:** 1 tomato
- **Mixed soup:** 1 onion + 1 tomato

(Exact recipe rules are enforced in the environment.)

---

## Reward design

The environment uses shaped rewards (plus terminal bonuses) to make training workable. The exact values are defined in `environment/env.py` under `reward_mode="shaped"`

**Note:** reward shaping is mainly used to make learning feasible and stable. For reporting results, the key metrics are still task outcomes (e.g. soups served / success rate), not just high return.

---

## Installation

### Requirements
- Python 3.10+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running training

Commands to run each training script:
```bash
python agents/train_independent.py
```
```bash
python agents/train_shared.py
```
```bash
python agents/train_shared_comms.py
```

---

## Evaluation

Example evaluation command:

```bash
python scripts/eval_model.py \                                              
  --model models/ppo_shared_level_1/ppo_shared_level_1_final.zip \
  --episodes 1000 \
  --seed 0 \
  --levels level_1 \
  --stack-n 4 \
  --deterministic
```

Evaluation outputs are saved to `eval_results/`.

Typical metrics reported include:

- average episode score (soups served),

- success/perfect rate,

- stability across multiple seeds/layouts.

---

## Testing

Commands to run the unit tests:

```bash
pytest tests/
```

---

## Notes / limitations

The environment is designed for controlled comparisons rather than being a full Overcooked clone.

Training stability is sensitive to reward shaping and observation design, so key config choices are kept consistent across all algorithms.