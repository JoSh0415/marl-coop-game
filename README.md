# Multi-Agent Cooperative Cooking Environment (Final Year Project)

This repository contains the code for my final-year project on multi-agent reinforcement learning in a custom two-agent cooperative cooking game.

The project is not just about getting a policy to score highly. The main aim is to compare different coordination setups in a controlled way and see how much the learning setup itself changes performance.

The current study compares PPO-based controllers under the same task, reward design, and levels.

---

## What the project is about

Two agents have to work together to complete soup orders before they expire.

A full order usually involves:

- picking up ingredients
- adding them to the pot
- waiting for cooking to finish
- getting a bowl
- taking the finished soup out
- sometimes passing it across a counter
- serving it before the deadline

The environment is designed so that coordination matters. Depending on the layout, the agents may need to share space, avoid blocking each other, or hand items across a partition.

The research question is basically: **how much does coordination change when I change what the agents are allowed to know?**

---

## Current algorithm setups

The project uses PPO in RLlib for all official results.

### 1. Fully centralised PPO

- One joint controller chooses both agents' actions.
- It sees a shared joint observation of the full state.
- This is the strongest reference baseline.

### 2. Decentralised PPO (no communication)

- Two separate policies: one for each agent.
- Shared team reward.
- Each agent sees its own local / embodied features plus shared public task-state features.
- Teammate-private action-ready information is masked out.

This is the main no-comms baseline.

### 3. Decentralised PPO + communication

- Planned next.
- This will keep the decentralised setup but add a small explicit message channel.
- The goal is to test whether explicit communication helps most in the layouts where pure decentralised learning struggles.

Older Stable-Baselines3 experiments were useful while the project was being built, but the final benchmark uses RLlib so the comparison is consistent.

---

## Main result so far

The current no-comms decentralised baseline behaves very differently depending on the level.

### Centralised PPO (final 2500-episode results)

- **level_1:** perfect rate `0.9584`, score mean `2.9388`
- **level_2:** perfect rate `0.9888`, score mean `2.9888`
- **level_3:** perfect rate `0.9100`, score mean `2.8660`

Centralised ordering:

**level_3 < level_1 < level_2**

### Pure decentralised PPO (final 2500-episode results)

- **level_1 (9M):** perfect rate `0.0040`, score mean `1.2400`
- **level_2 (8M):** perfect rate `0.0052`, score mean `1.1596`
- **level_3 (10M):** perfect rate `0.8752`, score mean `2.8184`

Decentralised ordering:

**level_2 < level_1 < level_3**

This is the most interesting finding in the repo at the moment.

The decentralised agents are not just uniformly worse. They are much weaker in the partitioned and bottleneck layouts, but they are surprisingly strong in the open obstacle layout. That gives the communication experiment a clear purpose.

---

## Repository layout

The main folders are:

- `environment/`
  - `env.py` - core cooking task logic
  - `levels.py` - fixed layouts
  - `gym_wrapper_rllib_centralised.py` - centralised RLlib wrapper
  - `gym_wrapper_rllib_decentralised.py` - decentralised RLlib wrapper

- `agents/`
  - `train_centralised_rllib.py` - RLlib training for the joint controller
  - `train_decentralised_rllib.py` - RLlib training for the no-comms decentralised baseline
  - communication training script to be added next

- `scripts/`
  - `eval_centralised_rllib.py` - evaluation for the centralised benchmark
  - `eval_decentralised_rllib.py` - evaluation for the decentralised baseline
  - debug / visualisation scripts for checking policy behaviour

- `models/`
  - saved checkpoints for training runs

- `eval_results/`
  - per-episode CSVs and summary JSON files from evaluation sweeps

- `tests/`
  - optional environment tests / sanity checks

---

## Environment summary

This is a two-agent grid-world, loosely inspired by Overcooked, but simplified and built specifically for controlled MARL experiments.

### Action space

| Index | Action | Meaning |
|------:|:-------|:--------|
| 0 | Stay | Do nothing |
| 1 | Up | Move up |
| 2 | Down | Move down |
| 3 | Left | Move left |
| 4 | Right | Move right |
| 5 | Interact | Pick up, drop, add ingredient, serve, etc. |

### Grid legend

| Char | Object | Meaning |
|:----:|:-------|:--------|
| ` ` | Floor | Walkable |
| `#` | Counter | Blocks movement, can hold items |
| `P` | Pot | Cooking station |
| `S` | Serving | Deliver finished soup |
| `I` | Onion box | Dispenses onions |
| `J` | Tomato box | Dispenses tomatoes |
| `R` | Bowl rack | Dispenses bowls |
| `G` | Garbage | Deletes held item |
| `A` | Agent 1 spawn | Start position |
| `B` | Agent 2 spawn | Start position |

### Recipes

- **Onion soup:** 1 onion
- **Tomato soup:** 1 tomato
- **Onion and Tomato soup:** 1 onion + 1 tomato

Each episode has exactly 3 orders, released over time.

---

## Reward design

The environment uses a shaped reward, but the shaping is fixed and part of the task definition, not something I keep changing between algorithms.

Key points:

- step penalty: `-0.01`
- failed order: `-2.0`
- valid ingredient added: `+1.0`
- cooking completes: `+0.5`
- correct soup pickup: `+2.0`
- correct serve: `+20.0 + time bonus`
- perfect episode bonus: `+10.0`

There are also penalties for invalid pot adds, burnt soup, and bad serves.

The intermediate rewards are capped to stop reward farming, so the agents cannot get high return just by repeating useless cooking behaviour.

The exact logic is in `environment/env.py`.

---

## Observation design

The base observation format is a **74-feature vector per frame**, stacked over **4 frames**.

### Centralised wrapper

The centralised policy gets the full joint state in a shared 74-feature observation layout.

### Decentralised wrapper

The decentralised policies keep the same 74-slot layout for fairness, but only their own local / embodied feature blocks are live.

They still get shared public task-state features such as:

- pot state
- pot contents
- pot timer
- next-order info

But they do **not** get live teammate-private information such as:

- teammate holding state
- teammate front-tile block
- teammate BFS block

Those slots are masked out.

This means the no-comms baseline is genuinely decentralised at execution, while still being a fair comparison against the same underlying task.

---

## Training setup

All official runs use RLlib PPO with the same overall training family.

Shared settings:

- 8 parallel envs
- rollout fragment length `4096`
- train batch size `32768`
- minibatch size `512`
- 12 PPO epochs
- learning rate `2e-4`
- `gamma = 0.995`
- `lambda = 0.97`
- model MLP: `[512, 512, 256]`
- `tanh` activations
- checkpoints every `500k` env steps
- total budget: `10M` env steps per run

The main difference between the scripts is the policy structure and wrapper, not the overall PPO family.

---

## Running training

### Centralised PPO

```bash
python agents/train_centralised_rllib.py
```

### Decentralised PPO (no comms)

```bash
python agents/train_decentralised_rllib.py
```

The communication variant will get its own training script once it is implemented.

---

## Running evaluation

### Example: decentralised evaluation

```bash
python scripts/eval_decentralised_rllib.py \
  --checkpoint models/ppo_decentralised_level_3/checkpoints/checkpoint_10000000 \
  --episodes 2500 \
  --seed 10000 \
  --levels level_3 \
  --deterministic
```

### Example: centralised evaluation

```bash
python scripts/eval_centralised_rllib.py \
  --checkpoint models/ppo_centralised_level_2/checkpoints/checkpoint_9000000 \
  --episodes 2500 \
  --seed 10000 \
  --levels level_2 \
  --deterministic
```

The evaluation scripts save:

- a per-episode CSV
- a summary JSON

under `eval_results/`.

---

## Checkpoint selection

Checkpoints are first swept on a smaller validation set (250 episodes, seeds `0-249`), then the chosen checkpoint is confirmed on the final 2500-episode test set (seeds `10000-12499`).

For strong models, perfect rate is still useful.

For weaker decentralised models, perfect episodes can be too sparse to use on their own, so checkpoint selection is based on the **best overall validation profile**, with score mean treated as the most stable signal of actual task completion.

That avoids picking a weaker model just because it got a handful of lucky perfect episodes.

---

## Current limitations

- The communication variant is not finished yet.
- The pure decentralised baseline is intentionally strict, so some layouts (especially `level_2`) are hard.
- Results are meaningful but still depend on level structure, so conclusions have to be made per layout rather than as one single blanket statement.

That is also part of the point of the project: the same algorithm design does not behave the same way under different coordination demands.
