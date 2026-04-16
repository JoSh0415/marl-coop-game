# Multi-Agent Cooperative Cooking Environment (Final Year Project)

This repository contains the code for my final-year project on multi-agent reinforcement learning in a custom two-agent cooperative cooking game.

The project is not just about getting a policy to score highly. The main aim is to compare different coordination setups in a controlled way and see how much the learning setup itself changes performance.

The current study compares three PPO-based controllers under the same task, reward design, and levels.

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

## Final algorithm setups

The project uses PPO in RLlib for all official results.

### 1. Fully centralised PPO

- One joint controller chooses both agents' actions.
- It sees a shared joint observation of the full state.
- This is the strongest reference baseline.

### 2. Decentralised PPO (no cue)

- Two separate policies: one for each agent.
- Shared team reward.
- Each agent sees its own local / embodied features plus shared public task-state features.
- Teammate-private action-ready information is masked out.

### 3. Decentralised PPO + lightweight teammate task-state cue

- Same decentralised setup: still two separate policies and still decentralised at execution.
- Adds a small 4-slot teammate task-state signal in the final observation block.
- That signal only tells the agent whether the teammate is currently holding:
  - an onion
  - a tomato
  - a bowl
  - a ready soup

This is a lightweight teammate task-state add-on, not full state sharing.

Older Stable-Baselines3 experiments were useful while the project was being built, but the final benchmark uses RLlib so the comparison is consistent.

---

## Final benchmark summary

The finished benchmark gives a clear three-way comparison across all three levels.

### Fully centralised PPO (final 2500-episode results)

| Level | Perfect rate | Score mean |
|:------|-------------:|-----------:|
| level_1 | 0.9584 | 2.9388 |
| level_2 | 0.9888 | 2.9888 |
| level_3 | 0.9100 | 2.8660 |

### Pure decentralised PPO (final 2500-episode results)

| Level | Perfect rate | Score mean |
|:------|-------------:|-----------:|
| level_1 | 0.0040 | 1.2400 |
| level_2 | 0.0052 | 1.1596 |
| level_3 | 0.8752 | 2.8184 |

### Decentralised PPO + lightweight teammate task-state cue (final 2500-episode results)

| Level | Perfect rate | Score mean |
|:------|-------------:|-----------:|
| level_1 | 0.9668 | 2.9592 |
| level_2 | 0.3212 | 2.0860 |
| level_3 | 0.6792 | 2.4384 |

---

## Repository layout

The main folders are:

- `environment/`
  - `env.py` - core cooking task logic
  - `levels.py` - fixed layouts
  - `gym_wrapper_rllib_centralised.py` - centralised RLlib wrapper
  - `gym_wrapper_rllib_decentralised.py` - decentralised RLlib wrapper
  - `gym_wrapper_rllib_decentralised_comms.py` - decentralised RLlib wrapper with task-state cue

- `agents/`
  - `train_centralised_rllib.py` - RLlib training for the joint controller
  - `train_decentralised_rllib.py` - RLlib training for the no-cue decentralised baseline
  - `train_decentralised_comms_rllib.py` - RLlib training script for the decentralised task-state cue variant

- `scripts/`
  - `eval_centralised_rllib.py` - evaluation for the centralised benchmark
  - `eval_decentralised_rllib.py` - evaluation for the decentralised baseline
  - `eval_decentralised_comms_rllib.py` - evaluation for the decentralised task-state cue benchmark
  - `run_eval_sweep_parallel.py` - helper for running checkpoint sweep evaluations
  - `plot_*.py` and `generate_sparse_table.py` - analysis scripts used to generate dissertation figures/tables
  - debug / visualisation scripts for checking policy behaviour

- `models/`
  - saved checkpoints for training runs
  - `best_checkpoint/` - best performing checkpoint for each level/algorithm combination

- `logs/`
  - RLlib training logs per run (including `progress.csv`, `result.json`, and TensorBoard event files)

- `eval_sweeps/`
  - checkpoint sweep outputs (per-checkpoint CSV + summary JSON) used for learning-curve plots and checkpoint selection

- `eval_results/`
  - final/best-checkpoint per-episode CSVs and summary JSON files used for locked benchmark tables and behavioural diagnostics

- `analytics/`
  - generated figures and tables used in the dissertation write-up
  - see `analytics/README.md` for the script/input/output mapping

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

### Task-state cue wrapper

The task-state cue wrapper keeps the same decentralised structure and the same overall observation size.

The only intentional difference from the no-cue baseline is that the final 4-slot comparison block is no longer masked. Instead, it carries a coarse teammate task-state signal:

- teammate holding onion
- teammate holding tomato
- teammate holding bowl
- teammate holding ready soup

So it is still decentralised, but it adds a small dense coordination cue.

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

### Decentralised PPO (no cue)

```bash
python agents/train_decentralised_rllib.py
```

### Decentralised PPO + lightweight teammate task-state cue

```bash
python agents/train_decentralised_comms_rllib.py
```

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

### Example: decentralised task-state cue evaluation

```bash
python scripts/eval_decentralised_comms_rllib.py \
  --checkpoint models/ppo_decentralised_comms_level_2/checkpoints/checkpoint_10000000 \
  --episodes 2500 \
  --seed 10000 \
  --levels level_2 \
  --deterministic
```

### Example: centralised evaluation

```bash
python scripts/eval_centralised_rllib.py \
  --checkpoint models/ppo_centralised_level_1/checkpoints/checkpoint_9000000 \
  --episodes 2500 \
  --seed 10000 \
  --levels level_1 \
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

## Analysis and figures

The analysis side of the repo sits on top of saved evaluation outputs.

- the figure/table scripts live in `scripts/`
- generated figure/table artifacts are saved to `analytics/`
- detailed input/output mapping is documented in `analytics/README.md`

### Analysis inputs

- `eval_sweeps/`:
  - validation checkpoint sweep outputs (`eval_checkpoint_<step>_level_<n>.summary.json` and matching CSVs)
  - used by learning-curve plots across 500k -> 10M checkpoints
- `eval_results/`:
  - best/final per-episode CSVs and summary JSONs
  - used by behavioural diagnostics, role split plots, and sparse error table generation
- `logs/`:
  - contains `progress.csv` and other RLlib training logs, but the current dissertation plotting scripts do not read these files directly

### Generated outputs

The current analysis scripts generate the following files under `analytics/`:

- `score_mean_over_checkpoints.png` - learning-curve view using `score_mean` across checkpoints
- `perfect_rate_over_checkpoints.png` - learning-curve view using perfect-episode rate across checkpoints
- `behaviour_rates_barchart.png` - behavioural diagnostics (collision attempts, wrong pot adds, both-idle steps)
- `role_split_lollipop.png` - role-split / division-of-labour plot (Agent 1 share vs Agent 2 share)
- `sparse_error_incidence_table.csv` - raw sparse-error incidence values
- `sparse_error_incidence_table.md` - markdown version of the sparse-error table

### Typical workflow

1. Train each RLlib controller and save checkpoints under `models/` with logs under `logs/`.
2. Run validation checkpoint sweeps (250 episodes, seeds `0-249`) and save outputs under `eval_sweeps/`.
3. Choose checkpoints using the benchmark rule, then run final deterministic evaluation (2500 episodes, seeds `10000-12499`) and save under `eval_results/`.
4. Run the analysis scripts in `scripts/` to turn `eval_sweeps/` and `eval_results/` into dissertation-ready figures/tables in `analytics/`.
