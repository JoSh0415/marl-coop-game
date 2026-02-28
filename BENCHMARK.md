# Benchmark

This file is the frozen benchmark for the project. The idea is simple: the task itself stays frozen, and only the learning setup changes. That way the comparisons are actually about coordination strategy, not about me quietly changing the environment between runs.

The reward function, level layouts, episode limits, order generation, and evaluation procedure are shared across all algorithm variants.

---

## 1. Scope

The project compares three PPO-based multi-agent setups in the same two-agent cooking task:

1. **Decentralised PPO (no communication)**
   - Two separate policies, one per agent.
   - Shared team reward.
   - Each agent sees its own local / embodied features plus shared public task-state features.
   - This is the main no-comms baseline.

2. **Decentralised PPO + communication**
   - Same decentralised setup as above, but with a small explicit message channel.
   - This is the next planned experiment.
   - It should stay clearly decentralised: communication is the only extra information source.

3. **Fully centralised PPO**
   - A single joint controller chooses both agents' actions from a shared joint observation.
   - This is the reference benchmark, not the deployment target.
   - It is expected to be the strongest overall because it sees the full joint state.

All official runs are in **RLlib**. Older Stable-Baselines3 runs were useful during development, but they are not part of the final apples-to-apples comparison.

---

## 2. What is frozen and what is allowed to differ

The following are frozen across all variants:

- same `environment/env.py` task logic
- same shaped reward function
- same three layouts
- same episode cap (`max_steps = 1000`)
- same order timing rules (`order_time = 450`)
- same seed splits for validation vs final evaluation
- same checkpoint cadence (every 500k env steps)
- same PPO training family in RLlib

The intended differences are:

- **Observation design**
  - Centralised PPO uses a shared joint observation.
  - Decentralised PPO uses per-agent observations derived from the same base state, but without teammate-private action-ready information.

- **Policy structure**
  - Centralised PPO uses one joint policy.
  - Decentralised PPO uses two separate policies (`agent_1_policy`, `agent_2_policy`).

- **Communication**
  - Only the comms variant will add an explicit message channel.
  - It should not add direct access to hidden state.

In short: the task is frozen, while the information available to the learner is the thing being varied on purpose.

---

## 3. Reward function

The reward logic is frozen in `environment/env.py`. The environment runs with:

- `max_steps = 1000`
- `order_time = 450`
- `COOK_TIME = 200`
- `BURN_TIME = 350`

### Reward table

| Event | Reward | Notes |
|:------|-------:|:------|
| Step penalty | -0.01 | Every step |
| Order expired | -2.0 | Per failed order |
| Invalid pot add | -0.01 | Wrong state, duplicate ingredient, or no matching order |
| Valid ingredient added to pot | +1.0 | Only for the first 3 soup cycles |
| Cooking completes | +0.5 | Only for the first 3 soup cycles |
| Pick up correct done soup | +2.0 | If it matches an active order and is within the first 3 soup cycles |
| Pick up burnt soup | -3.0 | - |
| Pot burns | -3.0 | - |
| Place done soup on handoff counter | +2.0 | Only for the first 3 rewarded handoffs |
| Serve correct soup | +20.0 + time bonus | `time_bonus = max(0, (deadline - step) * 0.01)` |
| Serve wrong / bad soup | -2.0 | Wrong order, undercooked, or burnt |
| Perfect episode bonus | +10.0 | All 3 orders served and 0 failed orders |

### Reward budget gating

The shaped rewards are intentionally capped so the agents cannot farm them by cooking extra soups that nobody needs.

The environment tracks:

- `soups_collected`
- `handoffs_rewarded`

Because there are exactly 3 orders in each episode, the intermediate shaping rewards only apply to the first 3 relevant soup cycles. That keeps the reward aligned with the real task instead of rewarding pointless extra work.

### Order generation

Each episode has 3 orders:

- Order 1 starts at step 0
- Order 2 starts at a random step in `[200, 300)`
- Order 3 starts at a random step in `[400, 499)`

Meals are sampled uniformly from:

- `onion-soup`
- `tomato-soup`
- `onion-tomato-soup`

Each order's deadline is:

- `deadline = start + 450`

---

## 4. Observation design

The base observation format is a flat `float32` vector with **74 features per frame**, stacked over the last **4 frames**, so the policy sees **296 values** per step.

The base 74-feature layout is:

| Group | Size | Meaning |
|:------|-----:|:--------|
| Directions | 4 | Facing direction features |
| Holdings | 12 | One-hot item state slots |
| Front tile | 20 | Front-of-agent tile and counter item summary |
| BFS distances | 24 | Normalised distance + reachability to 6 stations |
| Pot state | 4 | One-hot pot state |
| Pot contents + timer | 3 | Onion count, tomato count, timer |
| Order info | 3 | Most urgent order summary |
| Handoff summary | 4 | Summary of handoff-counter contents |
| **Total** | **74** | |

### 4.1 Fully centralised observation

In the centralised wrapper (`environment/gym_wrapper_rllib_centralised.py`), all 74 slots are live:

- both agents' directions
- both holdings
- both front-tile feature blocks
- both agents' BFS distance blocks
- pot state
- pot contents / timer
- next-order info
- handoff summary

This is a true joint-state controller.

### 4.2 Pure decentralised observation

In the decentralised wrapper (`environment/gym_wrapper_rllib_decentralised.py`), the same 74-slot layout is kept for fairness, but not all slots are live.

Live features:

- the agent's own direction
- the agent's own holding block
- the agent's own front-tile block
- the agent's own BFS distance block
- shared public task-state features:
  - pot state
  - pot contents
  - pot timer
  - next-order info

Masked features:

- teammate direction slots
- teammate holding slots
- teammate front-tile block
- teammate BFS block
- handoff summary block

Masked values use a sentinel `-1.0` so the shape stays aligned with the centralised baseline without leaking teammate-private information.

This means the no-comms baseline is not fully blind, but it is still properly decentralised at execution. It gets public task-state information, not the other agent's local action-ready state.

### Action spaces

- **Centralised PPO:** `MultiDiscrete([6, 6])`
- **Decentralised PPO:** `Discrete(6)` per agent

Shared action meanings:

| Index | Action |
|------:|:-------|
| 0 | Stay |
| 1 | Up |
| 2 | Down |
| 3 | Left |
| 4 | Right |
| 5 | Interact |

---

## 5. Levels

The three layouts are fixed in `environment/levels.py`.

### level_1 - The Bottleneck

```text
#####S#####
I         J
#         #
# A     B #
##### #####
#         #
#         #
##P##G##R##
```

A horizontal barrier with one gap. Both agents can access the full space, but the choke point causes a lot of blocking and timing issues.

### level_2 - The Partition

```text
###S#######
#    #    #
#    #    #
#    #    P
I    #    #
# B  #  A #
#    #    G
##R#####J##
```

A vertical wall splits the map. Agents cannot cross sides, so they have to coordinate through counter handoffs. This forces role separation.

### level_3 - The Obstacle Course

```text
#####S#####
I         #
#         #
#    P    #
#    #    #
# A  R  B #
#         J
#####G#####
```

This is more open, but the central pot / rack area creates path interference. Agents share the same space, but not through a single choke point like level 1.

---

## 6. Training setup (official RLlib runs)

The final benchmark uses RLlib PPO for all official runs.

### Shared PPO settings

- `num_env_runners = 0`
- `num_envs_per_env_runner = 8`
- `rollout_fragment_length = 4096`
- `batch_mode = "truncate_episodes"`
- `train_batch_size = 32768`
- `minibatch_size = 512`
- `num_epochs = 12` (or `num_sgd_iter = 12` on older API versions)
- `lr = 2e-4`
- `gamma = 0.995`
- `lambda_ = 0.97`
- `entropy_coeff = 0.005`
- `clip_param = 0.2`
- `vf_loss_coeff = 0.5`
- `grad_clip = 0.5`
- `use_kl_loss = False`
- `kl_coeff = 0.0`
- `vf_clip_param = 1000000.0`
- model hidden layers: `[512, 512, 256]`
- activation: `tanh`
- `vf_share_layers = False`
- `simple_optimizer = True`
- `_disable_preprocessor_api = True`
- fixed training seed: `12345`

### Centralised PPO

- wrapper: `environment/gym_wrapper_rllib_centralised.py`
- script: `agents/train_centralised_rllib.py`
- one joint policy controlling both agents
- total training budget: **10M env steps**

### Decentralised PPO (no comms)

- wrapper: `environment/gym_wrapper_rllib_decentralised.py`
- script: `agents/train_decentralised_rllib.py`
- two separate policies:
  - `agent_1_policy`
  - `agent_2_policy`
- shared team reward duplicated to both agents
- `count_steps_by = "env_steps"` so the training scale matches the centralised setup
- total training budget: **10M env steps**

Checkpoints are saved every **500k env steps**.

---

## 7. Evaluation and checkpoint selection

### Seed splits

These are the fixed evaluation splits used in practice:

| Set | Seeds | Episodes | Use |
|:----|:------|:---------|:----|
| Validation | `0-249` | 250 | 500k checkpoint sweep |
| Final test | `10000-12499` | 2500 | Locked final result |

The final 2500-episode test set is disjoint from the checkpoint-selection sweep.

### Deterministic evaluation

- Centralised PPO uses `explore=False` in RLlib (same intent as `deterministic=True` in SB3)
- Decentralised PPO also uses `explore=False`
- Frame stacking stays at 4 for both training and evaluation

### Checkpoint selection rule

The project originally used perfect-rate-first selection, but for the weaker decentralised runs that became too noisy because perfect episodes were very rare.

So the final rule is:

- For the **checkpoint sweep**, compare the 250-episode validation runs.
- When perfect rates are clearly separated, use that signal.
- When perfect rates are very sparse / noisy, choose the checkpoint with the **best overall validation profile**, with:
  1. strongest `score_mean`
  2. then `perfect_rate`
  3. then lower `failed_orders`
  4. then earlier checkpoint if still tied

This is what was used to select the final decentralised checkpoints in a way that reflects stable task completion rather than a handful of lucky perfect episodes.

### Output files

Evaluation scripts write:

- per-episode `.csv`
- aggregate `.summary.json`

under `eval_results/`.

---

## 8. Current locked benchmark results

### 8.1 Fully centralised PPO (official RLlib benchmark)

Chosen checkpoints and final 2500-episode deterministic results:

| Level | Checkpoint | Perfect rate | Score mean | Failed orders mean | Total reward mean |
|:------|-----------:|-------------:|-----------:|-------------------:|------------------:|
| level_1 | 9.0M | 0.9584 | 2.9388 | 0.0612 | 77.979248 |
| level_2 | 9.0M | 0.9888 | 2.9888 | 0.0112 | 85.348744 |
| level_3 | 7.0M | 0.9100 | 2.8660 | 0.1340 | 76.781020 |

Centralised ordering:

**level_3 < level_1 < level_2**

Level 2 is the strongest under the joint controller, which makes sense because the partition encourages stable role specialisation when the controller can see the full state.

### 8.2 Pure decentralised PPO (no comms)

Chosen checkpoints and final 2500-episode deterministic results:

| Level | Checkpoint | Perfect rate | Score mean | Failed orders mean | Total reward mean |
|:------|-----------:|-------------:|-----------:|-------------------:|------------------:|
| level_1 | 9.0M | 0.0040 | 1.2400 | 1.7600 | 18.895096 |
| level_2 | 8.0M | 0.0052 | 1.1596 | 1.8404 | 22.901084 |
| level_3 | 10.0M | 0.8752 | 2.8184 | 0.1816 | 74.867764 |

Decentralised ordering:

**level_2 < level_1 < level_3**

This is the main structural result so far. The no-comms independent learners struggle badly in the partitioned handoff layout, are also weak in the bottleneck layout, but perform strongly in the more open obstacle layout.

That contrast is the main reason the communication variant is worth testing next.

---

## 9. What is frozen from this point

The following are now frozen for the benchmark:

- reward function and reward magnitudes
- budget-gating logic for shaped rewards
- 3-order episode format and timing ranges
- level layouts (`level_1`, `level_2`, `level_3`)
- env constants (`max_steps = 1000`, `order_time = 450`, `COOK_TIME = 200`, `BURN_TIME = 350`)
- RLlib PPO config family used for official runs
- checkpoint cadence (every 500k env steps)
- validation and final-test seed splits
- centralised observation wrapper
- pure decentralised observation wrapper

The communication variant can add an explicit message mechanism, but it should keep the same task, reward function, training budget, and evaluation protocol.

That way any gain can be attributed to communication rather than to a different task definition.
