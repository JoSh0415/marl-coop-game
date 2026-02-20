# Benchmark

This is the frozen benchmark for the project. Everything here (rewards, levels, eval seeds) stays the same across all three algorithms so comparisons are fair.

---

## 1. Scope

Three algorithm variants, all PPO-based, compared under the same environment rules:

1. **Decentralised PPO (no comms)** - baseline. Each agent has its own policy and only sees its own local observation.
2. **Decentralised PPO + comms** - main comparison. Same as (1) but agents exchange a small learned message each step.
3. **Centralised shared-policy PPO** - upper bound / oracle coordinator. A single policy controls both agents using a joint observation of the full state. Already trained and evaluated.

The centralised PPO is the ceiling — it has access to everything, so it should outperform the decentralised setups. The interesting question is how close the decentralised variants (especially with comms) can get.

Env settings, reward values, level layouts, and eval protocol are all locked.

---

## 2. Algorithms (what differs)

All three algorithms share the **same** env rules, rewards, levels, `max_steps`, `order_time`, and seed protocol. The only differences are:

- **Observation structure:** centralised PPO gets a joint 74-feature vector covering both agents' full state. Decentralised variants will each get a local observation derived from the same base state, but without the other agent's privileged info.
- **Communication channel (comms variant only):** adds a small message vector that each agent sends/receives per step. The message is fed into the observation — it doesn't give access to any extra env state, just what the other agent chose to share.
- **Policy:** centralised uses one shared policy for both agents. Decentralised variants use separate per-agent policies (may or may not share weights — TBD).

The decentralised obs spec and comms protocol are not implemented yet (planned / upcoming). The benchmark rules are frozen regardless — when they're built, they'll use the same reward function, order generation, and evaluation seeds.

---

## 3. Rewards

All from `environment/env.py`. Env runs with `max_steps=1000`, `order_time=450`.

| Event | Reward | Notes |
|:------|-------:|:------|
| Step penalty | −0.01 | Every step |
| Order expired | −2.0 | Per failed order |
| Add ingredient to pot | +1.0 | Only first 3 soup cycles (`soups_collected < 3`) |
| Cooking finishes (timer ≥ 200) | +0.5 | Only first 3 soup cycles |
| Soup burns (timer ≥ 350) | −3.0 | - |
| Pick up correct done soup | +2.0 | Only if `soups_collected ≤ 3` |
| Pick up burnt soup | −3.0 | - |
| Serve correct soup | +20.0 + time bonus | `time_bonus = max(0, (deadline − step) × 0.01)` |
| Serve wrong / bad soup | −2.0 | Wrong order, burnt, or undercooked |
| Place done soup on handoff counter | +2.0 | Only first 3 (`handoffs_rewarded < 3`) |
| Invalid pot add | −0.01 | Wrong state, duplicate ingredient, or no matching order |
| Perfect bonus (terminal) | +10.0 | All 3 orders served, 0 failures |

### Budget gating

The env tracks `soups_collected` (goes up every time a soup is taken from the pot, done or burnt). Since there are exactly 3 orders per episode, intermediate shaping rewards (+1.0 ingredient, +0.5 cook done, +2.0 pickup, +2.0 handoff) are only given for the first 3 soup cycles. This stops the agents from farming reward by cooking extra soups that nobody ordered.

### Order generation

3 orders per episode (`_random_orders()`):
- Order 1: random meal, starts at step 0
- Order 2: random meal, starts at random step in [200, 300)
- Order 3: random meal, starts at random step in [400, 499)

Meals are picked uniformly from: onion-soup, tomato-soup, onion-tomato-soup.  
Each order's deadline = `start + 450`.

---

## 4. Observation (centralised PPO)

From `environment/gym_wrapper.py`. Flat `float32` vector, 74 features, clipped to [−1, 1].

| Group | Size | What it is |
|:------|-----:|:-----------|
| Directions | 4 | Both agents' facing direction (normalised) |
| Holdings | 6 + 6 | One-hot per agent: nothing / onion / tomato / bowl / done soup / burnt soup |
| Front tile | 10 + 10 | What's in front of each agent - tile type flags, item on counter, handoff flag |
| BFS distances | 24 | Normalised distance + reachable flag to each of 6 stations, per agent |
| Pot state | 4 | One-hot: idle / cooking / done / burnt |
| Pot contents + timer | 3 | Onion count, tomato count, timer normalised by 350 |
| Order info | 3 | Time left (normalised), target onions, target tomatoes for most urgent order |
| Handoff summary | 4 | Counts of items on handoff counters (normalised) |
| **Total** | **74** | |

Frame stacked × 4 (`VecFrameStack`), so the model sees 296 features per step.

Action space: `MultiDiscrete([6, 6])` - stay, up, down, left, right, interact - per agent.

---

## 5. Levels

From `environment/levels.py`.

### level_1 - The Bottleneck

```
#####S#####
I         J
#         #
# A     B #
##### #####
#         #
#         #
##P##G##R##
```

Horizontal counter with a single gap. Both agents have to pass through it, so there's a lot of collision risk in that one tile.

### level_2 - The Partition

```
###S#######
#    #    #
#    #    #
#    #    P
I    #    #
# B  #  A #
#    #    G
##R#####J##
```

Vertical wall splits the map in half. Agents can't cross to the other side, so they have to pass items over the counter. Forces actual role splitting.

### level_3 - The Obstacle Course

```
#####S#####
I         #
#         #
#    P    #
#    #    #
# A  R  B #
#         J
#####G#####
```

Open room, but the pot and rack are in the middle acting as obstacles. Agents share the whole space so collisions happen more dynamically.

### Difficulty

I expected: level_3 (easiest) < level_1 < level_2 (hardest).

From the centralised PPO results: level_2 is hardest (91.8% perfect rate), level_1 and level_3 are close (93.6% vs 94.2%). So the ordering roughly holds but all three are above 90% — the centralised policy is strong enough that the differences aren't huge. Full numbers in `eval_results/`.

---

## 6. Training setup (centralised PPO)

From `agents/train_shared.py`:

- PPO via `stable_baselines3`, `MlpPolicy`
- `net_arch = [512, 512, 256]`
- 10M timesteps, 8 parallel envs
- `n_steps=4096`, `batch_size=512`
- `ent_coef=0.01`, `gamma=0.99`, `gae_lambda=0.95`, `lr=0.0003`
- Frame stacking: 4
- `TRAIN_SEED = 12345` (passed to `make_vec_env` and `PPO`)

Checkpoints saved every 500k steps → files like `ppo_model_500000_steps.zip` in `models/ppo_shared_<level>/`.  
Best checkpoint after selection: `ppo_shared_<level>_best.zip`.

Episode seeds are generated inside `GymCoopEnv.reset()` using the wrapper's `_np_random` RNG (seeded by the vec env), so episodes vary but are deterministic for a given seed.

---

## 7. Evaluation & checkpoint selection

### Seed splits

| Set | Seeds | Episodes | Used for |
|:----|:------|:---------|:---------|
| Validation | 0–499 | 500 | Picking the best checkpoint |
| Test | 10,000–12,499 | 2,500 | Final numbers (run once) |

These are disjoint - test seeds are never seen during checkpoint selection.

### How the best checkpoint is picked

Run all checkpoints on the validation set (500 eps, deterministic). Pick by:

1. Highest `perfect_rate`
2. Tie-break: highest `score_mean`
3. Tie-break: lowest `failed_orders` mean
4. Tie-break: earliest checkpoint

Selected model gets saved as `ppo_shared_<level>_best.zip`.

### Eval settings

- `deterministic=True`
- Frame stack: 4 (same as training)
- `max_steps=1000`

### Output files

Per level, saved under `eval_results/ppo_shared_<level>/`:
- `.csv` - per-episode results (score, failures, reward, error counts)
- `.summary.json` - aggregated stats

Committed summaries:
- `eval_results/ppo_shared_level_1/eval_ppo_shared_level_1_best_level_1.summary.json`
- `eval_results/ppo_shared_level_2/eval_ppo_shared_level_2_best_level_2.summary.json`
- `eval_results/ppo_shared_level_3/eval_ppo_shared_level_3_best_level_3.summary.json`

---

## 8. What's frozen

- Reward function + budget gating thresholds
- Order generation (3 orders, same timing distributions, `order_time=450`)
- Level layouts (level_1, level_2, level_3)
- Eval seed splits + checkpoint selection protocol
- Env constants (`COOK_TIME=200`, `BURN_TIME=350`, `max_steps=1000`)
- Centralised observation spec (74 features, 4-frame stack) — locked for the centralised baseline

The decentralised observation spec will be defined separately when those agents are built, but it must be derived from the same base env state — no extra privileged info, no changes to rewards or env rules. The comms variant only adds a message input/output on top of the local obs, not access to hidden state.

In short: env rules and eval protocol are shared across all three algorithms. Observation structure is the thing that's allowed to differ (by design).
