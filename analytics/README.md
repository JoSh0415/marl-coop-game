# Analysis and figures

This folder stores generated analytics outputs used in the dissertation (figures and summary tables).

The scripts that generate these outputs are in `scripts/`, not in `analytics/`.

The analysis layer is for interpreting the frozen benchmark in `BENCHMARK.md`, not for changing the benchmark itself.

## Analysis inputs

The current plotting/table scripts read from two main sources:

- `eval_sweeps/`
  - per-checkpoint summary JSON files:
    - `eval_sweeps/ppo_<model>_level_<n>/eval_checkpoint_<step>_level_<n>.summary.json`
  - these come from checkpoint sweeps (usually 500k-step intervals)

- `eval_results/`
  - best/final checkpoint outputs per algorithm + level
  - both filename styles below are used by current scripts:
    - `eval_best_checkpoint_level_<n>.summary.json`
    - `eval_ppo_<model>_level_<n>_best.summary.json`
    - `eval_ppo_<model>_level_<n>_best.csv`

Notes:

- `logs/*/progress.csv` exists for training diagnostics, but the current dissertation figure scripts in `scripts/` do not read `progress.csv` directly.
- The analysis scripts use hard-coded relative paths, so they should be run from the repository root.

## Analysis scripts

### `scripts/plot_score_mean.py`

- Purpose: learning-curve figure for mean orders served (`score_mean`) over checkpoint steps.
- Reads: checkpoint-sweep summary JSONs in `eval_sweeps/`.
- Writes: `analytics/score_mean_over_checkpoints.png`.

### `scripts/plot_perfect_rate.py`

- Purpose: learning-curve figure for perfect-episode rate over checkpoint steps.
- Reads: checkpoint-sweep summary JSONs in `eval_sweeps/`.
- Writes: `analytics/perfect_rate_over_checkpoints.png`.

### `scripts/plot_behaviour_bars.py`

- Purpose: behavioural diagnostics (`collision_attempts`, `wrong_pot_adds`, `both_idle_steps`) shown as rates per 100 steps.
- Reads: `eval_results/ppo_<model>_level_<n>/eval_ppo_<model>_level_<n>_best.summary.json`.
- Writes: `analytics/behaviour_rates_barchart.png`.

### `scripts/plot_role_split_by_level.py`

- Purpose: role-split / division-of-labour lollipop plot using Agent 1 share of key actions.
- Reads: `eval_results/ppo_<model>_level_<n>/eval_best_checkpoint_level_<n>.summary.json`.
- Writes: `analytics/role_split_lollipop.png`.

### `scripts/generate_sparse_table.py`

- Purpose: sparse error incidence table (% of episodes with at least one event).
- Reads: `eval_results/ppo_<model>_level_<n>/eval_ppo_<model>_level_<n>_best.csv`.
- Writes:
  - `analytics/sparse_error_incidence_table.csv`
  - `analytics/sparse_error_incidence_table.md`

## Generated outputs

Current generated files in this folder:

- `score_mean_over_checkpoints.png`: learning progression by checkpoint (score mean).
- `perfect_rate_over_checkpoints.png`: learning progression by checkpoint (perfect rate).
- `behaviour_rates_barchart.png`: behavioural error/idle diagnostics per level and algorithm.
- `role_split_lollipop.png`: division-of-labour view (Agent 1 share vs Agent 2 share).
- `sparse_error_incidence_table.csv`: raw numeric sparse-error incidence values.
- `sparse_error_incidence_table.md`: markdown table version for report writing.

## Typical workflow

1. Train RLlib controllers (`agents/train_*.py`) and save checkpoints in `models/`.
2. Run checkpoint sweeps into `eval_sweeps/` (e.g. with `scripts/run_eval_sweep_parallel.py`) for validation curves and selection.
3. Run final deterministic best-checkpoint evaluations into `eval_results/`.
4. Run the analytics scripts:
   - `python scripts/plot_score_mean.py`
   - `python scripts/plot_perfect_rate.py`
   - `python scripts/plot_behaviour_bars.py`
   - `python scripts/plot_role_split_by_level.py`
   - `python scripts/generate_sparse_table.py`
5. Use files in `analytics/` directly in the dissertation figures/tables section.
