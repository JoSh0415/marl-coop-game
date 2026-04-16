import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def build_checkpoint_steps(start_step, end_step, step_size):
    return list(range(start_step, end_step + 1, step_size))


def build_command(python_bin, script_path, checkpoint_path, episodes, seed, level, stack_n,
                  deterministic, out_dir, max_steps_cap):
    cmd = [
        python_bin,
        script_path,
        "--checkpoint", str(checkpoint_path),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--levels", level,
        "--stack-n", str(stack_n),
        "--out-dir", str(out_dir),
    ]

    if deterministic:
        cmd.append("--deterministic")
    else:
        cmd.append("--no-deterministic")

    if max_steps_cap is not None:
        cmd.extend(["--max-steps-cap", str(max_steps_cap)])

    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack-n", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--out-dir", type=str, default="eval_results")
    parser.add_argument("--max-steps-cap", type=int, default=None)

    parser.add_argument("--start-step", type=int, default=5_000_000)
    parser.add_argument("--end-step", type=int, default=9_500_000)
    parser.add_argument("--step-size", type=int, default=500_000)

    parser.add_argument("--max-jobs", type=int, default=4)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--eval-script", type=str, default="scripts/eval_decentralised_comms_rllib.py")

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps = build_checkpoint_steps(args.start_step, args.end_step, args.step_size)

    jobs = []
    for step in steps:
        checkpoint_path = model_dir / "checkpoints" / f"checkpoint_{step}"
        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue

        cmd = build_command(
            python_bin=args.python_bin,
            script_path=args.eval_script,
            checkpoint_path=checkpoint_path,
            episodes=args.episodes,
            seed=args.seed,
            level=args.level,
            stack_n=args.stack_n,
            deterministic=args.deterministic,
            out_dir=out_dir,
            max_steps_cap=args.max_steps_cap,
        )

        log_file = logs_dir / f"eval_checkpoint_{step}.log"
        jobs.append((step, cmd, log_file))

    if not jobs:
        print("No valid checkpoints found.")
        sys.exit(1)

    env = os.environ.copy()
    env.pop("RAY_ADDRESS", None)

    running = []
    pending = jobs.copy()
    failures = []

    while pending or running:
        while pending and len(running) < args.max_jobs:
            step, cmd, log_file = pending.pop(0)
            f = open(log_file, "w")
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            running.append((step, proc, f, log_file))
            print(f"Started checkpoint_{step}")

        still_running = []
        for step, proc, f, log_file in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((step, proc, f, log_file))
                continue

            f.close()
            if ret == 0:
                print(f"Finished checkpoint_{step}")
            else:
                print(f"FAILED checkpoint_{step} (exit {ret}) -> {log_file}")
                failures.append((step, ret, log_file))

        running = still_running
        time.sleep(1)

    if failures:
        print("\nSome evaluations failed:")
        for step, ret, log_file in failures:
            print(f"  checkpoint_{step}: exit {ret} ({log_file})")
        sys.exit(1)

    print("\nAll evaluations completed successfully.")


if __name__ == "__main__":
    main()