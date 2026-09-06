"""
Repeat the Optuna tuning run several times with different sampler seeds,
to get a mean +/- std of the achieved loss and wall time instead of a
single (possibly lucky/unlucky) number.

The dataset (Data + MC) is held fixed across repeats -- only the
optimizer's own randomness (which points in the search space it tries)
varies, via `sampler_seed`. This isolates "how variable is this
optimizer's outcome on a fixed problem", which is what you want when
comparing engines/samplers.

Usage:
    python run_multi_seed.py --sampler random --n-repeats 10 --n-trials 150
    python run_multi_seed.py --sampler tpe    --n-repeats 10 --n-trials 150

Writes (append-only, so random/tpe/future runs accumulate in one place):
    python_results.csv   -- one row per repeat
    python_summary.csv    -- one row per (sampler, n_trials) config, with mean/std
"""

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, stdev

from config import N_TRIALS, N_REPEATS, REPEAT_SEEDS
from optimize import run

RESULTS_CSV = Path("output/python_results.csv")
SUMMARY_CSV = Path("output/python_summary.csv")

RESULTS_FIELDS = [
    "engine",
    "sampler",
    "sampler_seed",
    "n_trials",
    "best_loss",
    "elapsed_seconds",
]
SUMMARY_FIELDS = [
    "engine",
    "sampler",
    "n_trials",
    "n_repeats",
    "loss_mean",
    "loss_std",
    "loss_min",
    "loss_max",
    "time_mean",
    "time_std",
    "time_min",
    "time_max",
]


def append_csv(path: Path, fieldnames, row: dict):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_multi_seed(sampler_name: str, n_trials: int, n_repeats: int, seeds=None):
    seeds = seeds or [REPEAT_SEEDS[0] + i for i in range(n_repeats)]
    losses, times = [], []

    print(
        f"Running {n_repeats} repeats: sampler={sampler_name}, n_trials={n_trials}, "
        f"seeds={seeds}"
    )

    for i, seed in enumerate(seeds, start=1):
        t0 = time.perf_counter()
        result, _, _ = run(
            sampler_name=sampler_name,
            n_trials=n_trials,
            verbose=False,
            sampler_seed=seed,
        )
        wall = result["elapsed_seconds"]
        losses.append(result["best_loss"])
        times.append(wall)

        print(
            f"  [{i}/{n_repeats}] seed={seed:<4d} best_loss={result['best_loss']:.4f}  "
            f"time={wall:.2f}s"
        )

        append_csv(
            RESULTS_CSV,
            RESULTS_FIELDS,
            {
                "engine": "python-optuna",
                "sampler": sampler_name,
                "sampler_seed": seed,
                "n_trials": n_trials,
                "best_loss": result["best_loss"],
                "elapsed_seconds": wall,
            },
        )

    summary = {
        "engine": "python-optuna",
        "sampler": sampler_name,
        "n_trials": n_trials,
        "n_repeats": n_repeats,
        "loss_mean": mean(losses),
        "loss_std": stdev(losses) if len(losses) > 1 else 0.0,
        "loss_min": min(losses),
        "loss_max": max(losses),
        "time_mean": mean(times),
        "time_std": stdev(times) if len(times) > 1 else 0.0,
        "time_min": min(times),
        "time_max": max(times),
    }
    append_csv(SUMMARY_CSV, SUMMARY_FIELDS, summary)

    print("\n" + "=" * 60)
    print(f"Summary over {n_repeats} repeats ({sampler_name}, {n_trials} trials):")
    print(
        f"  Best loss (chi2) : {summary['loss_mean']:.4f} +/- {summary['loss_std']:.4f}  "
        f"[min {summary['loss_min']:.4f}, max {summary['loss_max']:.4f}]"
    )
    print(
        f"  Wall time (s)    : {summary['time_mean']:.2f} +/- {summary['time_std']:.2f}  "
        f"[min {summary['time_min']:.2f}, max {summary['time_max']:.2f}]"
    )
    print("=" * 60)
    print(f"Appended per-run rows to {RESULTS_CSV}, summary row to {SUMMARY_CSV}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", choices=["random", "tpe"], default="random")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of sampler seeds; overrides --n-repeats",
    )
    args = parser.parse_args()

    if args.seeds:
        run_multi_seed(args.sampler, args.n_trials, len(args.seeds), seeds=args.seeds)
    else:
        run_multi_seed(args.sampler, args.n_trials, args.n_repeats)
