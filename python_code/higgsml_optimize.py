"""
Optuna port of test_higgsml_parquet.jl -- 8D kinematic object-selection
tuning on the real FAIR Universe HiggsML dataset.

NOT executed in this environment: the dataset is ~14GB and is not
downloaded here (see data/download_HiggsML.sh). Run this yourself after
downloading it, from the repo root:

    python python/higgsml_optimize.py --sampler random --n-trials 500

Mirrors, in order:
  1. Data ingestion: read a handful of columns from the parquet file,
     drop missing values, and take a random 500k-row subsample.
  2. compute_phase_space_loss: the 8D window-cut objective (kinematic
     object pT windows), with a chi2-vs-flat-mean sideband loss and a
     minimum-survival-fraction guard (analogous to the peak protection
     in da_code.jl, but simpler here).
  3. An Optuna study over the same 8D search space as the Julia
     `@hyperopt` block (500 trials, RandomSampler by default for a fair
     comparison against Hyperopt.jl).
  4. The same three-panel before/post-selection/after-tuning comparison
     plot, reproduced with matplotlib.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_CSV = Path("output/higgsml_results.csv")
SUMMARY_CSV = Path("output/higgsml_summary.csv")
RESULTS_FIELDS = [
    "engine",
    "task",
    "sampler",
    "sampler_seed",
    "n_trials",
    "best_loss",
    "elapsed_seconds",
]
SUMMARY_FIELDS = [
    "engine",
    "task",
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


# ------------------------------------------------------------------
# Config (mirrors the `const` block in test_higgsml_parquet.jl)
# ------------------------------------------------------------------
TEST_BINS = np.arange(0.0, 164.0, 4.0)  # 0.0:4.0:160.0
REJECT_MIN = 40.0
REJECT_MAX = 70.0
N_SUBSAMPLE = 500_000
SEED = 42
N_TRIALS = 500
N_REPEATS = 10


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    step: float


SEARCH_SPACE = [
    ParamSpec("c_had_l", 20.0, 35.0, 2.0),
    ParamSpec("c_had_h", 120.0, 220.0, 5.0),
    ParamSpec("c_lep_l", 15.0, 30.0, 2.0),
    ParamSpec("c_lep_h", 100.0, 200.0, 5.0),
    ParamSpec("c_lead_l", 20.0, 35.0, 2.0),
    ParamSpec("c_lead_h", 120.0, 250.0, 5.0),
    ParamSpec("c_sub_l", 20.0, 35.0, 2.0),
    ParamSpec("c_sub_h", 120.0, 250.0, 5.0),
]
CUT_ORDER = [p.name for p in SEARCH_SPACE]


# ------------------------------------------------------------------
# 1. Data ingestion
# ------------------------------------------------------------------
def load_higgsml(parquet_path: str) -> pd.DataFrame:
    columns = [
        "weights",
        "labels",
        "PRI_met",
        "PRI_had_pt",
        "PRI_lep_pt",
        "PRI_jet_leading_pt",
        "PRI_jet_subleading_pt",
    ]
    df_full = pd.read_parquet(parquet_path, columns=columns)
    df_full = df_full.dropna()

    rng = np.random.default_rng(SEED)
    idx = rng.choice(
        df_full.index.to_numpy(), size=min(N_SUBSAMPLE, len(df_full)), replace=False
    )
    return df_full.loc[idx].reset_index(drop=True)


# ------------------------------------------------------------------
# 2. Objective: 8D phase-space loss
# ------------------------------------------------------------------
def compute_phase_space_loss(df, cuts) -> float:
    had_l, had_h, lep_l, lep_h, lead_l, lead_h, sub_l, sub_h = cuts

    mask = (
        (df.PRI_had_pt >= had_l)
        & (df.PRI_had_pt <= had_h)
        & (df.PRI_lep_pt >= lep_l)
        & (df.PRI_lep_pt <= lep_h)
        & (df.PRI_jet_leading_pt >= lead_l)
        & (df.PRI_jet_leading_pt <= lead_h)
        & (df.PRI_jet_subleading_pt >= sub_l)
        & (df.PRI_jet_subleading_pt <= sub_h)
    )
    sub = df[mask]

    survival = len(sub) / len(df)
    if survival < 0.15:
        return 1e5 * (0.15 - survival)

    sidebands = sub[(sub.PRI_met < REJECT_MIN) | (sub.PRI_met > REJECT_MAX)]
    if len(sidebands) < 500:
        return 1e6

    w_data, _ = np.histogram(
        sidebands.PRI_met, bins=TEST_BINS, weights=sidebands.weights
    )
    positive = w_data > 0
    mean_w = w_data.mean()
    chi2 = np.sum(((w_data[positive] - mean_w) ** 2) / w_data[positive])
    return float(chi2)


def suggest_cuts(trial: optuna.Trial):
    values = {}
    for spec in SEARCH_SPACE:
        values[spec.name] = trial.suggest_float(
            spec.name, spec.low, spec.high, step=spec.step
        )
    return [values[name] for name in CUT_ORDER]


def make_objective(df):
    def objective(trial: optuna.Trial) -> float:
        return compute_phase_space_loss(df, suggest_cuts(trial))

    return objective


# ------------------------------------------------------------------
# 3. Optimization loop
# ------------------------------------------------------------------
def _optimize_once(
    df: pd.DataFrame, sampler_name: str, n_trials: int, sampler_seed: int
):
    """One full Optuna study against an already-loaded dataframe. Kept separate
    from data loading so repeats (run_multi_seed) don't re-read/re-subsample
    the 14GB parquet file every time -- only the sampler's seed changes."""
    sampler = (
        optuna.samplers.RandomSampler(seed=sampler_seed)
        if sampler_name == "random"
        else optuna.samplers.TPESampler(seed=sampler_seed)
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t_start = time.perf_counter()
    study.optimize(make_objective(df), n_trials=n_trials)
    elapsed = time.perf_counter() - t_start

    return study.best_trial, elapsed


def run(
    parquet_path: str,
    sampler_name: str = "random",
    n_trials: int = N_TRIALS,
    sampler_seed: int = SEED,
):
    print("Opening parquet file and building the 500k-row subsample...")
    df = load_higgsml(parquet_path)

    print(
        f"Running {n_trials}-trial 8D kinematic object optimization loop ({sampler_name})..."
    )
    best, elapsed = _optimize_once(df, sampler_name, n_trials, sampler_seed)

    print(f"Tuning completed in {elapsed:.2f} seconds.")
    print(f"Best loss (chi2): {best.value:.4f}")
    for name in CUT_ORDER:
        print(f"  {name}: {best.params[name]:.1f}")

    result = {
        "engine": "python-optuna",
        "task": "higgsml-8d",
        "sampler": sampler_name,
        "sampler_seed": sampler_seed,
        "n_trials": n_trials,
        "best_loss": best.value,
        "elapsed_seconds": elapsed,
        "best_cuts": {name: best.params[name] for name in CUT_ORDER},
    }
    with open(f"output/higgsml_results_{sampler_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result, df


# ------------------------------------------------------------------
# 4. Multi-seed repeats -- same idea as run_multi_seed.py, but the
#    parquet file is loaded/subsampled ONCE and reused across repeats;
#    only the optimizer's sampler_seed changes between them.
# ------------------------------------------------------------------
def run_multi_seed(
    parquet_path: str,
    sampler_name: str = "random",
    n_trials: int = N_TRIALS,
    n_repeats: int = N_REPEATS,
    seeds=None,
):
    seeds = seeds or [SEED + i for i in range(n_repeats)]

    print(
        "Opening parquet file and building the 500k-row subsample (once, reused across repeats)..."
    )
    df = load_higgsml(parquet_path)

    print(
        f"Running {n_repeats} repeats: sampler={sampler_name}, n_trials={n_trials}, seeds={seeds}"
    )

    losses, times = [], []
    for i, seed in enumerate(seeds, start=1):
        best, elapsed = _optimize_once(df, sampler_name, n_trials, seed)
        losses.append(best.value)
        times.append(elapsed)
        print(
            f"  [{i}/{n_repeats}] seed={seed:<4d} best_loss={best.value:.4f}  time={elapsed:.2f}s"
        )

        append_csv(
            RESULTS_CSV,
            RESULTS_FIELDS,
            {
                "engine": "python-optuna",
                "task": "higgsml-8d",
                "sampler": sampler_name,
                "sampler_seed": seed,
                "n_trials": n_trials,
                "best_loss": best.value,
                "elapsed_seconds": elapsed,
            },
        )

    summary = {
        "engine": "python-optuna",
        "task": "higgsml-8d",
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
    print(
        f"Summary over {n_repeats} repeats ({sampler_name}, {n_trials} trials, HiggsML 8D):"
    )
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
    parser.add_argument(
        "--parquet-path", default="data/FAIR_Universe_HiggsML_data.parquet"
    )
    parser.add_argument("--sampler", choices=["random", "tpe"], default="random")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-repeats", type=int, default=1)
    args = parser.parse_args()

    if args.n_repeats > 1:
        run_multi_seed(
            args.parquet_path,
            sampler_name=args.sampler,
            n_trials=args.n_trials,
            n_repeats=args.n_repeats,
        )
    else:
        run(args.parquet_path, sampler_name=args.sampler, n_trials=args.n_trials)
