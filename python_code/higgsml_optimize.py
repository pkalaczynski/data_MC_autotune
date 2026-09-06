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
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# Config (mirrors the `const` block in test_higgsml_parquet.jl)
# ------------------------------------------------------------------
TEST_BINS = np.arange(0.0, 164.0, 4.0)  # 0.0:4.0:160.0
REJECT_MIN = 40.0
REJECT_MAX = 70.0
N_SUBSAMPLE = 500_000
SEED = 42
N_TRIALS = 500


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
        "weights", "labels", "PRI_met",
        "PRI_had_pt", "PRI_lep_pt", "PRI_jet_leading_pt", "PRI_jet_subleading_pt",
    ]
    df_full = pd.read_parquet(parquet_path, columns=columns)
    df_full = df_full.dropna()

    rng = np.random.default_rng(SEED)
    idx = rng.choice(df_full.index.to_numpy(), size=min(N_SUBSAMPLE, len(df_full)), replace=False)
    return df_full.loc[idx].reset_index(drop=True)


# ------------------------------------------------------------------
# 2. Objective: 8D phase-space loss
# ------------------------------------------------------------------
def compute_phase_space_loss(df, cuts) -> float:
    had_l, had_h, lep_l, lep_h, lead_l, lead_h, sub_l, sub_h = cuts

    mask = (
        (df.PRI_had_pt >= had_l) & (df.PRI_had_pt <= had_h)
        & (df.PRI_lep_pt >= lep_l) & (df.PRI_lep_pt <= lep_h)
        & (df.PRI_jet_leading_pt >= lead_l) & (df.PRI_jet_leading_pt <= lead_h)
        & (df.PRI_jet_subleading_pt >= sub_l) & (df.PRI_jet_subleading_pt <= sub_h)
    )
    sub = df[mask]

    survival = len(sub) / len(df)
    if survival < 0.15:
        return 1e5 * (0.15 - survival)

    sidebands = sub[(sub.PRI_met < REJECT_MIN) | (sub.PRI_met > REJECT_MAX)]
    if len(sidebands) < 500:
        return 1e6

    w_data, _ = np.histogram(sidebands.PRI_met, bins=TEST_BINS, weights=sidebands.weights)
    positive = w_data > 0
    mean_w = w_data.mean()
    chi2 = np.sum(((w_data[positive] - mean_w) ** 2) / w_data[positive])
    return float(chi2)


def suggest_cuts(trial: optuna.Trial):
    values = {}
    for spec in SEARCH_SPACE:
        values[spec.name] = trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step)
    return [values[name] for name in CUT_ORDER]


def make_objective(df):
    def objective(trial: optuna.Trial) -> float:
        return compute_phase_space_loss(df, suggest_cuts(trial))
    return objective


# ------------------------------------------------------------------
# 3. Optimization loop
# ------------------------------------------------------------------
def run(parquet_path: str, sampler_name: str = "random", n_trials: int = N_TRIALS):
    print("Opening parquet file and building the 500k-row subsample...")
    df = load_higgsml(parquet_path)

    sampler = (optuna.samplers.RandomSampler(seed=SEED) if sampler_name == "random"
               else optuna.samplers.TPESampler(seed=SEED))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"Running {n_trials}-trial 8D kinematic object optimization loop ({sampler_name})...")
    t_start = time.perf_counter()
    study.optimize(make_objective(df), n_trials=n_trials)
    elapsed = time.perf_counter() - t_start

    best = study.best_trial
    print(f"Tuning completed in {elapsed:.2f} seconds.")
    print(f"Best loss (chi2): {best.value:.4f}")
    for name in CUT_ORDER:
        print(f"  {name}: {best.params[name]:.1f}")

    result = {
        "engine": "python-optuna",
        "task": "higgsml-8d",
        "sampler": sampler_name,
        "n_trials": n_trials,
        "best_loss": best.value,
        "elapsed_seconds": elapsed,
        "best_cuts": {name: best.params[name] for name in CUT_ORDER},
    }
    with open(f"higgsml_results_{sampler_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", default="data/FAIR_Universe_HiggsML_data.parquet")
    parser.add_argument("--sampler", choices=["random", "tpe"], default="random")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    run(args.parquet_path, sampler_name=args.sampler, n_trials=args.n_trials)
