"""
Optuna-based 12D quality-window autotuning.

This is the direct Python/Optuna counterpart of the `@hyperopt` block in
da_code.jl: same search space (config.SEARCH_SPACE), same trial budget
(config.N_TRIALS = 150), same objective (loss.compute_agreement_loss).

Usage:
    python optimize.py                # RandomSampler, apples-to-apples vs Hyperopt.jl
    python optimize.py --sampler tpe  # Optuna's default smarter sampler, for contrast

Writes:
    results_random.json / results_tpe.json  -- best cuts, best loss, wall time
"""

import argparse
import json
import time

import optuna
import numpy as np

from config import (
    SEARCH_SPACE,
    CUT_ORDER,
    N_TRIALS,
    RANDOM_SAMPLER_SEED,
    N_DATA_EVENTS,
    N_MC_EVENTS,
    DATA_SEED,
    MC_SEED,
)
from synthetic_data import generate_synthetic_data
from loss import compute_agreement_loss

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)  # keep stdout clean like the Julia script


def suggest_cuts(trial: optuna.Trial):
    values = {}
    for spec in SEARCH_SPACE:
        if spec.is_int:
            values[spec.name] = trial.suggest_int(
                spec.name, int(spec.low), int(spec.high), step=int(spec.step)
            )
        else:
            values[spec.name] = trial.suggest_float(
                spec.name, spec.low, spec.high, step=spec.step
            )
    return [values[name] for name in CUT_ORDER]


def make_objective(df_data, df_mc):
    def objective(trial: optuna.Trial) -> float:
        cuts = suggest_cuts(trial)
        return compute_agreement_loss(df_data, df_mc, cuts)

    return objective


def run(
    sampler_name: str = "random",
    n_trials: int = N_TRIALS,
    verbose: bool = True,
    sampler_seed: int = RANDOM_SAMPLER_SEED,
):
    """
    sampler_seed controls the *optimizer's* randomness (which points in the
    search space get tried, and in what order) -- NOT the dataset, which
    stays fixed at DATA_SEED/MC_SEED so every repeat is optimizing the same
    fixed Data/MC problem. This is the knob run_multi_seed.py varies to get
    a mean/std over repeated optimizer runs.
    """
    if verbose:
        print("Generating synthetic Data/MC datasets...")
    df_data = generate_synthetic_data(N_DATA_EVENTS, seed=DATA_SEED, is_mc=False)
    df_mc = generate_synthetic_data(N_MC_EVENTS, seed=MC_SEED, is_mc=True)

    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    study = optuna.create_study(direction="minimize", sampler=sampler)

    if verbose:
        print(
            f"Running {n_trials}-trial 12D tuning loop via Optuna ({sampler_name} sampler)..."
        )

    t_start = time.perf_counter()
    study.optimize(
        make_objective(df_data, df_mc), n_trials=n_trials, show_progress_bar=False
    )
    elapsed = time.perf_counter() - t_start

    best = study.best_trial
    cuts_named = {name: best.params[name] for name in CUT_ORDER}

    if verbose:
        print(f"Tuning completed in {elapsed:.2f} seconds.")
        print("\n" + "=" * 40)
        print("OPTIMIZED 12D QUALITY WINDOW CUT RESULTS (Optuna):")
        print("=" * 40)
        print(
            f"  track_chi2        : [{cuts_named['c_chi2_l']:.2f}, {cuts_named['c_chi2_h']:.2f}]"
        )
        print(
            f"  isolation         : [{cuts_named['c_iso_l']:.2f}, {cuts_named['c_iso_h']:.2f}]"
        )
        print(
            f"  hits_count        : [{int(cuts_named['c_hits_l'])}, {int(cuts_named['c_hits_h'])}]"
        )
        print(
            f"  timing_ns         : [{cuts_named['c_time_l']:.2f}, {cuts_named['c_time_h']:.2f}]"
        )
        print(
            f"  hadronic_fraction : [{cuts_named['c_had_l']:.2f}, {cuts_named['c_had_h']:.2f}]"
        )
        print(
            f"  vertex_dist       : [{cuts_named['c_vtx_l']:.2f}, {cuts_named['c_vtx_h']:.2f}]"
        )
        print("=" * 40)
        print(f"Best loss (chi2)  : {best.value:.4f}")

    result = {
        "engine": "python-optuna",
        "sampler": sampler_name,
        "sampler_seed": sampler_seed,
        "n_trials": n_trials,
        "n_data_events": N_DATA_EVENTS,
        "n_mc_events": N_MC_EVENTS,
        "best_loss": best.value,
        "elapsed_seconds": elapsed,
        "best_cuts": cuts_named,
    }

    with open(f"output/results_{sampler_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result, df_data, df_mc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler",
        choices=["random", "tpe"],
        default="random",
        help="'random' matches Hyperopt.jl's RandomSampler for a fair comparison; "
        "'tpe' uses Optuna's default smarter sampler.",
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    run(sampler_name=args.sampler, n_trials=args.n_trials)
