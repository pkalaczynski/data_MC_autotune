"""
Command-line entry point for the `python-code` console script declared in
pyproject.toml (`python-code = "python_code:main"`).

Python is a subservient computation backend here: it reads the Data/MC
pair that Julia generates (data/synthetic_data.csv, data/synthetic_mc.csv;
see julia/generate_data.jl), runs the same Optuna-based optimization, and
writes its results back out as CSV/JSON -- the same files Julia's own
compare_engines.jl reads to build the cross-engine comparison. Python does
not generate data and does not own the comparison; both of those are
Julia's job.

    uv run python-code run        [--sampler random|tpe] [--n-trials N]
    uv run python-code multi-seed [--sampler random|tpe] [--n-trials N] [--n-repeats N]
    uv run python-code higgsml    [--parquet-path PATH] [--sampler random|tpe] [--n-trials N]
"""

import argparse

from .config import N_REPEATS, N_TRIALS
from .higgsml_optimize import N_TRIALS as HIGGS_N_TRIALS
from .higgsml_optimize import run as run_higgsml
from .optimize import run as run_single
from .plotting import plot_before_after
from .run_multi_seed import run_multi_seed


def main() -> None:
    parser = argparse.ArgumentParser(prog="python-code", description="Optuna backend for the Data/MC autotune task")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Single optimization run against Julia's data + Python's own diagnostic plot")
    p_run.add_argument("--sampler", choices=["random", "tpe"], default="random")
    p_run.add_argument("--n-trials", type=int, default=N_TRIALS)

    p_multi = sub.add_parser("multi-seed", help="Repeat the run across several sampler seeds, log to CSV")
    p_multi.add_argument("--sampler", choices=["random", "tpe"], default="random")
    p_multi.add_argument("--n-trials", type=int, default=N_TRIALS)
    p_multi.add_argument("--n-repeats", type=int, default=N_REPEATS)

    p_higgs = sub.add_parser("higgsml", help="Run the 8D Optuna tuning on the real HiggsML parquet file")
    p_higgs.add_argument("--parquet-path", default="data/FAIR_Universe_HiggsML_data.parquet")
    p_higgs.add_argument("--sampler", choices=["random", "tpe"], default="random")
    p_higgs.add_argument("--n-trials", type=int, default=HIGGS_N_TRIALS)

    args = parser.parse_args()

    if args.command == "run":
        result, df_data, df_mc = run_single(sampler_name=args.sampler, n_trials=args.n_trials)
        out_path = plot_before_after(df_data, df_mc, result["best_cuts"])
        print(f"\nSaved comparison figure to '{out_path}'")
    elif args.command == "multi-seed":
        run_multi_seed(args.sampler, args.n_trials, args.n_repeats)
    elif args.command == "higgsml":
        run_higgsml(args.parquet_path, sampler_name=args.sampler, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
