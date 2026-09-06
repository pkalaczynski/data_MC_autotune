# data_MC_autotune -- Python/Optuna port

A clean, modular Python translation of the Julia/Hyperopt.jl scripts, for
comparing Optuna against Hyperopt.jl on the exact same 12D Data/MC
agreement-tuning task (`da_code.jl`), plus an equivalent 8D port of the
real-dataset script (`test_higgsml_parquet.jl`).

## Layout

```
python/
  config.py             constants + 12D search space + repeat/seed settings (mirrors da_code.jl)
  synthetic_data.py      vectorized Data/MC generator (mirrors generate_synthetic_data)
  loss.py                 compute_agreement_loss (peak protection + sideband chi2)
  optimize.py             single-run Optuna study (RandomSampler by default, or TPE)
  run_all.py              single-run entry point: generate -> tune -> plot
  run_multi_seed.py       repeats optimize.py across seeds, logs every run + a summary to CSV
  plotting.py             before/after main+ratio comparison figure (matplotlib)
  compare_engines.py      side-by-side mean+/-std table from python_summary.csv + julia_summary.csv
julia/
  da_code_multiseed.jl    Julia-side counterpart of run_multi_seed.py (same repeats, same CSV shape)
requirements.txt
```

Nothing here downloads the 14GB HiggsML dataset -- `higgsml_optimize.py`
expects it already unpacked at `data/FAIR_Universe_HiggsML_data.parquet`
(see `data/download_HiggsML.sh` in the Julia repo) and is meant to be run
on your machine.

## Quick start (single run, synthetic 12D task)

```bash
pip install -r requirements.txt
cd python
python run_all.py --sampler random --n-trials 150   # apples-to-apples vs Hyperopt.jl
python run_all.py --sampler tpe    --n-trials 150   # Optuna's smarter default, for contrast
```

Each run writes `results_<sampler>.json` (best loss, best cuts, wall time)
and `energy_distribution_comparison.png`/`.pdf`.

## Getting a mean +/- std instead of one run

A single optimization run can land on a lucky or unlucky draw. `run_multi_seed.py`
repeats the whole tuning loop several times -- the Data/MC datasets stay
fixed (same `DATA_SEED`/`MC_SEED`), only the _optimizer's_ seed changes each
repeat -- and logs everything to CSV:

```bash
cd python
python run_multi_seed.py --sampler random --n-repeats 10 --n-trials 150
python run_multi_seed.py --sampler tpe    --n-repeats 10 --n-trials 150
```

This appends to:

- `python_results.csv` -- one row per repeat (`engine,sampler,sampler_seed,n_trials,best_loss,elapsed_seconds`)
- `python_summary.csv` -- one row per (sampler, n_trials) config, with `loss_mean/std/min/max` and `time_mean/std/min/max`

## Comparing against the Julia run

`julia/da_code_multiseed.jl` is the Julia-side counterpart of `run_multi_seed.py`:
same objective, same search space, same repeat structure, writing
`julia_results.csv` / `julia_summary.csv` in the same shape. Run it with:

```bash
cd julia
julia da_code_multiseed.jl
```

Then, still from `julia/`, generate the comparison plot in Julia (CairoMakie):

```bash
julia compare_engines.jl
```

This reads `julia_results.csv` (produced above) and `../python/python_results.csv`
(produced by `run_multi_seed.py`), prints the same mean +/- std / min-max summary
to the console, and saves `plots/engine_comparison.png` / `.pdf`: side-by-side
boxplots of best loss and wall time for Julia/RandomSampler, Python/RandomSampler,
and Python/TPE (whichever of those you've actually generated data for -- it skips
any group with no CSV rows), with each repeat overlaid as a jittered point so you
can see the spread, not just the mean.

There's also a text-only Python version (`python/compare_engines.py`) if you'd
rather not leave Julia for this step, but it just prints a table, no plot.

## Design notes / where this differs from the Julia version

- **RNG**: `synthetic_data.py` uses NumPy's PCG64 generator (`default_rng`),
  not Julia's Mersenne Twister, and draws are vectorized rather than
  drawn one row at a time in the exact order of `generate_synthetic_data`.
  Every distribution, fraction, and conditional branch is identical, so
  the two datasets are statistically equivalent and pose the same
  optimization difficulty, but they are not bit-for-bit the same rows.
  This is why the "fair" comparison uses `RandomSampler` on both sides
  and compares the achieved loss/time, not identical minimizer values.
- **Search space discretization**: Optuna's `suggest_float(..., step=...)`
  and `suggest_int(..., step=...)` reproduce Julia's `lo:step:hi` ranges.
  Where `step` doesn't evenly divide `hi - lo` (e.g. `22:2:35` in Julia
  actually only reaches 34), Optuna's warning about the same truncation
  is expected and matches Julia's own behavior -- not a bug.
- **What varies across repeats**: `run_multi_seed.py` / `da_code_multiseed.jl`
  hold the Data/MC datasets fixed and only reseed the _optimizer_ between
  repeats. That isolates "how variable is this optimizer on a fixed
  problem" -- if you instead want to know how sensitive the result is to
  which synthetic dataset realization you happened to generate, reseed
  `DATA_SEED`/`MC_SEED` instead (a different, and slower to compute,
  question).
- **Sampler choice**: `--sampler random` uses `optuna.samplers.RandomSampler`
  to match Hyperopt.jl's `RandomSampler()` used in both `.jl` scripts, so
  the comparison isolates "same search, different engine/language" rather
  than "different search strategies". `--sampler tpe` is included to show
  what Optuna's actual default (a smarter, non-random sampler) buys you
  on the same budget.
