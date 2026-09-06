# Data/MC autotune -- Julia (Hyperopt.jl) + Python (Optuna)

Julia is the primary implementation and owns data generation and the
cross-engine comparison. Python is a subservient computation backend: it
reads the exact dataset Julia generated, runs the same optimization with
Optuna, and writes its results back out in a format Julia's comparison
script consumes. Python never generates its own data and doesn't produce
the comparison itself.

## Layout

```
pyproject.toml
julia/
  common.jl             constants, generate_synthetic_data, compute_agreement_loss,
                          CSV read/write, load_or_generate() -- shared by every Julia script
  generate_data.jl        explicit "(re)generate the dataset now" entry point
  da_code.jl               single-run Hyperopt.jl tuning + before/after CairoMakie plot
  da_code_multiseed.jl     repeats the tuning across seeds, logs to CSV
  compare_engines.jl       reads both engines' per-repeat CSVs, CairoMakie boxplots + console summary
src/python_code/
  __init__.py            exposes main() for the `python-code` console script
  cli.py                  argparse dispatcher: run / multi-seed / higgsml
  config.py               12D search space + repeat/seed settings (kept in sync with common.jl by hand)
  data_io.py               *read-only*: loads data/*.csv; raises if Julia hasn't generated them yet
  loss.py                  compute_agreement_loss, ported from common.jl
  optimize.py              single-run Optuna study
  run_multi_seed.py        repeats optimize.run() across seeds, logs to CSV
  plotting.py              Python's own before/after diagnostic plot (not the engine comparison)
  higgsml_optimize.py      Optuna port of the real-dataset 8D task (reads its own parquet file directly)
data/                     data/synthetic_data.csv + data/synthetic_mc.csv, written by Julia;
                          also where the real HiggsML parquet file goes if you download it
```

## Setup

```bash
uv sync              # Python deps, from pyproject.toml
```

Your existing Julia `Project.toml` needs no changes -- `common.jl` only
uses `DataFrames`, `StatsBase`, and stdlib (`Random`, `DelimitedFiles`),
all already listed; `da_code.jl`/`da_code_multiseed.jl`/`compare_engines.jl`
only add `Hyperopt`, `CairoMakie`, `PairPlots`, `Statistics` on top, also
already there.

## Workflow

**1. Generate the data (Julia only):**

```bash
cd julia
julia generate_data.jl
```

Writes `data/synthetic_data.csv` (600k rows) and `data/synthetic_mc.csv`
(800k rows). `da_code.jl` and `da_code_multiseed.jl` also call the same
`load_or_generate()` internally, so running either of them first works
too -- `generate_data.jl` is just the explicit, no-side-effects-besides-
the-data entry point, useful when you only want the files (e.g. so Python
has something to read) without also running a tuning loop.

**2. Run each engine's optimization:**

```bash
# Julia
julia run.jl                 # single run + its own before/after plot
julia run_multiseed.jl       # 10 repeats -> julia_results.csv / julia_summary.csv

# Python (reads the same data/*.csv Julia just wrote)
uv run python cli.py run --sampler random --n-trials 150         # single run + its own diagnostic plot
uv run python cli.py multi-seed --sampler random --n-repeats 10  # -> python_results.csv / python_summary.csv
```

If you skip step 1, Python's `run`/`multi-seed` will fail with a clear
`FileNotFoundError` telling you to run `generate_data.jl` first -- it will
not silently generate its own data.

**3. Compare (Julia only):**

```bash
cd julia
julia compare_engines.jl
```

Reads `julia_results.csv` and `../python_results.csv` (adjust the
`PYTHON_RESULTS_CSV` constant at the top if you keep it elsewhere), prints
the mean +/- std / min-max summary, and saves
`plots/engine_comparison.png`/`.pdf` -- boxplots of loss and wall time per
engine/sampler with every repeat overlaid as a jittered point.

## On splitting into separate repositories

The only things that cross the Julia/Python boundary are plain files:

| File | Written by | Read by |
|---|---|---|
| `data/synthetic_data.csv`, `data/synthetic_mc.csv` | Julia (`generate_data.jl` / `load_or_generate()`) | Python (`data_io.load_datasets()`) |
| `python_results.csv`, `python_summary.csv` | Python (`run_multi_seed.py`) | Julia (`compare_engines.jl`) |
| `julia_results.csv`, `julia_summary.csv` | Julia (`da_code_multiseed.jl`) | Julia (`compare_engines.jl`) |

All five have a fixed, simple CSV schema (see the header rows each script
writes). Because the coupling is entirely file-based rather than shared
code, splitting into two repositories is low-risk: the Python repo would
just need the three CSV files copied in (or fetched from wherever the
Julia repo publishes them) before `run`/`multi-seed`, and the Julia repo
would need `python_results.csv`/`python_summary.csv` copied in before
`compare_engines.jl`. Nothing here assumes a shared filesystem beyond that.

## Design notes

- **RNG**: Julia's `generate_synthetic_data` (Mersenne Twister, row-by-row)
  is the only generator now -- there is no separate Python implementation
  to drift out of sync with it.
- **CSV, not Parquet, for the synthetic dataset**: `common.jl` writes it
  via `DelimitedFiles.writedlm` (promoting the whole DataFrame to a
  `Matrix{Float64}`, so `hits_count` round-trips as e.g. `18.0`) and reads
  it back the same way; Python's `data_io.py` restores `hits_count` to an
  int dtype cosmetically on its side. Plain CSV was chosen over Parquet2
  so both sides only need stdlib-adjacent tools (`DelimitedFiles`,
  `pandas.read_csv`) rather than a shared binary format library.
- **What varies across repeats**: `da_code_multiseed.jl` / `multi-seed`
  hold the Data/MC datasets fixed and only reseed the *optimizer* between
  repeats, isolating "how variable is this optimizer on a fixed problem"
  from "how sensitive is the result to which dataset realization we got".
- **Sampler choice**: `--sampler random` uses `optuna.samplers.RandomSampler`
  to match Hyperopt.jl's `RandomSampler()`, isolating "same search,
  different engine" from "different search strategy". `--sampler tpe`
  shows what Optuna's actual default buys you on the same budget.