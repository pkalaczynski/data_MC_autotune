"""
Shared configuration for the Data/MC quality-window autotuning task.

Mirrors the global constants and the 12D search space defined in the
original `da_code.jl` script, so that the Julia (Hyperopt.jl) and Python
(Optuna) runs are optimizing literally the same objective over the same
bounds and the same number of trials.
"""

from dataclasses import dataclass
import numpy as np

# ------------------------------------------------------------------
# Physics-ish constants (identical to da_code.jl)
# ------------------------------------------------------------------
ENERGY_BINS = np.arange(20.0, 122.0, 2.0)  # 20.0:2.0:120.0 -> 51 edges / 50 bins
SIGNAL_MIN = 82.0
SIGNAL_MAX = 98.0

# ------------------------------------------------------------------
# Synthetic dataset sizes (identical to da_code.jl)
# ------------------------------------------------------------------
N_DATA_EVENTS = 600_000
N_MC_EVENTS = 800_000
DATA_SEED = 123
MC_SEED = 456

# ------------------------------------------------------------------
# Optimization budget (identical to da_code.jl: `for i = 150`)
# ------------------------------------------------------------------
N_TRIALS = 150
RANDOM_SAMPLER_SEED = 42  # only affects the Python run's reproducibility

# ------------------------------------------------------------------
# Multi-seed repeat settings (for mean/std over independent optimizer runs)
# ------------------------------------------------------------------
N_REPEATS = 10
# Seeds are just RANDOM_SAMPLER_SEED, RANDOM_SAMPLER_SEED+1, ... -- deterministic
# and reproducible, but any list of distinct ints works fine.
REPEAT_SEEDS = [RANDOM_SAMPLER_SEED + i for i in range(N_REPEATS)]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable cut boundary, matching one `c_xxx = lo:step:hi` line."""
    name: str
    low: float
    high: float
    step: float
    is_int: bool = False


# The 12 dimensions, in the same order as `cuts = [...]` in da_code.jl
SEARCH_SPACE = [
    ParamSpec("c_chi2_l", 0.0, 1.0, 0.5),
    ParamSpec("c_chi2_h", 3.5, 8.0, 0.5),
    ParamSpec("c_iso_l", 0.0, 0.1, 0.02),
    ParamSpec("c_iso_h", 0.35, 0.8, 0.05),
    ParamSpec("c_hits_l", 5, 13, 2, is_int=True),
    ParamSpec("c_hits_h", 22, 35, 2, is_int=True),
    ParamSpec("c_time_l", -5.0, -2.0, 0.5),
    ParamSpec("c_time_h", 2.0, 5.0, 0.5),
    ParamSpec("c_had_l", 0.0, 0.06, 0.02),
    ParamSpec("c_had_h", 0.25, 0.6, 0.05),
    ParamSpec("c_vtx_l", 0.0, 0.8, 0.2),
    ParamSpec("c_vtx_h", 2.5, 5.0, 0.5),
]

CUT_ORDER = [p.name for p in SEARCH_SPACE]
