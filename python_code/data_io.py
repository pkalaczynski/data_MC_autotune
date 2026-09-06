"""
Loads the Data/MC pair that Julia generates and owns.

Data generation lives entirely on the Julia side now (see
julia/common.jl's `load_or_generate` and julia/generate_data.jl) --
Python never generates its own copy. This module's only job is reading
the CSV files Julia writes to data/, so that both sides are guaranteed
to be optimizing against the exact same dataset, not two independently
generated (even if statistically similar) ones.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("../data")
DATA_PATH = DATA_DIR / "synthetic_data.csv"
MC_PATH = DATA_DIR / "synthetic_mc.csv"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Data/MC pair written by Julia. Raises if it hasn't been generated yet."""
    missing = [p for p in (DATA_PATH, MC_PATH) if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing {missing_list}. Data generation is owned by Julia -- "
            f"run `julia generate_data.jl` (from the julia/ directory) first, "
            f"then re-run this."
        )

    df_data = pd.read_csv(DATA_PATH)
    df_mc = pd.read_csv(MC_PATH)

    # Julia writes hits_count as a Float64 column (see write_dataframe_csv in
    # common.jl, which promotes the whole DataFrame to a single numeric
    # matrix type); restore it to an integer dtype on this side for clarity.
    # This is cosmetic -- the loss function's comparisons work identically
    # either way -- but keeps hits_count reading as "13" rather than "13.0".
    for df in (df_data, df_mc):
        df["hits_count"] = df["hits_count"].round().astype(int)

    return df_data, df_mc
