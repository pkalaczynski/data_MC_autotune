using Pkg
Pkg.activate(".")
# Explicit entry point for (re)generating the synthetic Data/MC pair.
#
# `include("common.jl")`'s `load_or_generate()` already auto-generates on
# first use from da_code.jl / da_code_multiseed.jl, so running this script
# is only necessary when you want to force a fresh dataset (e.g. after
# changing DATA_SEED/MC_SEED or the generator itself), or when you want to
# produce data/*.csv before ever running an optimization (e.g. purely so
# the Python side has something to read).
include("common.jl")

load_or_generate(force=true)
println("Done. Python reads these same files via python_code.data_io.load_datasets().")
