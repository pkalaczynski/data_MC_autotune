using Parquet2
using DataFrames
using StatsBase
using Statistics
using Random

# ==========================================
# SHARED CONSTANTS (mirrors the `const` block in test_higgsml_parquet.jl)
# ==========================================
const TEST_BINS = 0.0:4.0:160.0
const REJECT_MIN = 10.0
const REJECT_MAX = 40.0
const N_SUBSAMPLE = 100_000
const SUBSAMPLE_SEED = 42  # fixed -- the row subsample stays the same across every repeat;
# only the optimizer's own seed varies (see higgsml_multiseed.jl)

# ==========================================
# DATA INGESTION (identical to test_higgsml_parquet.jl's ingestion section)
# ==========================================
function load_higgsml(parquet_path::AbstractString)
    println("Opening Parquet file lazily...")
    ds = Parquet2.Dataset(parquet_path)

    target_columns = [
        :weights, :labels, :PRI_met,
        :PRI_had_pt, :PRI_lep_pt, :PRI_jet_leading_pt, :PRI_jet_subleading_pt
    ]

    println("Streaming selected columns into RAM...")
    sub_table = Parquet2.select(ds, target_columns...)

    println("Materializing table into DataFrames.jl...")
    df_full = DataFrame(sub_table)

    println("Getting rid off missing values...")
    dropmissing!(df_full)
    disallowmissing!(df_full)

    println("Extracting a randomized $(N_SUBSAMPLE)-event subsample (seed=$SUBSAMPLE_SEED, fixed across repeats)...")
    Random.seed!(SUBSAMPLE_SEED)
    n = min(N_SUBSAMPLE, size(df_full, 1))
    random_indices = sample(1:size(df_full, 1), n, replace=false)
    return df_full[random_indices, :]
end

# ==========================================
# 8D PHASE-SPACE LOSS (identical to test_higgsml_parquet.jl's compute_phase_space_loss)
# ==========================================
function compute_phase_space_loss(df_all_data, cuts)
    had_l, had_h = cuts[1], cuts[2]
    lep_l, lep_h = cuts[3], cuts[4]
    lead_l, lead_h = cuts[5], cuts[6]
    sub_l, sub_h = cuts[7], cuts[8]

    sub = df_all_data[
        (had_l .<= df_all_data.PRI_had_pt .<= had_h) .& (lep_l .<= df_all_data.PRI_lep_pt .<= lep_h) .& (lead_l .<= df_all_data.PRI_jet_leading_pt .<= lead_h) .& (sub_l .<= df_all_data.PRI_jet_subleading_pt .<= sub_h), :]

    survival = size(sub, 1) / size(df_all_data, 1)
    if survival < 0.15
        return 1e5 * (0.15 - survival)
    end

    sidebands = sub[(sub.PRI_met .< REJECT_MIN) .| (sub.PRI_met .> REJECT_MAX), :]
    if size(sidebands, 1) < 500
        return 1e6
    end

    h_data = fit(Histogram, sidebands.PRI_met, weights(sidebands.weights), TEST_BINS)
    w_data = h_data.weights
    chi2 = 0.0
    for i in eachindex(w_data)
        obs = w_data[i]
        if obs > 0
            chi2 += ((obs - mean(w_data))^2) / obs
        end
    end
    return chi2
end
