using DataFrames
using StatsBase
using Random
using DelimitedFiles

# ==========================================
# SHARED CONSTANTS (single source of truth for the 12D task)
# ==========================================
const ENERGY_BINS = 20.0:2.0:120.0
const SIGNAL_MIN = 82.0
const SIGNAL_MAX = 98.0

const N_DATA_EVENTS = 600_000
const N_MC_EVENTS = 800_000
const DATA_SEED = 123
const MC_SEED = 456

const DATA_DIR = joinpath(@__DIR__, "data")
const DATA_CSV = joinpath(DATA_DIR, "synthetic_data.csv")
const MC_CSV = joinpath(DATA_DIR, "synthetic_mc.csv")

# ==========================================
# SYNTHETIC DATA GENERATOR
# ==========================================
function generate_synthetic_data(n_events; seed=42, is_mc=false)
    Random.seed!(seed)

    energy = Float64[]
    track_chi2 = Float64[]
    isolation = Float64[]
    hits_count = Int[]
    timing_ns = Float64[]
    hadronic_fraction = Float64[]
    vertex_dist = Float64[]

    for _ in 1:n_events
        is_signal = rand() < 0.25
        e = is_signal ? randn()*4.0 + 90.0 : randexp()*35 + 20

        t_chi2 = is_signal ? randexp()*1.0 : randexp()*2.0
        iso = is_signal ? rand()*0.1 : rand()*0.3
        hits = is_signal ? rand(15:25) : rand(10:20)

        if is_mc && !is_signal && rand() < 0.35
            e = rand()*30.0 + 40.0
            t_chi2 = rand()*5.0 + 4.0
            iso = rand()*0.4 + 0.4
            hits = rand(5:9)
        end

        push!(energy, e)
        push!(track_chi2, t_chi2)
        push!(isolation, iso)
        push!(hits_count, hits)
        push!(timing_ns, randn() * 1.5)
        push!(hadronic_fraction, rand() * 0.3)
        push!(vertex_dist, randexp() * 1.5)
    end

    return DataFrame(
        energy=energy, track_chi2=track_chi2, isolation=isolation,
        hits_count=hits_count, timing_ns=timing_ns,
        hadronic_fraction=hadronic_fraction, vertex_dist=vertex_dist
    )
end

# ==========================================
# AGREEMENT LOSS (12D window cuts + peak protection + sideband chi2)
# ==========================================
function compute_agreement_loss(df_data, df_mc, cuts)
    chi2_low, chi2_high = cuts[1], cuts[2]
    iso_low, iso_high = cuts[3], cuts[4]
    hits_low, hits_high = cuts[5], cuts[6]
    time_low, time_high = cuts[7], cuts[8]
    had_low, had_high = cuts[9], cuts[10]
    vtx_low, vtx_high = cuts[11], cuts[12]

    d_sub = df_data[
        (chi2_low .<= df_data.track_chi2 .<= chi2_high) .& (iso_low .<= df_data.isolation .<= iso_high) .& (hits_low .<= df_data.hits_count .<= hits_high) .& (time_low .<= df_data.timing_ns .<= time_high) .& (had_low .<= df_data.hadronic_fraction .<= had_high) .& (vtx_low .<= df_data.vertex_dist .<= vtx_high), :]

    m_sub = df_mc[
        (chi2_low .<= df_mc.track_chi2 .<= chi2_high) .& (iso_low .<= df_mc.isolation .<= iso_high) .& (hits_low .<= df_mc.hits_count .<= hits_high) .& (time_low .<= df_mc.timing_ns .<= time_high) .& (had_low .<= df_mc.hadronic_fraction .<= had_high) .& (vtx_low .<= df_mc.vertex_dist .<= vtx_high), :]

    raw_peak_count = count(SIGNAL_MIN .<= df_data.energy .<= SIGNAL_MAX)
    cut_peak_count = count(SIGNAL_MIN .<= d_sub.energy .<= SIGNAL_MAX)
    peak_survival = cut_peak_count / raw_peak_count

    if peak_survival < 0.50
        return 1e6 * (0.50 - peak_survival)
    end

    d_sidebands = d_sub[(d_sub.energy .< SIGNAL_MIN) .| (d_sub.energy .> SIGNAL_MAX), :]
    m_sidebands = m_sub[(m_sub.energy .< SIGNAL_MIN) .| (m_sub.energy .> SIGNAL_MAX), :]

    if size(d_sidebands, 1) < 100 || size(m_sidebands, 1) < 100
        return 1e6
    end

    h_data = fit(Histogram, d_sidebands.energy, ENERGY_BINS)
    h_mc = fit(Histogram, m_sidebands.energy, ENERGY_BINS)

    w_data = h_data.weights
    w_mc = h_mc.weights .* (sum(h_data.weights) / sum(h_mc.weights))

    chi2 = 0.0
    for i in eachindex(w_data)
        observed = w_data[i]
        expected = w_mc[i]
        if observed + expected > 0
            chi2 += ((observed - expected)^2) / (observed + expected)
        end
    end

    return chi2
end

# ==========================================
# CSV PERSISTENCE
# Julia owns data generation. Python (and anything else downstream) only
# ever *reads* these files -- it never generates its own copy. Plain CSV
# rather than Parquet2 so both sides can read/write with only
# stdlib-adjacent tools (DelimitedFiles here, pandas.read_csv there).
# ==========================================
function write_dataframe_csv(path::AbstractString, df::DataFrame)
    mkpath(dirname(path))
    header = permutedims(string.(names(df)))
    data = Matrix(df)  # promotes Int (hits_count) + Float64 columns to a Float64 matrix
    open(path, "w") do io
        writedlm(io, header, ',')
        writedlm(io, data, ',')
    end
end

function read_dataframe_csv(path::AbstractString)
    data, header = readdlm(path, ',', Float64, header=true)
    df = DataFrame(data, vec(String.(header)))
    if "hits_count" in names(df)
        df.hits_count = Int.(round.(df.hits_count))
    end
    return df
end

"""
    load_or_generate(; force=false)

Single source of truth for the Data/MC pair used by every Julia script in
this repo (and, via the CSV files it writes to `data/`, by the Python side
too). Loads `data/synthetic_data.csv` + `data/synthetic_mc.csv` if they
already exist; otherwise generates them (fixed seeds -> fully
deterministic) and writes them out. Pass `force=true` to regenerate even
if the files are already there (see generate_data.jl).
"""
function load_or_generate(; force::Bool=false)
    if !force && isfile(DATA_CSV) && isfile(MC_CSV)
        println("Loading cached Data/MC pair from $(DATA_DIR)...")
        return read_dataframe_csv(DATA_CSV), read_dataframe_csv(MC_CSV)
    end

    println("Generating synthetic Data/MC pair (seeds $DATA_SEED / $MC_SEED)...")
    data_raw = generate_synthetic_data(N_DATA_EVENTS, seed=DATA_SEED, is_mc=false)
    mc_raw = generate_synthetic_data(N_MC_EVENTS, seed=MC_SEED, is_mc=true)

    write_dataframe_csv(DATA_CSV, data_raw)
    write_dataframe_csv(MC_CSV, mc_raw)
    println("Wrote $(nrow(data_raw)) rows to $(DATA_CSV)")
    println("Wrote $(nrow(mc_raw)) rows to $(MC_CSV)")

    return data_raw, mc_raw
end
