using DataFrames
using StatsBase
using Hyperopt
using Random
using Statistics

# ==========================================
# Same constants, generator, and objective as da_code.jl
# (duplicated here rather than `include`-ing da_code.jl, so this script can
#  run standalone without triggering da_code.jl's plotting section)
# ==========================================
const ENERGY_BINS = 20.0:2.0:120.0
const SIGNAL_MIN = 82.0
const SIGNAL_MAX = 98.0

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
# Multi-seed repeat harness
# ==========================================
const N_TRIALS = 150
const N_REPEATS = 10
const SEEDS = 42:(42 + N_REPEATS - 1)          # sampler-only seeds; dataset stays fixed
const RESULTS_CSV = "julia_results.csv"
const SUMMARY_CSV = "julia_summary.csv"

function append_csv(path, header, row)
    is_new = !isfile(path)
    open(path, "a") do io
        if is_new
            println(io, header)
        end
        println(io, row)
    end
end

println("Generating fixed synthetic Data/MC datasets (shared across all repeats)...")
data_raw = generate_synthetic_data(600000, seed=123, is_mc=false)
mc_raw = generate_synthetic_data(800000, seed=456, is_mc=true)

losses = Float64[]
times = Float64[]

println("Running $N_REPEATS repeats: sampler=RandomSampler, n_trials=$N_TRIALS, seeds=$(collect(SEEDS))")

for (i, seed) in enumerate(SEEDS)
    # Seed the global RNG right before the @hyperopt call so RandomSampler's
    # draw sequence is controlled and reproducible per repeat, same idea as
    # Optuna's per-run `sampler_seed` in run_multi_seed.py.
    Random.seed!(seed)

    t_start = time()
    ho = @hyperopt for i = N_TRIALS,
        sampler = RandomSampler(),
        c_chi2_l = 0.0:0.5:1.0, c_chi2_h = 3.5:0.5:8.0,
        c_iso_l = 0.0:0.02:0.1, c_iso_h = 0.35:0.05:0.8,
        c_hits_l = 5:2:13, c_hits_h = 22:2:35,
        c_time_l = -5.0:0.5:-2.0, c_time_h = 2.0:0.5:5.0,
        c_had_l = 0.0:0.02:0.06, c_had_h = 0.25:0.05:0.6,
        c_vtx_l = 0.0:0.2:0.8, c_vtx_h = 2.5:0.5:5.0

        cuts = [c_chi2_l, c_chi2_h, c_iso_l, c_iso_h, c_hits_l, c_hits_h,
            c_time_l, c_time_h, c_had_l, c_had_h, c_vtx_l, c_vtx_h]

        compute_agreement_loss(data_raw, mc_raw, cuts)
    end
    elapsed = time() - t_start

    push!(losses, ho.minimum)
    push!(times, elapsed)

    println("  [$i/$N_REPEATS] seed=$seed  best_loss=$(round(ho.minimum, digits=4))  time=$(round(elapsed, digits=2))s")

    append_csv(RESULTS_CSV, "engine,sampler,sampler_seed,n_trials,best_loss,elapsed_seconds",
               "julia-hyperopt,random,$seed,$N_TRIALS,$(ho.minimum),$elapsed")
end

loss_mean, loss_std = mean(losses), std(losses)
time_mean, time_std = mean(times), std(times)

println("\n" * "="^60)
println("Summary over $N_REPEATS repeats (RandomSampler, $N_TRIALS trials):")
println("  Best loss (chi2) : $(round(loss_mean, digits=4)) +/- $(round(loss_std, digits=4))  ",
        "[min $(round(minimum(losses), digits=4)), max $(round(maximum(losses), digits=4))]")
println("  Wall time (s)    : $(round(time_mean, digits=2)) +/- $(round(time_std, digits=2))  ",
        "[min $(round(minimum(times), digits=2)), max $(round(maximum(times), digits=2))]")
println("="^60)

append_csv(SUMMARY_CSV,
    "engine,sampler,n_trials,n_repeats,loss_mean,loss_std,loss_min,loss_max,time_mean,time_std,time_min,time_max",
    "julia-hyperopt,random,$N_TRIALS,$N_REPEATS,$loss_mean,$loss_std,$(minimum(losses)),$(maximum(losses)),$time_mean,$time_std,$(minimum(times)),$(maximum(times))")

println("Appended per-run rows to $RESULTS_CSV, summary row to $SUMMARY_CSV")
