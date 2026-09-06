using Pkg
Pkg.activate(".")
using Hyperopt
using Statistics

include("common.jl")  # ENERGY_BINS, generate_synthetic_data, compute_agreement_loss, load_or_generate

# ==========================================
# Multi-seed repeat harness
# ==========================================
const N_TRIALS = 150
const N_REPEATS = 10
const SEEDS = 42:(42+N_REPEATS-1)          # sampler-only seeds; dataset stays fixed
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

data_raw, mc_raw = load_or_generate()

losses = Float64[]
times = Float64[]

println("Running $N_REPEATS repeats: sampler=RandomSampler, n_trials=$N_TRIALS, seeds=$(collect(SEEDS))")

for (i, seed) in enumerate(SEEDS)
    # Seed the global RNG right before the @hyperopt call so RandomSampler's
    # draw sequence is controlled and reproducible per repeat, same idea as
    # Optuna's per-run `sampler_seed` on the Python side.
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
