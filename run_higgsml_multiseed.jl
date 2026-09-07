using Pkg
Pkg.activate(".")
using Hyperopt
using Statistics

include("higgsml_common.jl")  # TEST_BINS, load_higgsml, compute_phase_space_loss

# ==========================================
# Multi-seed repeat harness (8D HiggsML task)
# ==========================================
const PARQUET_PATH = "data/FAIR_Universe_HiggsML_data.parquet"
const N_TRIALS = 500
const N_REPEATS = 10
const SEEDS = 42:(42+N_REPEATS-1)          # sampler-only seeds; the 500k-row subsample stays fixed
const RESULTS_CSV = "output/higgsml_julia_results.csv"
const SUMMARY_CSV = "output/higgsml_julia_summary.csv"

function append_csv(path, header, row)
    is_new = !isfile(path)
    open(path, "a") do io
        if is_new
            println(io, header)
        end
        println(io, row)
    end
end

# Load + subsample ONCE and reuse across every repeat -- same reasoning as
# run_multiseed.jl's load_or_generate() call outside the loop, and the same
# fix applied on the Python side (see higgsml_optimize.py's run_multi_seed):
# without this, each repeat would re-open and re-subsample the 14GB parquet
# file, which dwarfs the actual optimization time.
df_all = load_higgsml(PARQUET_PATH)

losses = Float64[]
times = Float64[]

println("Running $N_REPEATS repeats: sampler=RandomSampler, n_trials=$N_TRIALS, seeds=$(collect(SEEDS))")

for (i, seed) in enumerate(SEEDS)
    # Seed the global RNG right before the @hyperopt call so RandomSampler's
    # draw sequence is controlled and reproducible per repeat -- same idea as
    # Optuna's per-run `sampler_seed` on the Python side. This happens AFTER
    # load_higgsml's own Random.seed!(SUBSAMPLE_SEED) call, so it only
    # affects which cuts get tried, not which rows were subsampled.
    Random.seed!(seed)

    t_start = time()
    ho = @hyperopt for i = N_TRIALS,
        sampler = RandomSampler(),
        c_had_l = 20.0:2.0:35.0, c_had_h = 120.0:5.0:220.0,
        c_lep_l = 15.0:2.0:30.0, c_lep_h = 100.0:5.0:200.0,
        c_lead_l = 20.0:2.0:35.0, c_lead_h = 120.0:5.0:250.0,
        c_sub_l = 20.0:2.0:35.0, c_sub_h = 120.0:5.0:250.0

        cuts = [c_had_l, c_had_h, c_lep_l, c_lep_h, c_lead_l, c_lead_h, c_sub_l, c_sub_h]
        compute_phase_space_loss(df_all, cuts)
    end
    elapsed = time() - t_start

    push!(losses, ho.minimum)
    push!(times, elapsed)

    println("  [$i/$N_REPEATS] seed=$seed  best_loss=$(round(ho.minimum, digits=4))  time=$(round(elapsed, digits=2))s")

    append_csv(RESULTS_CSV, "engine,task,sampler,sampler_seed,n_trials,best_loss,elapsed_seconds",
        "julia-hyperopt,higgsml-8d,random,$seed,$N_TRIALS,$(ho.minimum),$elapsed")
end

loss_mean, loss_std = mean(losses), std(losses)
time_mean, time_std = mean(times), std(times)

println("\n" * "="^60)
println("Summary over $N_REPEATS repeats (RandomSampler, $N_TRIALS trials, HiggsML 8D):")
println("  Best loss (chi2) : $(round(loss_mean, digits=4)) +/- $(round(loss_std, digits=4))  ",
    "[min $(round(minimum(losses), digits=4)), max $(round(maximum(losses), digits=4))]")
println("  Wall time (s)    : $(round(time_mean, digits=2)) +/- $(round(time_std, digits=2))  ",
    "[min $(round(minimum(times), digits=2)), max $(round(maximum(times), digits=2))]")
println("="^60)

append_csv(SUMMARY_CSV,
    "engine,task,sampler,n_trials,n_repeats,loss_mean,loss_std,loss_min,loss_max,time_mean,time_std,time_min,time_max",
    "julia-hyperopt,higgsml-8d,random,$N_TRIALS,$N_REPEATS,$loss_mean,$loss_std,$(minimum(losses)),$(maximum(losses)),$time_mean,$time_std,$(minimum(times)),$(maximum(times))")

println("Appended per-run rows to $RESULTS_CSV, summary row to $SUMMARY_CSV")
