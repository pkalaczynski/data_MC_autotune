using Pkg
Pkg.activate(".")
using DataFrames
using Statistics
using CairoMakie
using Random

# ==========================================
# CONFIG
# ==========================================
# Run this script from the julia/ directory (same place da_code_multiseed.jl
# writes julia_results.csv). Adjust these paths if your layout differs.
const PYTHON_RESULTS_CSV = "../python/python_results.csv"
const JULIA_RESULTS_CSV = "julia_results.csv"
const N_TRIALS_FILTER = 150   # only compare repeats run with this trial budget

# ==========================================
# 1. MINIMAL CSV LOADER (Base only -- no CSV.jl dependency)
# ==========================================
# readlines() chomps both "\n" and "\r\n" line endings, so this is safe
# against the CRLF rows written by Python's csv module.
function load_results_csv(path::AbstractString)
    lines = readlines(path)
    @assert !isempty(lines) "Empty file: $path"

    header = strip.(split(lines[1], ','))
    cols = Dict(h => String[] for h in header)

    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = strip.(split(line, ','))
        for (h, v) in zip(header, fields)
            push!(cols[h], v)
        end
    end

    df = DataFrame()
    for h in header
        df[!, Symbol(h)] = cols[h]
    end
    return df
end

function coerce_types!(df::DataFrame)
    df.sampler_seed = parse.(Int, df.sampler_seed)
    df.n_trials = parse.(Int, df.n_trials)
    df.best_loss = parse.(Float64, df.best_loss)
    df.elapsed_seconds = parse.(Float64, df.elapsed_seconds)
    return df
end

# ==========================================
# 2. LOAD BOTH ENGINES' PER-REPEAT RESULTS
# ==========================================
println("Loading per-repeat results...")

if !isfile(JULIA_RESULTS_CSV)
    error("$JULIA_RESULTS_CSV not found. Run da_code_multiseed.jl first (in this same directory).")
end
if !isfile(PYTHON_RESULTS_CSV)
    error("$PYTHON_RESULTS_CSV not found. Run python/run_multi_seed.py first, or fix the path above.")
end

df_jl = coerce_types!(load_results_csv(JULIA_RESULTS_CSV))
df_py = coerce_types!(load_results_csv(PYTHON_RESULTS_CSV))

df_jl = df_jl[df_jl.n_trials .== N_TRIALS_FILTER, :]
df_py = df_py[df_py.n_trials .== N_TRIALS_FILTER, :]

df_jl_random = df_jl[df_jl.sampler .== "random", :]
df_py_random = df_py[df_py.sampler .== "random", :]
df_py_tpe = df_py[df_py.sampler .== "tpe", :]

# ==========================================
# 3. BUILD GROUPS (skip any engine/sampler combo with no data)
# ==========================================
candidate_groups = [
    ("Julia\nRandomSampler", df_jl_random, :gray45),
    ("Python\nRandomSampler", df_py_random, :dodgerblue),
    ("Python\nTPE", df_py_tpe, :darkorange),
]
groups = [(label, df, color) for (label, df, color) in candidate_groups if nrow(df) > 0]
@assert length(groups) >= 2 "Need at least two non-empty groups to compare."

labels = [g[1] for g in groups]
n_groups = length(groups)

# ==========================================
# 4. CONSOLE SUMMARY (mean +/- std, min/max)
# ==========================================
println("\n" * "="^70)
println("ENGINE COMPARISON -- $(N_TRIALS_FILTER) trials per run")
println("="^70)
for (label, df, _) in groups
    lbl = replace(label, "\n" => " ")
    loss_mean, loss_std = mean(df.best_loss), std(df.best_loss)
    time_mean, time_std = mean(df.elapsed_seconds), std(df.elapsed_seconds)
    println(rpad(lbl, 24), " n=$(nrow(df))  ",
        "loss=$(round(loss_mean, digits=2)) +/- $(round(loss_std, digits=2))",
        "  (min $(round(minimum(df.best_loss), digits=2)), max $(round(maximum(df.best_loss), digits=2)))  ",
        "time=$(round(time_mean, digits=2))s +/- $(round(time_std, digits=2))s")
end
println("="^70)

if nrow(df_jl_random) > 0 && nrow(df_py_random) > 0
    jl_time = mean(df_jl_random.elapsed_seconds)
    py_time = mean(df_py_random.elapsed_seconds)
    jl_loss = mean(df_jl_random.best_loss)
    py_loss = mean(df_py_random.best_loss)
    speed_ratio = jl_time / py_time
    faster = speed_ratio > 1 ? "Python" : "Julia"
    println("With the same RandomSampler search: $(faster) was $(round(max(speed_ratio, 1/speed_ratio), digits=2))x faster on average.")
    println("Mean loss: Julia $(round(jl_loss, digits=2)) vs Python $(round(py_loss, digits=2)) ",
        "(lower is better; same objective and search space on both sides).")
end
println()

# ==========================================
# 5. PLOTTING (boxplots + jittered individual repeats)
# ==========================================
mkpath("plots")
Random.seed!(1)  # only used for the visual jitter below, not the comparison itself

fig = Figure(size=(1000, 480), font="DejaVu Sans")

ax_loss = Axis(fig[1, 1],
    title="Best Loss (χ²) across repeats",
    ylabel="χ² (lower is better)",
    xticks=(1:n_groups, labels),
    xgridvisible=false, ygridvisible=true,
    ygridstyle=:dash, ygridcolor=(:gray, 0.25)
)
ax_time = Axis(fig[1, 2],
    title="Wall Time across repeats",
    ylabel="Seconds",
    xticks=(1:n_groups, labels),
    xgridvisible=false, ygridvisible=true,
    ygridstyle=:dash, ygridcolor=(:gray, 0.25)
)

for (i, (label, df, color)) in enumerate(groups)
    boxplot!(ax_loss, fill(i, nrow(df)), df.best_loss, color=(color, 0.55), width=0.5, show_outliers=false)
    boxplot!(ax_time, fill(i, nrow(df)), df.elapsed_seconds, color=(color, 0.55), width=0.5, show_outliers=false)

    jitter = (rand(nrow(df)) .- 0.5) .* 0.25
    scatter!(ax_loss, fill(i, nrow(df)) .+ jitter, df.best_loss, color=(:black, 0.55), markersize=7)
    scatter!(ax_time, fill(i, nrow(df)) .+ jitter, df.elapsed_seconds, color=(:black, 0.55), markersize=7)
end

save("plots/engine_comparison.png", fig, px_per_unit=2)
save("plots/engine_comparison.pdf", fig)
println("Saved comparison figure to 'plots/engine_comparison.png' and '.pdf'")
