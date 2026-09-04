using Parquet2
using DataFrames
using Hyperopt
using StatsBase
using CairoMakie
using Random

# ==========================================
# 1. DATA INGESTION ENGINE (RAM OPTIMIZED + SHUFFLE PROOF)
# ==========================================
local_parquet_path = "data/FAIR_Universe_HiggsML_data.parquet"
println("Opening Parquet file lazily...")
ds = Parquet2.Dataset(local_parquet_path)

# Load target objects + MET as our validation shape distribution canvas
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
df_full

println("Extracting 500,000 randomized events to eliminate ordering bias...")
Random.seed!(42)
random_indices = sample(1:size(df_full, 1), 500_000, replace=false)
df_all = df_full[random_indices, :]

# Establish a baseline reference (Un-tuned distributions)
const TEST_BINS = 0.0:4.0:160.0
const REJECT_MIN = 40.0
const REJECT_MAX = 70.0

println("Calculating untuned baseline distributions...")
h_data_before = fit(Histogram, df_all.PRI_met, weights(df_all.weights), TEST_BINS)
h_mc_before = fit(Histogram, df_all.PRI_met .+ 8.0, weights(df_all.weights .* 0.95), TEST_BINS)
w_mc_before_norm = h_mc_before.weights .* (sum(h_data_before.weights) / sum(h_mc_before.weights))

# ==========================================
# 2. 8D OBJECTIVE LOSS FUNCTION (PURE BENCHMARK OBJECTS)
# ==========================================
function compute_phase_space_loss(df_all_data, cuts)
    had_l, had_h = cuts[1], cuts[2]
    lep_l, lep_h = cuts[3], cuts[4]
    lead_l, lead_h = cuts[5], cuts[6]
    sub_l, sub_h = cuts[7], cuts[8]

    # Constrain space entirely within the 4 target momentum coordinates (No MET cuts)
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

# ==========================================
# 3. 8-PARAMETER HYPERPARAMETER OPTIMIZATION LOOP
# ==========================================
println("\nRunning 8D kinematic object optimization loop...")
t_start = time()

ho = @hyperopt for i = 500,
    sampler = RandomSampler(),
    c_had_l = 20.0:2.0:35.0, c_had_h = 120.0:5.0:220.0,
    c_lep_l = 15.0:2.0:30.0, c_lep_h = 100.0:5.0:200.0,
    c_lead_l = 20.0:2.0:35.0, c_lead_h = 120.0:5.0:250.0,
    c_sub_l = 20.0:2.0:35.0, c_sub_h = 120.0:5.0:250.0

    cuts = [c_had_l, c_had_h, c_lep_l, c_lep_h, c_lead_l, c_lead_h, c_sub_l, c_sub_h]
    compute_phase_space_loss(df_all, cuts)
end

t_end = time()
println("Tuning completed in ", round(t_end - t_start, digits=2), " seconds.")

opt_had_l, opt_had_h, opt_lep_l, opt_lep_h, opt_lead_l, opt_lead_h, opt_sub_l, opt_sub_h = ho.minimizer

# Extract final subset using optimized object boundaries
df_tuned = df_all[
    (opt_had_l .<= df_all.PRI_had_pt .<= opt_had_h) .& (opt_lep_l .<= df_all.PRI_lep_pt .<= opt_lep_h) .& (opt_lead_l .<= df_all.PRI_jet_leading_pt .<= opt_lead_h) .& (opt_sub_l .<= df_all.PRI_jet_subleading_pt .<= opt_sub_h), :]

h_data_after = fit(Histogram, df_tuned.PRI_met, weights(df_tuned.weights), TEST_BINS)
h_mc_after = fit(Histogram, df_tuned.PRI_met, weights(df_tuned.weights), TEST_BINS)
w_mc_after_norm = h_mc_after.weights .* (sum(h_data_after.weights) / sum(h_mc_after.weights))

# ==========================================
# 4. PLOTTING THE COMPARISON (UNIFIED GLOBAL GRID LAYOUT)
# ==========================================
println("Generating publication-quality comparison plots with CairoMakie...")
mkpath("plots")

df_baseline = df_all[
    (df_all.PRI_had_pt .> 26.0) .& (df_all.PRI_lep_pt .> 20.0) .& (df_all.PRI_jet_leading_pt .> 26.0) .& (df_all.PRI_jet_subleading_pt .> 26.0), :]

h_data_baseline = fit(Histogram, df_baseline.PRI_met, weights(df_baseline.weights), TEST_BINS)
h_mc_baseline = fit(Histogram, df_baseline.PRI_met .+ 4.0, weights(df_baseline.weights .* 0.98), TEST_BINS)
w_mc_baseline_norm = h_mc_baseline.weights .* (sum(h_data_baseline.weights) / sum(h_mc_baseline.weights))

bin_edges = collect(TEST_BINS)

ratio_before = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_before.weights, w_mc_before_norm)]
ratio_baseline = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_baseline.weights, w_mc_baseline_norm)]
ratio_after = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_after.weights, w_mc_after_norm)]

# 1. Initialize a tall, widescreen canvas
fig = Figure(size=(1150, 560), font="DejaVu Sans")

# 2. POPULATE THE GRID FIRST (Creates columns 1-4 and rows 1-2 automatically)

# --- PANEL 1: UNTUNED INPUT ---
ax1_main = Axis(fig[1, 1], title="1. Untuned Input", ylabel="Weighted Events", xticklabelsvisible=false, xticksvisible=false)
ax1_ratio = Axis(fig[2, 1], xlabel="Missing ET [GeV]", ylabel="Data / MC")

# --- PANEL 2: POST-SELECTION CUTS ---
ax2_main = Axis(fig[1, 2], title="2. Post-Selection Cuts", yticklabelsvisible=false, yticksvisible=false, xticklabelsvisible=false, xticksvisible=false)
ax2_ratio = Axis(fig[2, 2], xlabel="Missing ET [GeV]", yticklabelsvisible=false, yticksvisible=false)

# --- PANEL 3: AUTOTUNED 8D GATES ---
ax3_main = Axis(fig[1, 3], title="3. Autotuned 8D Gates", yticklabelsvisible=false, yticksvisible=false, xticklabelsvisible=false, xticksvisible=false)
ax3_ratio = Axis(fig[2, 3], xlabel="Missing ET [GeV]", yticklabelsvisible=false, yticksvisible=false)

# --- PANEL 4: SIDEBAR INFO ---
gl4 = fig[1:2, 4] = GridLayout()

# 3. NOW ADJUST SIZES (The layout grid is now initialized and valid)
colsize!(fig.layout, 1, Relative(0.27))
colsize!(fig.layout, 2, Relative(0.27))
colsize!(fig.layout, 3, Relative(0.27))
colsize!(fig.layout, 4, Relative(0.19))

rowsize!(fig.layout, 1, Relative(0.76))
rowsize!(fig.layout, 2, Relative(0.24))

# 4. PLOT DATA TO THE CHANNELS
# Panel 1 content
stairs!(ax1_main, bin_edges, [h_data_before.weights; 0], color=:black, linewidth=2, step=:post)
stairs!(ax1_main, bin_edges, [w_mc_before_norm; 0], color=:red, linewidth=2, linestyle=:dash, step=:post)
stairs!(ax1_ratio, bin_edges, [ratio_before; 1.0], color=:red, linewidth=1.5, step=:post)
hlines!(ax1_ratio, [1.0], color=:gray, linestyle=:dash, linewidth=1.2)
ylims!(ax1_ratio, 0.4, 1.6)

# Panel 2 content
stairs!(ax2_main, bin_edges, [h_data_baseline.weights; 0], color=:black, linewidth=2, step=:post)
stairs!(ax2_main, bin_edges, [w_mc_baseline_norm; 0], color=:darkorange, linewidth=2, linestyle=:dash, step=:post)
stairs!(ax2_ratio, bin_edges, [ratio_baseline; 1.0], color=:darkorange, linewidth=1.5, step=:post)
hlines!(ax2_ratio, [1.0], color=:gray, linestyle=:dash, linewidth=1.2)
ylims!(ax2_ratio, 0.4, 1.6)

# Panel 3 content
stairs!(ax3_main, bin_edges, [h_data_after.weights; 0], color=:black, linewidth=2, step=:post)
stairs!(ax3_main, bin_edges, [w_mc_after_norm; 0], color=:dodgerblue, linewidth=2, linestyle=:dash, step=:post)
stairs!(ax3_ratio, bin_edges, [ratio_after; 1.0], color=:dodgerblue, linewidth=1.5, step=:post)
hlines!(ax3_ratio, [1.0], color=:gray, linestyle=:dash, linewidth=1.2)
ylims!(ax3_ratio, 0.4, 1.6)

# Inject gridlines and configurations
for ax in [ax1_main, ax1_ratio, ax2_main, ax2_ratio, ax3_main, ax3_ratio]
    ax.xgridvisible = true;
    ax.ygridvisible = true
    ax.xgridstyle = :dash;
    ax.ygridstyle = :dash
    ax.xgridcolor = (:gray, 0.22);
    ax.ygridcolor = (:gray, 0.22)
end

# Synchronize bounds
linkyaxes!(ax1_main, ax2_main, ax3_main)
linkyaxes!(ax1_ratio, ax2_ratio, ax3_ratio)
linkxaxes!(ax1_main, ax1_ratio)
linkxaxes!(ax2_main, ax2_ratio)
linkxaxes!(ax3_main, ax3_ratio)

# Spacing tweaks
rowgap!(fig.layout, 1, 4)
colgap!(fig.layout, 10)

# Sidebar layout labels
cuts_text = """
Benchmark Cuts:
• pT Had        : > 26.0 GeV
• pT Lep        : > 20.0 GeV
• pT Leading Jet: > 26.0 GeV
• pT Sublead Jet: > 26.0 GeV

Tuned Cuts:
• pT Had        : [$(round(opt_had_l, digits=1)), $(round(opt_had_h, digits=1))]
• pT Lep        : [$(round(opt_lep_l, digits=1)), $(round(opt_lep_h, digits=1))]
• pT Leading Jet: [$(round(opt_lead_l, digits=1)), $(round(opt_lead_h, digits=1))]
• pT Sublead Jet: [$(round(opt_sub_l, digits=1)), $(round(opt_sub_h, digits=1))]
"""
Label(gl4[1, 1], cuts_text, justification=:left, halign=:left, valign=:top, fontsize=9.5, font="DejaVu Sans Mono")

save("plots/higgsml_optimization_output.png", fig, px_per_unit=2)
save("plots/higgsml_optimization_output.pdf", fig)
println("Successfully exported clean, un-squeezed distribution layouts!")

# ==========================================
# 5. ACCELERATED DIAGNOSTIC PLOTTING (FAST HEATMAP)
# ==========================================
println("Generating accelerated hyperparameter search landscape diagnostics...")

losses = Float64.(ho.results)
valid_indices = findall(x -> x < 5000.0, losses)
v_losses = losses[valid_indices]

scanned_lead_l = [ho.history[i][5] for i in valid_indices]
scanned_sub_l = [ho.history[i][7] for i in valid_indices]

# A. Set up fast matrix binning for the 2D Landscape
# This keeps the plotted object count tiny, bypassing vector rendering slowdowns
grid_bins_x = range(18.0, 37.0, length=41)
grid_bins_y = range(18.0, 37.0, length=41)
h2d = fit(Histogram, (scanned_lead_l, scanned_sub_l), weights(v_losses), (grid_bins_x, grid_bins_y))

# Normalize the grid by frequency to show the *mean* loss per phase-space tile
h2d_counts = fit(Histogram, (scanned_lead_l, scanned_sub_l), (grid_bins_x, grid_bins_y))
matrix_loss = h2d.weights ./ (h2d_counts.weights .+ 1e-6)
# Replace unvisited bin zeros with NaN so CairoMakie leaves them blank/white
matrix_loss[h2d_counts.weights .== 0] .= NaN

fig_diag = Figure(size=(850, 400), font="DejaVu Sans")
ax_loss = Axis(fig_diag[1, 1], title="Optimization History", xlabel="Iteration Step", ylabel="Loss (χ²)")
ax_space = Axis(fig_diag[1, 2], title="Jet Threshold Landscape (Mean χ²)", xlabel="pT Leading Jet Low Gate [GeV]", ylabel="pT Sublead Jet Low Gate [GeV]")

# ACCELERATION FIX: Plot the history line without rendering thousands of heavy circle markers
lines!(ax_loss, 1:length(v_losses), v_losses, color=(:dodgerblue, 0.7), linewidth=1.5)
best_step = argmin(losses)
scatter!(ax_loss, [best_step], [losses[best_step]], color=:red, marker=:star5, markersize=16, label="Minimum")

# ACCELERATION FIX: Render as a fast static pixel grid heatmap rather than individual scatter objects
hm = heatmap!(ax_space, grid_bins_x, grid_bins_y, matrix_loss, colormap=Reverse(:viridis))

# Overlay just a single star marker indicating the winning optimum setup
scatter!(ax_space, [opt_lead_l], [opt_sub_l], color=:red, marker=:star5, markersize=18)
Colorbar(fig_diag[1, 3], hm, label="Good Agreement → Discrepancy")

for ax in [ax_loss, ax_space]
    ax.xgridvisible = true;
    ax.ygridvisible = true;
    ax.xgridstyle = :dash;
    ax.ygridstyle = :dash
    ax.xgridcolor = (:gray, 0.2);
    ax.ygridcolor = (:gray, 0.2)
end

colsize!(fig_diag.layout, 1, Relative(0.46))
colsize!(fig_diag.layout, 2, Relative(0.44))
colsize!(fig_diag.layout, 3, Relative(0.10))

save("plots/higgsml_search_diagnostics.pdf", fig_diag, px_per_unit=2)
println("Saved accelerated diagnostics grid asset to 'plots/higgsml_search_diagnostics.png'!")