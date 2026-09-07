using Pkg
Pkg.activate(".")
using DataFrames
using StatsBase
using Hyperopt
using CairoMakie
using Random
using PairPlots

include("common.jl")  # ENERGY_BINS, SIGNAL_MIN/MAX, generate_synthetic_data,
# compute_agreement_loss, load_or_generate
include("hyperopt_diagnostics.jl")  # plot_convergence, plot_pairwise_heatmaps, plot_parallel_coordinates

# ==========================================
# 1. LOAD (OR GENERATE, ON FIRST RUN) THE SYNTHETIC DATA/MC PAIR
# ==========================================
data_raw, mc_raw = load_or_generate()

# ==========================================
# 2. HYPERPARAMETER TUNING LOOP (EXPANDED SEARCH SPACE)
# ==========================================
println("Running 12D two-sided hyperparameter tuning loop via Hyperopt...")
t_start = time()

ho = @hyperopt for i = 150,
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

t_end = time()
println("Tuning completed in $(round(t_end - t_start, digits=2)) seconds.")

# ==========================================
# RE-CALCULATE BASELINE HISTOGRAMS BEFORE PLOTTING
# ==========================================
h_data_before = fit(Histogram, data_raw.energy, ENERGY_BINS)
h_mc_before = fit(Histogram, mc_raw.energy, ENERGY_BINS)
w_mc_before_norm = h_mc_before.weights .* (sum(h_data_before.weights) / sum(h_mc_before.weights))

# Unpack all 12 optimized bounds explicitly from the minimizer
c_chi2_l, c_chi2_h, c_iso_l, c_iso_h, c_hits_l, c_hits_h, c_time_l, c_time_h, c_had_l, c_had_h, c_vtx_l, c_vtx_h = ho.minimizer

println("\n" * "="^40)
println("OPTIMIZED 12D QUALITY WINDOW CUT RESULTS:")
println("="^40)
println("  track_chi2        : [", round(c_chi2_l, digits=2), ", ", round(c_chi2_h, digits=2), "]")
println("  isolation         : [", round(c_iso_l, digits=2), ", ", round(c_iso_h, digits=2), "]")
println("  hits_count        : [", Int(c_hits_l), ", ", Int(c_hits_h), "]")
println("  timing_ns         : [", round(c_time_l, digits=2), ", ", round(c_time_h, digits=2), "]")
println("  hadronic_fraction : [", round(c_had_l, digits=2), ", ", round(c_had_h, digits=2), "]")
println("  vertex_dist       : [", round(c_vtx_l, digits=2), ", ", round(c_vtx_h, digits=2), "]")
println("="^40 * "\n")

# ==========================================
# 2b. HYPEROPT DIAGNOSTIC PLOTS (convergence, loss landscape, parallel coords)
# ==========================================
const PARAM_NAMES_12D = [
    :c_chi2_l, :c_chi2_h, :c_iso_l, :c_iso_h, :c_hits_l, :c_hits_h,
    :c_time_l, :c_time_h, :c_had_l, :c_had_h, :c_vtx_l, :c_vtx_h,
]  # must match the @hyperopt declaration order above exactly

plot_convergence(ho; title="12D Quality-Window Tuning (RandomSampler)", out_path="output/convergence")
plot_pairwise_heatmaps(ho, PARAM_NAMES_12D, default_lo_hi_pairs(length(PARAM_NAMES_12D));
    out_path="output/loss_landscape")
plot_parallel_coordinates(ho, PARAM_NAMES_12D; out_path="output/parallel_coordinates")

# Apply 12D constraints to filter the data array blocks
d_opt = data_raw[
    (c_chi2_l .<= data_raw.track_chi2 .<= c_chi2_h) .& (c_iso_l .<= data_raw.isolation .<= c_iso_h) .& (c_hits_l .<= data_raw.hits_count .<= c_hits_h) .& (c_time_l .<= data_raw.timing_ns .<= c_time_h) .& (c_had_l .<= data_raw.hadronic_fraction .<= c_had_h) .& (c_vtx_l .<= data_raw.vertex_dist .<= c_vtx_h), :]

m_opt = mc_raw[
    (c_chi2_l .<= mc_raw.track_chi2 .<= c_chi2_h) .& (c_iso_l .<= mc_raw.isolation .<= c_iso_h) .& (c_hits_l .<= mc_raw.hits_count .<= c_hits_h) .& (c_time_l .<= mc_raw.timing_ns .<= c_time_h) .& (c_had_l .<= mc_raw.hadronic_fraction .<= c_had_h) .& (c_vtx_l .<= mc_raw.vertex_dist .<= c_vtx_h), :]

h_data_after = fit(Histogram, d_opt.energy, ENERGY_BINS)
h_mc_after = fit(Histogram, m_opt.energy, ENERGY_BINS)
w_mc_after_norm = h_mc_after.weights .* (sum(h_data_after.weights) / sum(h_mc_after.weights))

# ==========================================
# 3. PLOTTING THE OUTCOME (WITH LOG SYNC & GRIDLINES)
# ==========================================
println("Generating publication-quality main/ratio plots with CairoMakie...")

bin_edges = collect(ENERGY_BINS)

# Compute ratios cleanly (Handling division by zero safely)
ratio_before = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_before.weights, w_mc_before_norm)]
ratio_after = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_after.weights, w_mc_after_norm)]

# Establish standard publication canvas dimensions
fig = Figure(size=(850, 450), font="DejaVu Sans")

gl1 = fig[1, 1] = GridLayout()
gl2 = fig[1, 2] = GridLayout()
gl3 = fig[1, 3] = GridLayout()

# Force layout columns to share space proportionally (Sidebar gets 20%)
colsize!(fig.layout, 1, Relative(0.40))
colsize!(fig.layout, 2, Relative(0.38)) # Slightly smaller to account for hidden y-axis labels
colsize!(fig.layout, 3, Relative(0.22))

# ----------------------------------------------------
# PANEL A: BEFORE TUNING (Main + Ratio)
# ----------------------------------------------------
ax1_main = Axis(gl1[1, 1],
    title="Before Tuning (Poor Agreement)",
    ylabel="Events / 2 GeV",
    xticklabelsvisible=false, xticksvisible=false,
    xgridvisible=true, ygridvisible=true,    # Enabled clean background gridlines
    xgridstyle=:dash, ygridstyle=:dash,
    xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25)
)
ax1_ratio = Axis(gl1[2, 1],
    xlabel="Particle Energy [GeV]", ylabel="Data / MC",
    xgridvisible=true, ygridvisible=true,
    xgridstyle=:dash, ygridstyle=:dash,
    xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25)
)
rowsize!(gl1, 1, Relative(3/4))

p_data_pre = stairs!(ax1_main, bin_edges, [h_data_before.weights; 0], color=:black, linewidth=2, step=:post)
p_mc_pre = stairs!(ax1_main, bin_edges, [w_mc_before_norm; 0], color=:red, linewidth=2, linestyle=:dash, step=:post)

stairs!(ax1_ratio, bin_edges, [ratio_before; 1.0], color=:red, linewidth=1.5, step=:post)
hlines!(ax1_ratio, [1.0], color=:gray, linestyle=:dash, linewidth=1.2)
ylims!(ax1_ratio, 0.4, 1.6)

# ----------------------------------------------------
# PANEL B: AFTER TUNING (Main + Ratio)
# ----------------------------------------------------
ax2_main = Axis(gl2[1, 1],
    title="After Quality Window Tuning",
    yticklabelsvisible=false, yticksvisible=false, # Hiding redundant labels since scales are linked
    xticklabelsvisible=false, xticksvisible=false,
    xgridvisible=true, ygridvisible=true,    # Enabled clean background gridlines
    xgridstyle=:dash, ygridstyle=:dash,
    xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25)
)
ax2_ratio = Axis(gl2[2, 1],
    xlabel="Particle Energy [GeV]",
    yticklabelsvisible=false, yticksvisible=false, # Hiding redundant ratio labels as well
    xgridvisible=true, ygridvisible=true,
    xgridstyle=:dash, ygridstyle=:dash,
    xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25)
)
rowsize!(gl2, 1, Relative(3/4))

p_data_post = stairs!(ax2_main, bin_edges, [h_data_after.weights; 0], color=:black, linewidth=2, step=:post)
p_mc_post = stairs!(ax2_main, bin_edges, [w_mc_after_norm; 0], color=:dodgerblue, linewidth=2, linestyle=:dash, step=:post)

stairs!(ax2_ratio, bin_edges, [ratio_after; 1.0], color=:dodgerblue, linewidth=1.5, step=:post)
hlines!(ax2_ratio, [1.0], color=:gray, linestyle=:dash, linewidth=1.2)
ylims!(ax2_ratio, 0.4, 1.6)

# ----------------------------------------------------
# AXIS LINKING MATRIX
# ----------------------------------------------------
linkyaxes!(ax1_main, ax2_main)   # FIXED: Synchronized vertical heights perfectly
linkyaxes!(ax1_ratio, ax2_ratio) # FIXED: Synchronized ratio vertical heights perfectly

linkxaxes!(ax1_main, ax1_ratio)
linkxaxes!(ax2_main, ax2_ratio)

rowgap!(gl1, 6)
rowgap!(gl2, 6)
colgap!(fig.layout, 12) # Tighten up center gutter whitespace

# ----------------------------------------------------
# SIDEBAR INFO PANEL (LEGEND & METRICS)
# ----------------------------------------------------
Legend(gl3[1, 1],
    [p_data_pre, p_mc_pre, p_mc_post],
    ["Data", "MC Raw", "MC Tuned"],
    framevisible=false, halign=:left, fontsize=11
)

cuts_text = """
Optimized Boundaries:
• χ²: [$(round(c_chi2_l, digits=1)), $(round(c_chi2_h, digits=1))]
• Iso: [$(round(c_iso_l, digits=2)), $(round(c_iso_h, digits=2))]
• Hits: [$(Int(c_hits_l)), $(Int(c_hits_h))]
• Time: [$(round(c_time_l, digits=1)), $(round(c_time_h, digits=1))]
• Had: [$(round(c_had_l, digits=2)), $(round(c_had_h, digits=2))]
• Vtx: [$(round(c_vtx_l, digits=1)), $(round(c_vtx_h, digits=1))]
"""

Label(gl3[2, 1], cuts_text, justification=:left, halign=:left, valign=:top, fontsize=11, font="DejaVu Sans Mono")
rowsize!(gl3, 1, Auto())
rowsize!(gl3, 2, Auto())

# Save the final synced vector graphic assets
save("output/energy_distribution_comparison.png", fig, px_per_unit=2)
save("output/energy_distribution_comparison.pdf", fig)
println("Saved complete synchronized assets to 'energy_distribution_comparison.png' and '.pdf'")