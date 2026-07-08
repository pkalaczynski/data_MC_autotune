using DataFrames
using StatsBase
using Hyperopt
using CairoMakie
using Random
using PairPlots

# ==========================================
# CONSTANTS & CONFIGURATION (Moved to global scope)
# ==========================================
const ENERGY_BINS = 20.0:2.0:120.0
const SIGNAL_MIN = 82.0
const SIGNAL_MAX = 98.0

# ==========================================
# 1. REVISED SYNTHETIC DATA GENERATOR (WITH REAL MISMATCH)
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

println("Generating distorted datasets...")
# Using slightly optimized event targets to guarantee fast memory allocations
data_raw = generate_synthetic_data(600000, seed=123, is_mc=false)
mc_raw = generate_synthetic_data(800000, seed=456, is_mc=true)

# ==========================================
# 2. EVALUATION & OBJECTIVE FUNCTION (WITH EFFICIENCY CONSTRAINT)
# ==========================================
function compute_agreement_loss(df_data, df_mc, cuts)
    chi2_low, chi2_high = cuts[1], cuts[2]
    iso_low, iso_high = cuts[3], cuts[4]
    hits_low, hits_high = cuts[5], cuts[6]
    time_low, time_high = cuts[7], cuts[8]
    had_low, had_high = cuts[9], cuts[10]
    vtx_low, vtx_high = cuts[11], cuts[12]

    # Apply window constraints
    d_sub = df_data[
        (chi2_low .<= df_data.track_chi2 .<= chi2_high) .& (iso_low .<= df_data.isolation .<= iso_high) .& (hits_low .<= df_data.hits_count .<= hits_high) .& (time_low .<= df_data.timing_ns .<= time_high) .& (had_low .<= df_data.hadronic_fraction .<= had_high) .& (vtx_low .<= df_data.vertex_dist .<= vtx_high), :]

    m_sub = df_mc[
        (chi2_low .<= df_mc.track_chi2 .<= chi2_high) .& (iso_low .<= df_mc.isolation .<= iso_high) .& (hits_low .<= df_mc.hits_count .<= hits_high) .& (time_low .<= df_mc.timing_ns .<= time_high) .& (had_low .<= df_mc.hadronic_fraction .<= had_high) .& (vtx_low .<= df_mc.vertex_dist .<= vtx_high), :]

    # Calculate global data survival fraction
    survival_fraction = size(d_sub, 1) / size(df_data, 1)

    # CRITICAL FIX: If the optimizer tries to wipe out the dataset (survival < 20%),
    # hit it with a massive penalty to protect the signal peak.
    if survival_fraction < 0.20
        return 1e5 * (0.20 - survival_fraction)
    end

    # Isolate sidebands for shape matching
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
# 3. HYPERPARAMETER TUNING LOOP (EXPANDED SEARCH SPACE)
# ==========================================
println("Running 12D two-sided hyperparameter tuning loop via Hyperopt...")
t_start = time()

# FIXED: Upper bounds expanded significantly to allow windows to stay wide open
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
# 4. PLOTTING THE OUTCOME WITH MAIN, RATIO, AND CUT LABELS
# ==========================================
h_data_before = fit(Histogram, data_raw.energy, ENERGY_BINS)
h_mc_before = fit(Histogram, mc_raw.energy, ENERGY_BINS)
w_mc_before_norm = h_mc_before.weights .* (sum(h_data_before.weights) / sum(h_mc_before.weights))

# FIXED: Unpack all 12 optimized bounds from the minimizer instead of 6
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

# Apply 12D constraints to filter the data array blocks
d_opt = data_raw[
    (c_chi2_l .<= data_raw.track_chi2 .<= c_chi2_h) .& (c_iso_l .<= data_raw.isolation .<= c_iso_h) .& (c_hits_l .<= data_raw.hits_count .<= c_hits_h) .& (c_time_l .<= data_raw.timing_ns .<= c_time_h) .& (c_had_l .<= data_raw.hadronic_fraction .<= c_had_h) .& (c_vtx_l .<= data_raw.vertex_dist .<= c_vtx_h), :]

m_opt = mc_raw[
    (c_chi2_l .<= mc_raw.track_chi2 .<= c_chi2_h) .& (c_iso_l .<= mc_raw.isolation .<= c_iso_h) .& (c_hits_l .<= mc_raw.hits_count .<= c_hits_h) .& (c_time_l .<= mc_raw.timing_ns .<= c_time_h) .& (c_had_l .<= mc_raw.hadronic_fraction .<= c_had_h) .& (c_vtx_l .<= mc_raw.vertex_dist .<= c_vtx_h), :]

h_data_after = fit(Histogram, d_opt.energy, ENERGY_BINS)
h_mc_after = fit(Histogram, m_opt.energy, ENERGY_BINS)
w_mc_after_norm = h_mc_after.weights .* (sum(h_data_after.weights) / sum(h_mc_after.weights))

println("Generating publication-quality main/ratio plots with CairoMakie...")
bin_edges = collect(ENERGY_BINS)

ratio_before = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_before.weights, w_mc_before_norm)]
ratio_after = [m > 0 ? d / m : 1.0 for (d, m) in zip(h_data_after.weights, w_mc_after_norm)]

fig = Figure(size=(1100, 600), font="DejaVu Sans")
gl1 = fig[1, 1] = GridLayout()
gl2 = fig[1, 2] = GridLayout()
gl3 = fig[1, 3] = GridLayout()

# PANEL A: Before Tuning
ax1_main = Axis(gl1[1, 1], title="Before Tuning (Poor Agreement)", ylabel="Events / 2 GeV", xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)
ax1_ratio = Axis(gl1[2, 1], xlabel="Particle Energy [GeV]", ylabel="Data / MC", xgridvisible=false, ygridvisible=false)
rowsize!(gl1, 1, Relative(3/4))
p_data_pre = stairs!(ax1_main, bin_edges, [h_data_before.weights; 0], color=:black, linewidth=2, step=:post)
p_mc_pre = stairs!(ax1_main, bin_edges, [w_mc_before_norm; 0], color=:red, linewidth=2, linestyle=:dash, step=:post)
stairs!(ax1_ratio, bin_edges, [ratio_before; 1.0], color=:red, linewidth=1.5, step=:post)
hlines!(ax1_ratio, [1.0], color=:gray, linestyle=:dash)
ylims!(ax1_ratio, 0.4, 1.6)

# PANEL B: After Tuning
ax2_main = Axis(gl2[1, 1], title="After Quality Window Tuning", ylabel="Events / 2 GeV", xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)
ax2_ratio = Axis(gl2[2, 1], xlabel="Particle Energy [GeV]", ylabel="Data / MC", xgridvisible=false, ygridvisible=false)
rowsize!(gl2, 1, Relative(3/4))
p_data_post = stairs!(ax2_main, bin_edges, [h_data_after.weights; 0], color=:black, linewidth=2, step=:post)
p_mc_post = stairs!(ax2_main, bin_edges, [w_mc_after_norm; 0], color=:dodgerblue, linewidth=2, linestyle=:dash, step=:post)
stairs!(ax2_ratio, bin_edges, [ratio_after; 1.0], color=:dodgerblue, linewidth=1.5, step=:post)
hlines!(ax2_ratio, [1.0], color=:gray, linestyle=:dash)
ylims!(ax2_ratio, 0.4, 1.6)

linkxaxes!(ax1_main, ax1_ratio)
linkxaxes!(ax2_main, ax2_ratio)
rowgap!(gl1, 5)
rowgap!(gl2, 5)

Legend(gl3[1, 1], [p_data_pre, p_mc_pre, p_mc_post], ["Data", "MC Raw (Normed)", "MC Tuned (Normed)"], framevisible=false, halign=:left)

cuts_text = """
Optimized Boundaries:
• χ²: [$(round(c_chi2_l, digits=1)), $(round(c_chi2_h, digits=1))]
• Iso: [$(round(c_iso_l, digits=2)), $(round(c_iso_h, digits=2))]
• Hits: [$(Int(c_hits_l)), $(Int(c_hits_h))]
• Time: [$(round(c_time_l, digits=1)), $(round(c_time_h, digits=1))]
• Had: [$(round(c_had_l, digits=2)), $(round(c_had_h, digits=2))]
• Vtx: [$(round(c_vtx_l, digits=1)), $(round(c_vtx_h, digits=1))]
"""

Label(gl3[2, 1], cuts_text, justification=:left, halign=:left, valign=:top, fontsize=12, font="DejaVu Sans Mono")
rowsize!(gl3, 1, Auto())
rowsize!(gl3, 2, Auto())

save("energy_distribution_comparison.png", fig, px_per_unit=2)
save("energy_distribution_comparison.pdf", fig)
println("Saved full assets with cut configurations to 'energy_distribution_comparison.png' and '.pdf'")

# ==========================================
# 5. BASELINE CORRELATION MATRIX (DATASET DESCRIPTION)
# ==========================================
println("Generating baseline correlation matrix pair plots for dataset description (Contour mode)...")

labels_mapping = Dict(
    :energy => "Energy [GeV]",
    :track_chi2 => "Track χ²",
    :isolation => "Isolation",
    :hits_count => "Hits Count",
    :timing_ns => "Timing [ns]",
    :hadronic_fraction => "Hadronic Frac",
    :vertex_dist => "Vertex Dist [mm]"
)

plot_cols = [:energy, :track_chi2, :isolation, :hits_count, :timing_ns, :hadronic_fraction, :vertex_dist]
df_data_raw_subset = data_raw[!, plot_cols]
df_mc_raw_subset = mc_raw[!, plot_cols]

fig_corner = Figure(size=(1200, 1200), font="DejaVu Sans")

pairplot(
    fig_corner[1, 1],
    df_data_raw_subset => (
        PairPlots.Contour(color=:black, linewidth=1.5),
        PairPlots.MarginHist(color=(:black, 0.35), bins=30)
    ),
    df_mc_raw_subset => (
        PairPlots.Contour(color=:red, linewidth=1.5),
        PairPlots.MarginHist(color=(:red, 0.35), bins=30)
    );
    labels=labels_mapping
)

Legend(fig_corner[1, 1],
    [LineElement(color=:black, linewidth=2), LineElement(color=:red, linewidth=2)],
    ["Raw Data", "Raw MC Simulation"],
    tellheight=false, tellwidth=false,
    halign=:right, valign=:top, margin=(0, 40, 40, 0),
    framevisible=false
)

save("baseline_correlation_matrix.png", fig_corner, px_per_unit=2)
save("baseline_correlation_matrix.pdf", fig_corner)
println("Saved baseline dataset correlation pairplot matrix to 'baseline_correlation_matrix.png' and '.pdf'")