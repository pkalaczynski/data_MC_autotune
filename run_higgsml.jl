using Pkg
Pkg.activate(".")
using DataFrames
using StatsBase
using Hyperopt
using CairoMakie
using Random
using PairPlots

include("common_higgsml.jl")     # TEST_BINS, REJECT_MIN/MAX, load_higgsml, compute_phase_space_loss
include("hyperopt_diagnostics.jl") # plot_convergence, plot_pairwise_heatmaps, plot_parallel_coordinates,
# default_lo_hi_pairs

# ==========================================
# CONFIG
# ==========================================
const PARQUET_PATH = "data/FAIR_Universe_HiggsML_data.parquet"
const N_TRIALS = 500   # matches higgsml_optimize.py / run_higgsml_multiseed.jl

# ==========================================
# 1. LOAD (500k-row random subsample of) THE REAL HIGGSML DATASET
# ==========================================
df_all = load_higgsml(PARQUET_PATH)

# ==========================================
# 2. HYPERPARAMETER TUNING LOOP (8D KINEMATIC OBJECT SELECTION)
# ==========================================
println("Running 8D kinematic object-selection tuning loop via Hyperopt...")
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

t_end = time()
println("Tuning completed in $(round(t_end - t_start, digits=2)) seconds.")

# Unpack the 8 optimized bounds explicitly from the minimizer
c_had_l, c_had_h, c_lep_l, c_lep_h, c_lead_l, c_lead_h, c_sub_l, c_sub_h = ho.minimizer

println("\n" * "="^40)
println("OPTIMIZED 8D KINEMATIC OBJECT SELECTION RESULTS:")
println("="^40)
println("  PRI_had_pt (τhad)        : [", round(c_had_l, digits=1), ", ", round(c_had_h, digits=1), "]")
println("  PRI_lep_pt (τlep)        : [", round(c_lep_l, digits=1), ", ", round(c_lep_h, digits=1), "]")
println("  PRI_jet_leading_pt       : [", round(c_lead_l, digits=1), ", ", round(c_lead_h, digits=1), "]")
println("  PRI_jet_subleading_pt    : [", round(c_sub_l, digits=1), ", ", round(c_sub_h, digits=1), "]")
println("="^40 * "\n")

# ==========================================
# 2b. HYPEROPT DIAGNOSTIC PLOTS (convergence, loss landscape, parallel coords)
# ==========================================
const PARAM_NAMES_8D = [
    :c_had_l, :c_had_h, :c_lep_l, :c_lep_h, :c_lead_l, :c_lead_h, :c_sub_l, :c_sub_h,
]  # must match the @hyperopt declaration order above exactly

plot_convergence(ho; title="8D HiggsML Kinematic Selection Tuning (RandomSampler)",
    out_path="output/higgsml_convergence")
plot_pairwise_heatmaps(ho, PARAM_NAMES_8D, default_lo_hi_pairs(length(PARAM_NAMES_8D));
    out_path="output/higgsml_loss_landscape")
plot_parallel_coordinates(ho, PARAM_NAMES_8D; out_path="output/higgsml_parallel_coordinates")

# ==========================================
# 3. DEFAULT (REFERENCE) PHYSICS CUTS
#
# From the standard HiggsML object-selection table. Only lower ("post
# selection") pT thresholds are quoted for the four objects tuned here --
# there is no reference upper bound, so it's left open (Inf) rather than
# invented. Number-of-object / opposite-charge requirements aren't part of
# this 8D pT-window search space and so aren't applied here.
# ==========================================
const DEFAULT_CUTS = (
    had_l=26.0, had_h=Inf,
    lep_l=20.0, lep_h=Inf,
    lead_l=26.0, lead_h=Inf,
    sub_l=26.0, sub_h=Inf,
)

function apply_object_cuts(df, had_l, had_h, lep_l, lep_h, lead_l, lead_h, sub_l, sub_h)
    df[
        (had_l .<= df.PRI_had_pt .<= had_h) .& (lep_l .<= df.PRI_lep_pt .<= lep_h) .& (lead_l .<= df.PRI_jet_leading_pt .<= lead_h) .& (sub_l .<= df.PRI_jet_subleading_pt .<= sub_h), :]
end

d_untuned = df_all
d_default = apply_object_cuts(df_all, DEFAULT_CUTS.had_l, DEFAULT_CUTS.had_h,
    DEFAULT_CUTS.lep_l, DEFAULT_CUTS.lep_h,
    DEFAULT_CUTS.lead_l, DEFAULT_CUTS.lead_h,
    DEFAULT_CUTS.sub_l, DEFAULT_CUTS.sub_h)
d_tuned = apply_object_cuts(df_all, c_had_l, c_had_h, c_lep_l, c_lep_h,
    c_lead_l, c_lead_h, c_sub_l, c_sub_h)

println("Selection survival (of $(nrow(df_all)) events):")
println("  Untuned  : $(nrow(d_untuned)) ($(round(100*nrow(d_untuned)/nrow(df_all), digits=1))%)")
println("  Default  : $(nrow(d_default)) ($(round(100*nrow(d_default)/nrow(df_all), digits=1))%)")
println("  Autotuned: $(nrow(d_tuned)) ($(round(100*nrow(d_tuned)/nrow(df_all), digits=1))%)")

# ==========================================
# 4. SIGNAL / BACKGROUND SPLIT, AND DATA vs MC HISTOGRAMS
#
# "MC" is the physics-weight-corrected (luminosity-scaled) prediction.
# "Data" is the same events treated as pseudo-observed collision data --
# i.e. raw, unweighted event *counts* -- exactly the Data/MC convention
# used for the synthetic 12D task (common.jl's compute_agreement_loss):
# MC is rate-normalized so its total matches Data's total, then Data/MC
# should sit near 1.0 wherever the (weighted) MC shape describes the
# (unweighted) observed rate well.
# ==========================================
function split_signal_background(df)
    labels = df.labels
    is_signal = if eltype(labels) <: AbstractString
        (labels .== "s") .| (labels .== "S") .| (lowercase.(String.(labels)) .== "signal")
    else
        labels .== 1
    end
    return df[is_signal, :], df[.!is_signal, :]
end

# Unweighted event counts -- stands in for "Data".
counts_hist(df) = isempty(df.PRI_met) ? zeros(length(TEST_BINS) - 1) :
                  fit(Histogram, df.PRI_met, TEST_BINS).weights

# Physics-weighted yields -- stands in for "MC".
weighted_hist(df) = isempty(df.PRI_met) ? zeros(length(TEST_BINS) - 1) :
                    fit(Histogram, df.PRI_met, weights(df.weights), TEST_BINS).weights

bin_edges = collect(TEST_BINS)
bin_centers = 0.5 .* (bin_edges[1:(end-1)] .+ bin_edges[2:end])
const SIDEBAND_MASK = (bin_centers .< REJECT_MIN) .| (bin_centers .> REJECT_MAX)

"""
    sideband_chi2(data_total, mc_total)

Data-vs-MC chi2 in the sideband bins only (same convention as
common.jl's compute_agreement_loss on the synthetic 12D task):
sum((data - mc)^2 / (data + mc)) over bins outside [REJECT_MIN, REJECT_MAX],
skipping empty bins. Purely a display/diagnostic metric for the plot below
-- it is NOT the objective compute_phase_space_loss actually optimizes
against (that one compares weighted MC to a flat mean, with no separate
"Data" side; see common_higgsml.jl), so don't expect the two to match.
"""
function sideband_chi2(data_total, mc_total)
    chi2 = 0.0
    for i in findall(SIDEBAND_MASK)
        d, m = data_total[i], mc_total[i]
        if d + m > 0
            chi2 += (d - m)^2 / (d + m)
        end
    end
    return chi2
end

"""
    panel_histograms(df)

Returns (data_sig, data_bkg, mc_sig, mc_bkg, ratio, chi2) for one
selection's DataFrame: Data (unweighted counts) and MC (weighted yields,
both split into signal/background), MC rate-normalized to Data's total so
the Data/MC ratio is meaningful, and the sideband Data-vs-MC chi2 (see
sideband_chi2 above).
"""
function panel_histograms(df)
    d_sig, d_bkg = split_signal_background(df)

    data_sig = counts_hist(d_sig)
    data_bkg = counts_hist(d_bkg)
    data_total = data_sig .+ data_bkg

    mc_sig_raw = weighted_hist(d_sig)
    mc_bkg_raw = weighted_hist(d_bkg)
    mc_total_raw = mc_sig_raw .+ mc_bkg_raw

    norm = sum(mc_total_raw) > 0 ? sum(data_total) / sum(mc_total_raw) : 1.0
    mc_sig = mc_sig_raw .* norm
    mc_bkg = mc_bkg_raw .* norm
    mc_total = mc_sig .+ mc_bkg

    ratio = [m > 0 ? d / m : 1.0 for (d, m) in zip(data_total, mc_total)]
    chi2 = sideband_chi2(data_total, mc_total)

    return data_sig, data_bkg, mc_sig, mc_bkg, ratio, chi2
end

panels = [
    ("1. Untuned Input", d_untuned, :gray45),
    ("2. Post-Selection Cuts", d_default, :darkorange),
    ("3. Autotuned 8D Gates", d_tuned, :dodgerblue),
]

# ==========================================
# 5. PLOTTING: 3-PANEL DATA/MC COMPARISON (SIGNAL + BACKGROUND, WITH RATIO)
# ==========================================
"""
    make_higgsml_comparison_plot(panels, bin_edges)

Builds and saves the 3-panel Data/MC comparison figure. Wrapped in a
function (rather than left as top-level code) so that the
p_*_first legend-handle variables assigned inside the `for` loop use
normal function-local scoping -- at top level, Julia's soft-scope rules
would otherwise treat each loop-body assignment as a new local shadowing
the outer variable, silently leaving the outer one as `nothing`.
"""
function make_higgsml_comparison_plot(panels, bin_edges)
    fig = Figure(size=(1500, 520), font="DejaVu Sans")

    gl_main = [fig[1, i] = GridLayout() for i in 1:3]
    gl_side = fig[1, 4] = GridLayout()
    colsize!(fig.layout, 4, Relative(0.17))

    axes_top = Axis[]
    axes_ratio = Axis[]
    p_data_sig_first = p_data_bkg_first = p_mc_sig_first = p_mc_bkg_first = nothing

    for (i, (title, df, color)) in enumerate(panels)
        data_sig, data_bkg, mc_sig, mc_bkg, ratio, chi2 = panel_histograms(df)

        ax_top = Axis(gl_main[i][1, 1],
            title=title,
            ylabel=i == 1 ? "Weighted Events" : "",
            yticklabelsvisible=(i == 1),
            xticklabelsvisible=false, xticksvisible=false,
            xgridvisible=true, ygridvisible=true,
            xgridstyle=:dash, ygridstyle=:dash,
            xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25),
        )
        ax_ratio = Axis(gl_main[i][2, 1],
            xlabel="Missing ET [GeV]",
            ylabel=i == 1 ? "Data / MC" : "",
            yticklabelsvisible=(i == 1),
            xgridvisible=true, ygridvisible=true,
            xgridstyle=:dash, ygridstyle=:dash,
            xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25),
        )
        rowsize!(gl_main[i], 1, Relative(3 / 4))

        p_data_sig = stairs!(ax_top, bin_edges, [data_sig; 0], color=:black, linewidth=2.0, step=:post)
        p_data_bkg = stairs!(ax_top, bin_edges, [data_bkg; 0], color=:black, linewidth=1.3,
            linestyle=:dot, step=:post)
        p_mc_sig = stairs!(ax_top, bin_edges, [mc_sig; 0], color=color, linewidth=2.0,
            linestyle=:dash, step=:post)
        p_mc_bkg = stairs!(ax_top, bin_edges, [mc_bkg; 0], color=(color, 0.6), linewidth=1.3,
            linestyle=:dashdot, step=:post)

        vspan!(ax_top, [REJECT_MIN], [REJECT_MAX], color=(:red, 0.06))

        text!(ax_top, 0.97, 0.95, text="χ² (sideband) = $(round(chi2, digits=1))",
            space=:relative, align=(:right, :top), fontsize=11, color=:black)

        stairs!(ax_ratio, bin_edges, [ratio; 1.0], color=color, linewidth=1.5, step=:post)
        hlines!(ax_ratio, [1.0], color=:gray, linestyle=:dash, linewidth=1.2)
        ylims!(ax_ratio, 0.4, 1.6)
        vspan!(ax_ratio, [REJECT_MIN], [REJECT_MAX], color=(:red, 0.06))

        linkxaxes!(ax_top, ax_ratio)

        if i == 1
            p_data_sig_first, p_data_bkg_first = p_data_sig, p_data_bkg
            p_mc_sig_first, p_mc_bkg_first = p_mc_sig, p_mc_bkg
        end

        push!(axes_top, ax_top)
        push!(axes_ratio, ax_ratio)

        rowgap!(gl_main[i], 6)
    end

    linkyaxes!(axes_top...)
    linkyaxes!(axes_ratio...)

    Legend(gl_side[1, 1],
        [p_data_sig_first, p_data_bkg_first, p_mc_sig_first, p_mc_bkg_first],
        ["Data (signal)", "Data (background)", "MC (signal)", "MC (background)"],
        framevisible=false, halign=:left, fontsize=10
    )

    cuts_text = """
    Benchmark Cuts:
    • pT Had     : > $(round(DEFAULT_CUTS.had_l, digits=1)) GeV
    • pT Lep     : > $(round(DEFAULT_CUTS.lep_l, digits=1)) GeV
    • pT Leading Jet: > $(round(DEFAULT_CUTS.lead_l, digits=1)) GeV
    • pT Sublead Jet: > $(round(DEFAULT_CUTS.sub_l, digits=1)) GeV

    Tuned Cuts:
    • pT Had     : [$(round(c_had_l, digits=1)), $(round(c_had_h, digits=1))]
    • pT Lep     : [$(round(c_lep_l, digits=1)), $(round(c_lep_h, digits=1))]
    • pT Leading Jet: [$(round(c_lead_l, digits=1)), $(round(c_lead_h, digits=1))]
    • pT Sublead Jet: [$(round(c_sub_l, digits=1)), $(round(c_sub_h, digits=1))]

    Shaded band: sideband
    reject window
    [$(REJECT_MIN), $(REJECT_MAX)] GeV
    """

    Label(gl_side[2, 1], cuts_text, justification=:left, halign=:left, valign=:top,
        fontsize=9, font="DejaVu Sans Mono")
    rowsize!(gl_side, 1, Auto())
    rowsize!(gl_side, 2, Auto())

    save("output/higgsml_energy_distribution_comparison.png", fig, px_per_unit=2)
    save("output/higgsml_energy_distribution_comparison.pdf", fig)
    println("Saved 3-panel HiggsML Data/MC comparison figure to 'output/higgsml_energy_distribution_comparison.png' and '.pdf'")

    return fig
end

println("Generating 3-panel HiggsML Data/MC comparison with CairoMakie...")
make_higgsml_comparison_plot(panels, bin_edges)