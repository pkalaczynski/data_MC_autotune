using CairoMakie
using StatsBase
using Statistics

# ==========================================
# Generic Hyperopt.jl diagnostics.
#
# These functions only assume `ho` is the object returned by `@hyperopt`:
#   ho.results   -- Vector of objective values, one per trial, in trial order
#   ho.history   -- Vector of per-trial parameter tuples, SAME order as
#                   declared in the @hyperopt macro (ho.history[t][k] is the
#                   k-th declared parameter's value on trial t)
#
# You must pass `param_names` yourself (a Vector{Symbol} in the exact order
# the parameters were declared in your @hyperopt call) -- this is safer than
# relying on any Hyperopt.jl internal field for parameter names, which this
# code doesn't depend on.
# ==========================================

"""
    default_lo_hi_pairs(n_params)

Convenience default for `plot_pairwise_heatmaps`: assumes parameters come in
adjacent (low, high) pairs, as every search space in this repo does --
e.g. for 12 params, returns [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)].
"""
default_lo_hi_pairs(n_params::Int) = [(2i - 1, 2i) for i in 1:div(n_params, 2)]

# ------------------------------------------------------------------
# 1. Convergence: loss vs. trial, plus running-best-so-far
# ------------------------------------------------------------------
function plot_convergence(ho; title::String="Optimization History", out_path::String="output/convergence")
    losses = Float64.(ho.results)
    losses_plot = max.(losses, 1e-3)  # guard against log10(0) on the off chance a trial scores exactly 0
    n = length(losses)
    running_min = accumulate(min, losses_plot)

    fig = Figure(size=(700, 420), font="DejaVu Sans")
    ax = Axis(fig[1, 1],
        title=title,
        xlabel="Trial", ylabel="Loss (χ², log scale)",
        yscale=log10,
        xgridvisible=true, ygridvisible=true,
        xgridstyle=:dash, ygridstyle=:dash,
        xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25),
    )

    scatter!(ax, 1:n, losses_plot, color=(:dodgerblue, 0.5), markersize=6, label="Trial loss")
    lines!(ax, 1:n, running_min, color=:black, linewidth=2, label="Best so far")

    best_i = argmin(losses_plot)
    scatter!(ax, [best_i], [losses_plot[best_i]], color=:red, marker=:star5, markersize=16, label="Minimum")

    axislegend(ax, position=:rt, margin=(10, 10, 10, 10))

    save("$(out_path).png", fig, px_per_unit=2)
    save("$(out_path).pdf", fig)
    println("Saved convergence plot to '$(out_path).png' and '.pdf'")
    return out_path
end

# ------------------------------------------------------------------
# 2. Pairwise loss-landscape heatmaps (mean loss per 2D bin), one panel
#    per (param_i, param_j) pair, laid out in a grid with a shared colorbar.
#    Same binning approach as the original diagnostic in
#    test_higgsml_parquet.jl, generalized to arbitrary parameter pairs.
# ------------------------------------------------------------------
function plot_pairwise_heatmaps(ho, param_names::Vector{Symbol}, pairs::Vector{Tuple{Int,Int}};
                                  color_percentile::Float64=0.90, bins::Int=25,
                                  out_path::String="output/loss_landscape")
    losses = Float64.(ho.results)
    n = length(losses)

    # Rather than guessing an absolute "penalty" magnitude cutoff (fragile --
    # it depends on dataset scale, and "artificially penalized" vs "just a
    # poorly-tuned random draw" produce overlapping ranges of values), clip
    # the COLOR SCALE at a percentile of the observed losses instead. Every
    # trial still contributes to the binning; a few extreme values just get
    # visually saturated rather than dominating the gradient.
    color_lo = minimum(losses)
    color_hi = quantile(losses, color_percentile)

    n_pairs = length(pairs)
    ncols = min(3, n_pairs)
    nrows = cld(n_pairs, ncols)

    fig = Figure(size=(380 * ncols, 340 * nrows), font="DejaVu Sans")
    hm_last = nothing

    for (k, (i, j)) in enumerate(pairs)
        row = cld(k, ncols)
        col = k - (row - 1) * ncols

        xi = [ho.history[t][i] for t in 1:n]
        yj = [ho.history[t][j] for t in 1:n]
        xname, yname = string(param_names[i]), string(param_names[j])

        ax = Axis(fig[row, col],
            title="$(xname) vs $(yname)",
            xlabel=xname, ylabel=yname,
            xgridvisible=false, ygridvisible=false,
        )

        xedges = range(minimum(xi), maximum(xi), length=bins + 1)
        yedges = range(minimum(yj), maximum(yj), length=bins + 1)

        h_sum = fit(Histogram, (xi, yj), weights(losses), (xedges, yedges))
        h_cnt = fit(Histogram, (xi, yj), (xedges, yedges))
        grid = h_sum.weights ./ (h_cnt.weights .+ 1e-9)
        grid[h_cnt.weights .== 0] .= NaN

        hm_last = heatmap!(ax, xedges, yedges, grid, colormap=Reverse(:viridis),
                            colorrange=(color_lo, color_hi))

        best_t = argmin(losses)
        scatter!(ax, [ho.history[best_t][i]], [ho.history[best_t][j]], color=:red, marker=:star5, markersize=14)
    end

    Colorbar(fig[1:nrows, ncols + 1], hm_last,
             label="Mean χ² (color clipped at $(round(Int, 100 * color_percentile))th pct)")

    save("$(out_path).png", fig, px_per_unit=2)
    save("$(out_path).pdf", fig)
    println("Saved pairwise loss-landscape plot to '$(out_path).png' and '.pdf'")
    return out_path
end

# ------------------------------------------------------------------
# 3. Parallel coordinates: every non-penalized trial as one polyline across
#    all parameters (each axis min-max normalized to [0,1] using its own
#    observed range), colored by loss. Optuna's plot_parallel_coordinate,
#    picked because it's the one view here that shows all N dimensions in a
#    single figure rather than 2 at a time.
# ------------------------------------------------------------------
function plot_parallel_coordinates(ho, param_names::Vector{Symbol};
                                     color_percentile::Float64=0.90,
                                     out_path::String="output/parallel_coordinates")
    losses = Float64.(ho.results)
    n_params = length(param_names)
    n_trials = length(losses)

    raw = [[ho.history[t][p] for t in 1:n_trials] for p in 1:n_params]
    mins = [minimum(r) for r in raw]
    maxs = [maximum(r) for r in raw]
    norm = [(raw[p] .- mins[p]) ./ max(maxs[p] - mins[p], 1e-12) for p in 1:n_params]

    # Same percentile-based color clipping as plot_pairwise_heatmaps -- every
    # trial is still drawn, only the color scale saturates past the cutoff.
    lo = minimum(losses)
    hi = quantile(losses, color_percentile)

    fig = Figure(size=(max(750, 70 * n_params), 480), font="DejaVu Sans")
    ax = Axis(fig[1, 1],
        title="Parallel Coordinates (colored by loss)",
        xticks=(1:n_params, string.(param_names)),
        xticklabelrotation=pi / 4,
        ylabel="Normalized value (per-parameter min-max)",
        xgridvisible=false, ygridvisible=false,
    )
    ylims!(ax, -0.05, 1.05)

    for t in 1:n_trials
        y = [norm[p][t] for p in 1:n_params]
        lines!(ax, 1:n_params, y, color=losses[t], colormap=Reverse(:viridis), colorrange=(lo, hi),
               linewidth=1.0, alpha=0.35)
    end

    # Overlay the single best trial in bold
    best_t = argmin(losses)
    y_best = [norm[p][best_t] for p in 1:n_params]
    lines!(ax, 1:n_params, y_best, color=:red, linewidth=2.5)

    Colorbar(fig[1, 2], colormap=Reverse(:viridis), limits=(lo, hi),
             label="χ² (color clipped at $(round(Int, 100 * color_percentile))th pct, lower = better)")

    # Actual (unnormalized) axis ranges, as a text sidebar rather than
    # per-axis annotations -- same pattern as the cuts_text sidebar in
    # da_code.jl / test_higgsml_parquet.jl.
    ranges_text = join(
        ["$(string(param_names[p])): [$(round(mins[p], digits=3)), $(round(maxs[p], digits=3))]" for p in 1:n_params],
        "\n"
    )
    Label(fig[2, 1:2], ranges_text, justification=:left, halign=:left, fontsize=9, font="DejaVu Sans Mono")

    save("$(out_path).png", fig, px_per_unit=2)
    save("$(out_path).pdf", fig)
    println("Saved parallel coordinates plot to '$(out_path).png' and '.pdf'")
    return out_path
end