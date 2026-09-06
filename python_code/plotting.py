"""
Publication-style before/after Data/MC comparison plot.

Matplotlib port of the CairoMakie figure at the bottom of da_code.jl:
main energy spectra (Data vs MC) on top, Data/MC ratio underneath, for
"before tuning" and "after tuning", plus a small text sidebar with the
optimized cut values.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import ENERGY_BINS, CUT_ORDER


def _hist_and_ratio(df_data, df_mc):
    h_data, _ = np.histogram(df_data.energy, bins=ENERGY_BINS)
    h_mc_raw, _ = np.histogram(df_mc.energy, bins=ENERGY_BINS)
    h_mc = h_mc_raw * (h_data.sum() / h_mc_raw.sum())
    ratio = np.where(h_mc > 0, h_data / h_mc, 1.0)
    return h_data, h_mc, ratio


def _apply_cuts(df, cuts_named):
    c = cuts_named
    return df[
        (df.track_chi2 >= c["c_chi2_l"])
        & (df.track_chi2 <= c["c_chi2_h"])
        & (df.isolation >= c["c_iso_l"])
        & (df.isolation <= c["c_iso_h"])
        & (df.hits_count >= c["c_hits_l"])
        & (df.hits_count <= c["c_hits_h"])
        & (df.timing_ns >= c["c_time_l"])
        & (df.timing_ns <= c["c_time_h"])
        & (df.hadronic_fraction >= c["c_had_l"])
        & (df.hadronic_fraction <= c["c_had_h"])
        & (df.vertex_dist >= c["c_vtx_l"])
        & (df.vertex_dist <= c["c_vtx_h"])
    ]


def plot_before_after(
    df_data, df_mc, cuts_named, out_path="energy_distribution_comparison.png"
):
    d_opt = _apply_cuts(df_data, cuts_named)
    m_opt = _apply_cuts(df_mc, cuts_named)

    h_data_before, h_mc_before, ratio_before = _hist_and_ratio(df_data, df_mc)
    h_data_after, h_mc_after, ratio_after = _hist_and_ratio(d_opt, m_opt)

    edges = ENERGY_BINS
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(11.5, 5.5))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 0.95, 0.55],
        height_ratios=[3, 1],
        hspace=0.08,
        wspace=0.35,
    )

    ax1_main = fig.add_subplot(gs[0, 0])
    ax1_ratio = fig.add_subplot(gs[1, 0], sharex=ax1_main)
    ax2_main = fig.add_subplot(gs[0, 1], sharey=ax1_main)
    ax2_ratio = fig.add_subplot(gs[1, 1], sharex=ax2_main, sharey=ax1_ratio)
    ax_side = fig.add_subplot(gs[:, 2])
    ax_side.axis("off")

    def style(ax):
        ax.grid(True, linestyle="--", color="gray", alpha=0.25)

    for ax in (ax1_main, ax1_ratio, ax2_main, ax2_ratio):
        style(ax)

    ax1_main.step(
        centers, h_data_before, where="mid", color="black", lw=2, label="Data"
    )
    ax1_main.step(
        centers, h_mc_before, where="mid", color="red", lw=2, ls="--", label="MC Raw"
    )
    ax1_main.set_title("Before Tuning (Poor Agreement)")
    ax1_main.set_ylabel("Events / 2 GeV")
    ax1_main.tick_params(labelbottom=False)

    ax1_ratio.step(centers, ratio_before, where="mid", color="red", lw=1.5)
    ax1_ratio.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax1_ratio.set_ylim(0.4, 1.6)
    ax1_ratio.set_xlabel("Particle Energy [GeV]")
    ax1_ratio.set_ylabel("Data / MC")

    ax2_main.step(centers, h_data_after, where="mid", color="black", lw=2)
    ax2_main.step(
        centers,
        h_mc_after,
        where="mid",
        color="dodgerblue",
        lw=2,
        ls="--",
        label="MC Tuned",
    )
    ax2_main.set_title("After Quality Window Tuning")
    ax2_main.tick_params(labelbottom=False, labelleft=False)

    ax2_ratio.step(centers, ratio_after, where="mid", color="dodgerblue", lw=1.5)
    ax2_ratio.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax2_ratio.set_xlabel("Particle Energy [GeV]")
    ax2_ratio.tick_params(labelleft=False)

    handles = ax1_main.get_legend_handles_labels()[0] + [
        ax2_main.get_legend_handles_labels()[0][-1]
    ]
    labels = ["Data", "MC Raw", "MC Tuned"]
    ax_side.legend(handles, labels, loc="upper left", frameon=False, fontsize=10)

    c = cuts_named
    cuts_text = (
        "Optimized Boundaries:\n"
        f"chi2: [{c['c_chi2_l']:.1f}, {c['c_chi2_h']:.1f}]\n"
        f"Iso:  [{c['c_iso_l']:.2f}, {c['c_iso_h']:.2f}]\n"
        f"Hits: [{int(c['c_hits_l'])}, {int(c['c_hits_h'])}]\n"
        f"Time: [{c['c_time_l']:.1f}, {c['c_time_h']:.1f}]\n"
        f"Had:  [{c['c_had_l']:.2f}, {c['c_had_h']:.2f}]\n"
        f"Vtx:  [{c['c_vtx_l']:.1f}, {c['c_vtx_h']:.1f}]"
    )
    ax_side.text(
        0.0, 0.55, cuts_text, family="monospace", fontsize=9.5, va="top", ha="left"
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    return out_path
