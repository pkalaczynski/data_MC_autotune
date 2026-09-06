"""
Data/MC sideband agreement loss.

Direct port of `compute_agreement_loss` from da_code.jl:
  1. Apply the 12D window cuts to both Data and MC.
  2. Reject the cut combo outright if it destroys more than half of the
     82-98 GeV signal peak (peak protection).
  3. Compare Data vs (rate-normalized) MC in the *sidebands* only, via a
     binned chi2 over ENERGY_BINS.

Returns a plain float, exactly like the Julia version, so it plugs into
any optimizer (Optuna, Hyperopt.jl, grid search, ...) as a black-box
objective.
"""

import numpy as np
import pandas as pd

from config import ENERGY_BINS, SIGNAL_MIN, SIGNAL_MAX


def compute_agreement_loss(df_data: pd.DataFrame, df_mc: pd.DataFrame, cuts) -> float:
    (
        chi2_low,
        chi2_high,
        iso_low,
        iso_high,
        hits_low,
        hits_high,
        time_low,
        time_high,
        had_low,
        had_high,
        vtx_low,
        vtx_high,
    ) = cuts

    def window_mask(df):
        return (
            (df.track_chi2 >= chi2_low)
            & (df.track_chi2 <= chi2_high)
            & (df.isolation >= iso_low)
            & (df.isolation <= iso_high)
            & (df.hits_count >= hits_low)
            & (df.hits_count <= hits_high)
            & (df.timing_ns >= time_low)
            & (df.timing_ns <= time_high)
            & (df.hadronic_fraction >= had_low)
            & (df.hadronic_fraction <= had_high)
            & (df.vertex_dist >= vtx_low)
            & (df.vertex_dist <= vtx_high)
        )

    d_sub = df_data[window_mask(df_data)]
    m_sub = df_mc[window_mask(df_mc)]

    # --- Peak protection -------------------------------------------------
    raw_peak_count = (
        (df_data.energy >= SIGNAL_MIN) & (df_data.energy <= SIGNAL_MAX)
    ).sum()
    cut_peak_count = ((d_sub.energy >= SIGNAL_MIN) & (d_sub.energy <= SIGNAL_MAX)).sum()
    peak_survival = cut_peak_count / raw_peak_count

    if peak_survival < 0.50:
        return 1e6 * (0.50 - peak_survival)

    # --- Sideband shape comparison ---------------------------------------
    d_side = d_sub[(d_sub.energy < SIGNAL_MIN) | (d_sub.energy > SIGNAL_MAX)]
    m_side = m_sub[(m_sub.energy < SIGNAL_MIN) | (m_sub.energy > SIGNAL_MAX)]

    if len(d_side) < 100 or len(m_side) < 100:
        return 1e6

    w_data, _ = np.histogram(d_side.energy, bins=ENERGY_BINS)
    w_mc_raw, _ = np.histogram(m_side.energy, bins=ENERGY_BINS)
    w_mc = w_mc_raw * (w_data.sum() / w_mc_raw.sum())

    denom = w_data + w_mc
    valid = denom > 0
    chi2 = np.sum(((w_data[valid] - w_mc[valid]) ** 2) / denom[valid])

    return float(chi2)
