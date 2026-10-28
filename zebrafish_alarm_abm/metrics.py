"""
Metrics module for the Zebrafish Alarm Contagion ABM.
Computes frame-by-frame and aggregate replicate-level outcome statistics:
  - Total Alarm Prevalence P_A(t) and Peak P_A^max
  - Socially Acquired Alarm Fraction F_soc(t) and Peak F_soc^max
  - Cumulative Social Incidence C_soc and Chemical Incidence C_chem
  - Nearest-Neighbour Distance NND(t) and Minimum NND_min (Shoal Tightening)
  - Group Polarisation Phi(t) and Minimum Phi_min (Schooling Alignment)
  - Inter-Individual Distance IID(t) and Cluster Count N_clusters (Shoal Fission diagnostic)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from .config import STATE_S, STATE_A_CHEM, STATE_A_SOC, STATE_R


def compute_frame_metrics(
    positions: np.ndarray,
    headings: np.ndarray,
    states: np.ndarray,
    r_cluster_threshold: float = 10.0
) -> Dict[str, float]:
    """
    Calculates macroscopic collective and epidemiological metrics for a single time step.
    """
    num_agents = positions.shape[0]

    # Epidemiological state counts
    n_s = int(np.count_nonzero(states == STATE_S))
    n_a_chem = int(np.count_nonzero(states == STATE_A_CHEM))
    n_a_soc = int(np.count_nonzero(states == STATE_A_SOC))
    n_r = int(np.count_nonzero(states == STATE_R))
    n_alarm = n_a_chem + n_a_soc

    p_a = n_alarm / float(num_agents)
    f_soc = n_a_soc / float(num_agents)
    f_chem = n_a_chem / float(num_agents)

    # Pairwise distance matrix
    pairwise_dists = squareform(pdist(positions))
    np.fill_diagonal(pairwise_dists, np.inf)

    # Nearest-Neighbour Distance (NND)
    min_dists = np.min(pairwise_dists, axis=1)
    mean_nnd = float(np.mean(min_dists))

    # Inter-Individual Distance (IID): average of all unique pairs
    np.fill_diagonal(pairwise_dists, 0.0)
    mean_iid = float(np.mean(pairwise_dists[np.triu_indices(num_agents, k=1)]))

    # Group Polarisation Phi = (1/N) * || sum_i (cos(theta_i), sin(theta_i)) ||
    heading_vecs = np.column_stack([np.cos(headings), np.sin(headings)])
    mean_heading_vector = np.sum(heading_vecs, axis=0) / float(num_agents)
    polarisation = float(np.linalg.norm(mean_heading_vector))

    # Single-linkage clustering / fragmentation check
    adjacency = (pairwise_dists <= r_cluster_threshold).astype(int)
    n_clusters, _ = connected_components(adjacency, directed=False)

    return {
        "n_s": n_s,
        "n_a_chem": n_a_chem,
        "n_a_soc": n_a_soc,
        "n_r": n_r,
        "p_alarm": p_a,
        "f_soc": f_soc,
        "f_chem": f_chem,
        "nnd": mean_nnd,
        "iid": mean_iid,
        "polarisation": polarisation,
        "n_clusters": int(n_clusters)
    }


def summarize_replicate(
    df: pd.DataFrame,
    ever_chem_count: int,
    ever_soc_count: int,
    num_agents: int = 50,
    cue_start: float = 20.0,
    cue_end: float = 50.0
) -> Dict[str, Any]:
    """
    Computes replicate-level scalar summary statistics used for hypothesis testing.
    Units of analysis are independent replicates, preventing temporal pseudoreplication.
    """
    # Baseline window [10s, 20s]
    baseline_mask = (df["time"] >= 10.0) & (df["time"] < cue_start)
    if not baseline_mask.any():
        # Fallback for short test runs
        baseline_mask = (df["time"] < cue_start)
        if not baseline_mask.any():
            baseline_mask = pd.Series(True, index=df.index)

    baseline_nnd = float(df.loc[baseline_mask, "nnd"].mean())
    baseline_phi = float(df.loc[baseline_mask, "polarisation"].mean())
    baseline_iid = float(df.loc[baseline_mask, "iid"].mean())

    # Acute response window [20s, 35s] - captures sustained collective state shift
    acute_mask = (df["time"] >= cue_start) & (df["time"] <= cue_start + 15.0)
    if not acute_mask.any():
        acute_mask = (df["time"] >= cue_start)
    acute_nnd = float(df.loc[acute_mask, "nnd"].mean())
    acute_phi = float(df.loc[acute_mask, "polarisation"].mean())
    acute_iid = float(df.loc[acute_mask, "iid"].mean())

    # Whole post-stimulus window [20s, 90s]
    post_mask = (df["time"] >= cue_start)
    if not post_mask.any():
        post_mask = pd.Series(True, index=df.index)

    peak_pa = float(df.loc[post_mask, "p_alarm"].max())
    peak_fsoc = float(df.loc[post_mask, "f_soc"].max())
    min_nnd = float(df.loc[post_mask, "nnd"].min())
    min_phi = float(df.loc[post_mask, "polarisation"].min())

    # Cumulative incidence fractions (mutually exclusive primary trigger)
    c_chem = ever_chem_count / float(num_agents)
    c_soc = ever_soc_count / float(num_agents)
    c_total = c_chem + c_soc

    # Time to peak alarm
    if post_mask.any() and df.loc[post_mask, "p_alarm"].max() > 0:
        idx_peak = df.loc[post_mask, "p_alarm"].idxmax()
        t_peak = float(df.loc[idx_peak, "time"])
    else:
        t_peak = np.nan

    return {
        "baseline_nnd": baseline_nnd,
        "baseline_phi": baseline_phi,
        "baseline_iid": baseline_iid,
        "acute_nnd": acute_nnd,
        "acute_phi": acute_phi,
        "acute_iid": acute_iid,
        "peak_pa": peak_pa,
        "peak_fsoc": peak_fsoc,
        "min_nnd": min_nnd,
        "min_phi": min_phi,
        "c_chem": c_chem,
        "c_soc": c_soc,
        "c_total": c_total,
        "t_peak": t_peak,
        "delta_nnd_acute": acute_nnd - baseline_nnd,
        "delta_phi_acute": acute_phi - baseline_phi,
        "delta_iid_acute": acute_iid - baseline_iid,
        "delta_nnd": min_nnd - baseline_nnd,
        "delta_phi": min_phi - baseline_phi
    }
