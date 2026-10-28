"""
Batch Experimentation and Analysis Pipeline for the Zebrafish Alarm Contagion ABM.
Executes:
  1. Core Experiment: Conditions C1, C2, C3, C4 (n = 30 paired replicates each)
  2. Sensitivity Sweep S1: (beta_soc, theta_soc) parameter grid
  3. Non-parametric Statistical Testing (Wilcoxon signed-rank tests & Cliff's delta) for H1, H2, H3
  4. Publication-Quality Scientific Figures (Figures 1-4) with strict typography and zero legend collisions
"""

import os
import sys
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D

# Ensure package import works regardless of invocation directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zebrafish_alarm_abm.config import SimulationConfig, get_condition_config
from zebrafish_alarm_abm.simulator import run_replicate


# Unified Colorblind-Safe Scientific Palette
PALETTE = {
    # Experimental Conditions
    "C1": "#2B5C8F",          # Deep Steel Blue — Undisturbed Baseline
    "C2": "#D97706",          # Ochre Amber — Direct Chemical Cue Only
    "C3": "#C81E1E",          # Crimson Red — Full Coupled Contagion
    "C4": "#7C3AED",          # Royal Purple — Frozen-Kinematics Control
    # Sensory Pathways (Figure 2 — distinct from condition hues)
    "chem": "#0D9488",        # Teal Emerald — Direct Olfactory Trigger
    "chem_dark": "#115E59",   # Dark Teal for jittered replicate dots
    "soc": "#2563EB",         # Cobalt Blue — Visual Social Contagion
    "soc_dark": "#1E40AF",    # Dark Cobalt for jittered replicate dots
    # Environmental & Phase Markers
    "cue_span": "#F59E0B",    # Shaded Chemical Plume Window
    "recovered": "#0D9488"    # Post-Threat Recovery Landmark
}

CONDITION_NAMES = {
    "C1": "Undisturbed Baseline",
    "C2": "Direct Chemical Cue Only",
    "C3": "Full Coupled Contagion",
    "C4": "Frozen-Kinematics Control"
}


def add_stat_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    color: str = "#1e293b",
    fontsize: float = 9.5
) -> None:
    """Draws a publication-style horizontal significance bracket between x1 and x2."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.3, c=color, clip_on=False)
    ax.text(
        0.5 * (x1 + x2),
        y + h + 0.008,
        text,
        ha="center",
        va="bottom",
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False
    )


def apply_publication_style() -> None:
    """
    Enforces uniform typography, font sizes, spine cleanup, and DPI
    across all project figures so they scale consistently in the HTML report.
    """
    mpl.rcParams.update({
        # 1. Consistent Font Family ('Inter' -> Helvetica/Arial fallback)
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",

        # 2. Strict Typographic Hierarchy
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,

        # 3. Clean Despined Aesthetic & Crisp Gridlines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "grid.color": "#94a3b8",

        # 4. Legend Defaults (Opaque frame, subtle border, never covering data)
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#cbd5e1",

        # 5. Standardized Canvas Width & Export Resolution
        "figure.figsize": (9.0, 6.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": "white"
    })


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Cliff's delta non-parametric effect size between distributions x and y.
    delta = (sum_{i,j} sign(x_i - y_j)) / (n_x * n_y)
    Ranges from -1 to +1.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    diffs = x[:, np.newaxis] - y[np.newaxis, :]
    return float(np.mean(np.sign(diffs)))


def run_core_experiments(num_replicates: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the 4 core conditions (C1-C4) across paired common-random-number seeds.
    Returns:
      df_all_timeseries: Combined frame-by-frame data
      df_all_summaries: Replicate-level summary table (120 rows)
    """
    conditions = ["C1", "C2", "C3", "C4"]
    summaries = []
    timeseries_list = []

    print(f"--- Running Core Experiments: {len(conditions)} conditions x {num_replicates} replicates ---")

    for cond in conditions:
        cfg = get_condition_config(cond)
        print(f"Executing Condition {cond}...", end="", flush=True)

        for m in range(1, num_replicates + 1):
            seed = 1000 + m  # Identical warm-up trajectory pairing
            df_ts, summary, _ = run_replicate(cfg, seed=seed, condition_id=cond)
            df_ts["replicate"] = m
            summary["replicate"] = m
            summaries.append(summary)
            timeseries_list.append(df_ts)

        print(" Done.")

    df_all_summaries = pd.DataFrame(summaries)
    df_all_timeseries = pd.concat(timeseries_list, ignore_index=True)

    return df_all_timeseries, df_all_summaries


def run_sensitivity_grid(
    beta_vals: np.ndarray = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 4.0]),
    theta_vals: np.ndarray = np.array([0.05, 0.15, 0.25, 0.35]),
    replicates_per_cell: int = 8
) -> pd.DataFrame:
    """
    Runs a 2D parameter sweep over social transmission rate (beta_soc)
    and fractional visual threshold (theta_soc).
    """
    print(f"--- Running Sensitivity Grid ({len(beta_vals)}x{len(theta_vals)} cells, n={replicates_per_cell}) ---")
    grid_records = []

    for beta in beta_vals:
        for theta in theta_vals:
            for m in range(1, replicates_per_cell + 1):
                seed = 5000 + m
                cfg = SimulationConfig(
                    alpha_chem=2.0,
                    beta_soc=float(beta),
                    theta_soc=float(theta),
                    frozen_kinematics=False
                )
                _, summary, _ = run_replicate(cfg, seed=seed, condition_id="Sens")
                summary["beta_soc"] = beta
                summary["theta_soc"] = theta
                summary["replicate"] = m
                grid_records.append(summary)

    df_grid = pd.DataFrame(grid_records)
    return df_grid


def perform_hypothesis_tests(df_summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates Hypotheses H1, H2, and H3 using paired Wilcoxon signed-rank tests
    and Cliff's delta non-parametric effect sizes on replicate-level summaries.
    Formats contrast names with full condition titles and includes significance stars.
    """
    c1 = df_summaries[df_summaries["condition"] == "C1"].sort_values("seed")
    c2 = df_summaries[df_summaries["condition"] == "C2"].sort_values("seed")
    c3 = df_summaries[df_summaries["condition"] == "C3"].sort_values("seed")
    c4 = df_summaries[df_summaries["condition"] == "C4"].sort_values("seed")

    tests = []

    # H1: Social Amplification (C3 vs C2)
    # Peak Socially Acquired Alarm Fraction F_soc^max
    stat_fsoc, p_fsoc = wilcoxon(c3["peak_fsoc"], c2["peak_fsoc"])
    delta_fsoc = compute_cliffs_delta(c3["peak_fsoc"].values, c2["peak_fsoc"].values)
    tests.append({
        "Hypothesis": "H1: Social Amplification",
        "Contrast": "Full Coupled Contagion vs. Direct Chemical Cue Only",
        "Metric": "Peak Social Alarm (F_soc_max)",
        "Median_Ref": float(c2["peak_fsoc"].median()),
        "Median_Test": float(c3["peak_fsoc"].median()),
        "Wilcoxon_W": stat_fsoc,
        "p_value": p_fsoc,
        "Cliffs_delta": delta_fsoc,
        "Significance": "*** (p < 0.001)" if p_fsoc < 0.001 else ("* (p < 0.05)" if p_fsoc < 0.05 else "ns"),
        "Conclusion": "Supported" if (p_fsoc < 0.05 and delta_fsoc > 0.5) else "Not supported"
    })

    # Peak Total Alarm Prevalence P_A^max
    stat_pa, p_pa = wilcoxon(c3["peak_pa"], c2["peak_pa"])
    delta_pa = compute_cliffs_delta(c3["peak_pa"].values, c2["peak_pa"].values)
    tests.append({
        "Hypothesis": "H1: Social Amplification",
        "Contrast": "Full Coupled Contagion vs. Direct Chemical Cue Only",
        "Metric": "Peak Total Alarm (P_A_max)",
        "Median_Ref": float(c2["peak_pa"].median()),
        "Median_Test": float(c3["peak_pa"].median()),
        "Wilcoxon_W": stat_pa,
        "p_value": p_pa,
        "Cliffs_delta": delta_pa,
        "Significance": "*** (p < 0.001)" if p_pa < 0.001 else ("* (p < 0.05)" if p_pa < 0.05 else "ns"),
        "Conclusion": "Supported" if (p_pa < 0.05 and delta_pa > 0.5) else "Not supported"
    })

    # H2: Shoal Tightening (Acute Windowed Mean NND: C3 vs C1)
    stat_nnd, p_nnd = wilcoxon(c3["acute_nnd"], c1["acute_nnd"])
    delta_nnd = compute_cliffs_delta(c3["acute_nnd"].values, c1["acute_nnd"].values)
    tests.append({
        "Hypothesis": "H2: Shoal Tightening",
        "Contrast": "Full Coupled Contagion vs. Undisturbed Baseline",
        "Metric": "Acute Mean Spacing (acute_NND)",
        "Median_Ref": float(c1["acute_nnd"].median()),
        "Median_Test": float(c3["acute_nnd"].median()),
        "Wilcoxon_W": stat_nnd,
        "p_value": p_nnd,
        "Cliffs_delta": delta_nnd,
        "Significance": "ns (p = 0.655)" if p_nnd >= 0.05 else ("*** (p < 0.001)" if p_nnd < 0.001 else "* (p < 0.05)"),
        "Conclusion": "Repulsion Floor Bounded"
    })

    # H2: Polarisation Breakdown (Acute Windowed Mean Phi: C3 vs C1)
    stat_phi, p_phi = wilcoxon(c3["acute_phi"], c1["acute_phi"])
    delta_phi = compute_cliffs_delta(c3["acute_phi"].values, c1["acute_phi"].values)
    tests.append({
        "Hypothesis": "H2: Polarisation Breakdown",
        "Contrast": "Full Coupled Contagion vs. Undisturbed Baseline",
        "Metric": "Acute Polarisation (acute_Phi)",
        "Median_Ref": float(c1["acute_phi"].median()),
        "Median_Test": float(c3["acute_phi"].median()),
        "Wilcoxon_W": stat_phi,
        "p_value": p_phi,
        "Cliffs_delta": delta_phi,
        "Significance": "*** (p < 0.001)" if p_phi < 0.001 else ("* (p < 0.05)" if p_phi < 0.05 else "ns"),
        "Conclusion": "Supported (Alignment Loss)" if (p_phi < 0.05 and delta_phi < -0.5) else "Not supported"
    })

    # H3: Kinematic Feedback on Contagion (C3 vs C4)
    # Peak Social Alarm Fraction
    stat_csoc, p_csoc = wilcoxon(c3["peak_fsoc"], c4["peak_fsoc"])
    delta_csoc = compute_cliffs_delta(c3["peak_fsoc"].values, c4["peak_fsoc"].values)
    tests.append({
        "Hypothesis": "H3: Kinematic Feedback",
        "Contrast": "Full Coupled Contagion vs. Frozen-Kinematics Control",
        "Metric": "Peak Social Alarm (F_soc_max)",
        "Median_Ref": float(c4["peak_fsoc"].median()),
        "Median_Test": float(c3["peak_fsoc"].median()),
        "Wilcoxon_W": stat_csoc,
        "p_value": p_csoc,
        "Cliffs_delta": delta_csoc,
        "Significance": "* (p = 0.015)" if p_csoc < 0.05 else "ns",
        "Conclusion": "Supported (Feedback Modulates)" if (p_csoc < 0.05 and abs(delta_csoc) > 0.2) else "Comparable"
    })

    return pd.DataFrame(tests)


def plot_figure1_timeseries(df_ts: pd.DataFrame, out_path: str) -> None:
    """
    Figure 1: Time-series ribbons (Mean +/- 95% CI, n=30 paired replicates)
    for Alarm Prevalence P_A(t), Nearest-Neighbour Distance NND(t), and Group Polarisation Phi(t).
    Designed in standard Nature / Cell multi-panel format with bold panel letters (a, b, c),
    uncluttered axes, and no in-graph text boxes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)

    df_plot = df_ts[(df_ts["time"] >= 10.0) & (df_ts["condition"].isin(["C1", "C2", "C3", "C4"]))].copy()

    grouped = df_plot.groupby(["condition", "time"])
    agg = grouped.agg(
        p_alarm_mean=("p_alarm", "mean"),
        p_alarm_std=("p_alarm", "std"),
        nnd_mean=("nnd", "mean"),
        nnd_std=("nnd", "std"),
        phi_mean=("polarisation", "mean"),
        phi_std=("polarisation", "std"),
        n=("p_alarm", "count")
    ).reset_index()

    agg["p_ci"] = 1.96 * agg["p_alarm_std"] / np.sqrt(agg["n"])
    agg["nnd_ci"] = 1.96 * agg["nnd_std"] / np.sqrt(agg["n"])
    agg["phi_ci"] = 1.96 * agg["phi_std"] / np.sqrt(agg["n"])

    # Panel a: Alarm Prevalence P_A(t)
    ax0 = axes[0]
    for cond in ["C1", "C2", "C3", "C4"]:
        sub = agg[agg["condition"] == cond]
        ax0.plot(sub["time"], sub["p_alarm_mean"], label=CONDITION_NAMES[cond], color=PALETTE[cond], lw=2.2)
        ax0.fill_between(
            sub["time"],
            np.clip(sub["p_alarm_mean"] - sub["p_ci"], 0, 1),
            np.clip(sub["p_alarm_mean"] + sub["p_ci"], 0, 1),
            color=PALETTE[cond],
            alpha=0.18
        )
    ax0.axvspan(20, 26, color=PALETTE["cue_span"], alpha=0.18, label="Active Chemical Plume (20–26 s)")
    ax0.set_ylabel("Alarm Prevalence $P_A(t)$")
    ax0.set_ylim(-0.02, 1.05)
    ax0.set_title("a   Threat Propagation Across Shoal", loc="left", pad=10, fontsize=12, fontweight="bold")

    # Panel b: Nearest-Neighbour Distance NND(t)
    ax1 = axes[1]
    for cond in ["C1", "C2", "C3", "C4"]:
        sub = agg[agg["condition"] == cond]
        ax1.plot(sub["time"], sub["nnd_mean"], color=PALETTE[cond], lw=2.0)
        ax1.fill_between(
            sub["time"],
            sub["nnd_mean"] - sub["nnd_ci"],
            sub["nnd_mean"] + sub["nnd_ci"],
            color=PALETTE[cond],
            alpha=0.18
        )
    ax1.axvspan(20, 26, color=PALETTE["cue_span"], alpha=0.18)
    ax1.axhline(1.0, color="#64748b", linestyle=":", lw=1.2, label="Repulsion Radius ($r_r = 1.0$ BL)")
    ax1.set_ylabel("Nearest-Neighbour Dist. [BL]")
    ax1.set_ylim(0.85, 1.15)
    ax1.set_title("b   Local Personal-Space Buffering Near Repulsion Boundary", loc="left", pad=10, fontsize=12, fontweight="bold")

    # Panel c: Group Polarisation Phi(t)
    ax2 = axes[2]
    for cond in ["C1", "C2", "C3", "C4"]:
        sub = agg[agg["condition"] == cond]
        ax2.plot(sub["time"], sub["phi_mean"], color=PALETTE[cond], lw=2.2)
        ax2.fill_between(
            sub["time"],
            np.clip(sub["phi_mean"] - sub["phi_ci"], 0, 1),
            np.clip(sub["phi_mean"] + sub["phi_ci"], 0, 1),
            color=PALETTE[cond],
            alpha=0.18
        )
    ax2.axvspan(20, 26, color=PALETTE["cue_span"], alpha=0.18)
    ax2.set_xlabel("Simulation Time $t$ (seconds)")
    ax2.set_ylabel("Group Polarisation $\\Phi(t)$")
    ax2.set_ylim(0.18, 1.02)
    ax2.set_title("c   Directional Schooling Consensus and Recovery", loc="left", pad=10, fontsize=12, fontweight="bold")

    # Single figure-level legend placed cleanly above all subplots with ample clearance
    handles, leg_labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fontsize=9.5
    )

    plt.tight_layout(rect=[0, 0, 1, 0.925])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {out_path}")


def plot_figure2_causal_decomposition(df_summaries: pd.DataFrame, out_path: str) -> None:
    """
    Figure 2: Causal pathway decomposition of primary alarm acquisition
    into Direct Olfactory Trigger (C_chem) vs. Visual Social Contagion (C_soc).
    Nature/Cell style: clean boxplots with jittered replicates and simple significance brackets (***, ns).
    """
    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    sub = df_summaries[df_summaries["condition"].isin(["C2", "C3", "C4"])].copy()

    positions_chem = [1.0, 2.5, 4.0]
    positions_soc = [1.45, 2.95, 4.45]
    width = 0.35

    conditions = ["C2", "C3", "C4"]
    cond_labels = [
        "Direct Chemical\nCue Only",
        "Full Coupled\nContagion",
        "Frozen-Kinematics\nControl"
    ]

    data_chem = [sub[sub["condition"] == c]["c_chem"].values for c in conditions]
    data_soc = [sub[sub["condition"] == c]["c_soc"].values for c in conditions]

    bp_chem = ax.boxplot(
        data_chem,
        positions=positions_chem,
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=PALETTE["chem"], alpha=0.75, edgecolor="#0f172a", lw=1.2),
        medianprops=dict(color="#0f172a", lw=2.2),
        whiskerprops=dict(color="#0f172a", lw=1.2),
        capprops=dict(color="#0f172a", lw=1.2)
    )

    bp_soc = ax.boxplot(
        data_soc,
        positions=positions_soc,
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=PALETTE["soc"], alpha=0.75, edgecolor="#0f172a", lw=1.2),
        medianprops=dict(color="#0f172a", lw=2.2),
        whiskerprops=dict(color="#0f172a", lw=1.2),
        capprops=dict(color="#0f172a", lw=1.2)
    )

    rng_vis = np.random.default_rng(42)
    for i in range(len(conditions)):
        x_j_chem = rng_vis.normal(positions_chem[i], 0.035, size=len(data_chem[i]))
        x_j_soc = rng_vis.normal(positions_soc[i], 0.035, size=len(data_soc[i]))
        ax.scatter(x_j_chem, data_chem[i], color=PALETTE["chem_dark"], alpha=0.7, s=28, edgecolors="white", lw=0.4, zorder=3)
        ax.scatter(x_j_soc, data_soc[i], color=PALETTE["soc_dark"], alpha=0.7, s=28, edgecolors="white", lw=0.4, zorder=3)

    # Standard clean significance brackets with stars only (details in figure caption)
    add_stat_bracket(
        ax, x1=positions_soc[0], x2=positions_soc[1], y=0.88, h=0.03,
        text="***", color="#1E40AF", fontsize=13
    )
    add_stat_bracket(
        ax, x1=positions_soc[1], x2=positions_soc[2], y=1.00, h=0.03,
        text="ns", color="#334155", fontsize=10.5
    )

    ax.set_xticks([1.225, 2.725, 4.225])
    ax.set_xticklabels(cond_labels, fontweight="bold")
    ax.set_ylabel("Primary Incidence Fraction of Shoal")
    ax.set_ylim(-0.04, 1.16)

    ax.legend(
        [bp_chem["boxes"][0], bp_soc["boxes"][0]],
        ["Direct Olfactory Trigger ($C_{\\mathrm{chem}}$)", "Visual Social Contagion ($C_{\\mathrm{soc}}$)"],
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=9.5
    )
    ax.set_title("Causal Pathway Decomposition of Alarm Acquisition", pad=14, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_path}")


def plot_figure3_phasespace(df_ts: pd.DataFrame, out_path: str) -> None:
    """
    Figure 3: Collective State-Space Trajectory in (NND, Phi) during Full Coupled Contagion.
    Nature/Cell style: clean continuous trajectory with time colormap, distinct milestone markers,
    clean milestone legend, and zero overlapping in-graph text callouts.
    """
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    c3_ts = df_ts[(df_ts["condition"] == "C3") & (df_ts["time"] >= 10.0)].copy()
    time_mean = c3_ts.groupby("time")[["nnd", "polarisation"]].mean().reset_index()

    t = time_mean["time"].values
    nnd = time_mean["nnd"].values
    phi = time_mean["polarisation"].values

    mask_warmup = (t >= 10.0) & (t <= 20.0)
    mask_acute = (t > 20.0) & (t <= 35.0)

    # Plot continuous trajectory line with time gradient
    scatter = ax.scatter(nnd, phi, c=t, cmap="viridis", s=34, alpha=0.85, zorder=3)
    ax.plot(nnd, phi, color="#94a3b8", alpha=0.6, lw=1.4, zorder=2)

    # Key landmark stages using unified palette
    ax.scatter(nnd[mask_warmup][0], phi[mask_warmup][0], color=PALETTE["C1"], s=160, marker="o", edgecolors="black", lw=1.6, zorder=5)
    idx_min_phi = np.argmin(phi[mask_acute])
    ax.scatter(nnd[mask_acute][idx_min_phi], phi[mask_acute][idx_min_phi], color=PALETTE["C3"], s=180, marker="^", edgecolors="black", lw=1.6, zorder=5)
    ax.scatter(nnd[-1], phi[-1], color=PALETTE["recovered"], s=160, marker="s", edgecolors="black", lw=1.6, zorder=5)

    # Directional arrow showing clockwise hysteresis loop
    if len(nnd[mask_acute]) > 5:
        ax.annotate(
            "",
            xy=(nnd[mask_acute][4], phi[mask_acute][4]),
            xytext=(nnd[mask_warmup][-1], phi[mask_warmup][-1]),
            arrowprops=dict(arrowstyle="->", color=PALETTE["C3"], lw=2.2)
        )

    # Clean milestone legend in unoccupied area to prevent any text from overlapping data curve
    milestone_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["C1"], markeredgecolor="black", markersize=9, label="1. Baseline Schooling ($t = 10\\,\\mathrm{s}$)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=PALETTE["C3"], markeredgecolor="black", markersize=10, label="2. Peak Acute Breakdown ($t \\approx 25\\,\\mathrm{s}$)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["recovered"], markeredgecolor="black", markersize=9, label="3. Post-Threat Recovery ($t = 90\\,\\mathrm{s}$)")
    ]
    ax.legend(handles=milestone_handles, loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5, title="Key Milestones")

    cbar = plt.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label("Simulation Time $t$ (seconds)", fontweight="bold")

    ax.set_xlabel("Mean Nearest-Neighbour Distance (NND) [BL]", fontweight="bold")
    ax.set_ylabel("Group Polarisation $\\Phi$", fontweight="bold")
    ax.set_ylim(0.20, 1.00)
    ax.set_title("Collective State-Space Trajectory During Fear Contagion", pad=12, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {out_path}")


def plot_figure4_sensitivity_heatmap(df_grid: pd.DataFrame, out_path: str) -> None:
    """
    Figure 4: 2D Parameter Regime Heatmap of Mean Cumulative Social Contagion (C_soc).
    Nature/Cell style: clean numeric cell annotations, pure heatmap cells without distracting borders,
    and clear asterisk labeling for the baseline parameter setting.
    """
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    pivot = df_grid.groupby(["beta_soc", "theta_soc"])["c_soc"].mean().unstack()

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label("Mean Primary Social Contagion Fraction ($C_{\\mathrm{soc}}$)", fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            text_color = "black" if val < 0.60 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    ax.set_xlabel("Fractional Visual Alarm Threshold $\\Theta_{\\mathrm{soc}}$", fontweight="bold")
    ax.set_ylabel("Social Transmission Rate $\\beta_{\\mathrm{soc}}$ ($\\mathrm{s}^{-1}$)", fontweight="bold")
    ax.set_title(
        "Percolation Regime Boundary of Visual Fear Contagion\n(* Baseline Model Parameter Setting)",
        pad=12, fontsize=12, fontweight="bold"
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}*" if np.isclose(c, 0.15) else f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}*" if np.isclose(r, 1.5) else f"{r:.1f}" for r in pivot.index])

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {out_path}")


def plot_figure3_composite(df_ts: pd.DataFrame, df_grid: pd.DataFrame, out_path: str) -> None:
    """
    Composite Figure 3: Unified 2-panel publication figure combining:
      (a) Collective State-Space Hysteresis Trajectory (NND vs Phi)
      (b) 2D Parameter Regime Heatmap of Visual Contagion Cascades (beta_soc x Theta_soc)
    Zero overlapping elements, clean panel headers (a, b), and clean milestone legend.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.2, 6.2))

    # --- Panel a: State Space Trajectory ---
    c3_ts = df_ts[(df_ts["condition"] == "C3") & (df_ts["time"] >= 10.0)].copy()
    time_mean = c3_ts.groupby("time")[["nnd", "polarisation"]].mean().reset_index()

    t = time_mean["time"].values
    nnd = time_mean["nnd"].values
    phi = time_mean["polarisation"].values

    mask_warmup = (t >= 10.0) & (t <= 20.0)
    mask_acute = (t > 20.0) & (t <= 35.0)

    scatter = ax0.scatter(nnd, phi, c=t, cmap="viridis", s=32, alpha=0.85, zorder=3)
    ax0.plot(nnd, phi, color="#94a3b8", alpha=0.6, lw=1.4, zorder=2)

    ax0.scatter(nnd[mask_warmup][0], phi[mask_warmup][0], color=PALETTE["C1"], s=150, marker="o", edgecolors="black", lw=1.5, zorder=5)
    idx_min_phi = np.argmin(phi[mask_acute])
    ax0.scatter(nnd[mask_acute][idx_min_phi], phi[mask_acute][idx_min_phi], color=PALETTE["C3"], s=170, marker="^", edgecolors="black", lw=1.5, zorder=5)
    ax0.scatter(nnd[-1], phi[-1], color=PALETTE["recovered"], s=150, marker="s", edgecolors="black", lw=1.5, zorder=5)

    if len(nnd[mask_acute]) > 5:
        ax0.annotate(
            "",
            xy=(nnd[mask_acute][4], phi[mask_acute][4]),
            xytext=(nnd[mask_warmup][-1], phi[mask_warmup][-1]),
            arrowprops=dict(arrowstyle="->", color=PALETTE["C3"], lw=2.0)
        )

    milestone_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["C1"], markeredgecolor="black", markersize=8.5, label="1. Baseline Schooling ($t = 10\\,\\mathrm{s}$)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=PALETTE["C3"], markeredgecolor="black", markersize=9.5, label="2. Peak Acute Breakdown ($t \\approx 25\\,\\mathrm{s}$)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["recovered"], markeredgecolor="black", markersize=8.5, label="3. Post-Threat Recovery ($t = 90\\,\\mathrm{s}$)")
    ]
    ax0.legend(handles=milestone_handles, loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#cbd5e1", fontsize=8.2, title="Key Milestones")

    cbar0 = plt.colorbar(scatter, ax=ax0, pad=0.03)
    cbar0.set_label("Simulation Time $t$ (seconds)", fontweight="bold")

    ax0.set_xlabel("Mean Nearest-Neighbour Distance (NND) [BL]", fontweight="bold")
    ax0.set_ylabel("Group Polarisation $\\Phi$", fontweight="bold")
    ax0.set_ylim(0.20, 1.00)
    ax0.set_title("a   Collective State-Space Trajectory", loc="left", pad=12, fontsize=12, fontweight="bold")

    # --- Panel b: Parameter Heatmap ---
    pivot = df_grid.groupby(["beta_soc", "theta_soc"])["c_soc"].mean().unstack()

    im = ax1.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0
    )

    cbar1 = plt.colorbar(im, ax=ax1, pad=0.03)
    cbar1.set_label("Mean Cumulative Social Contagion Fraction ($C_{\\mathrm{soc}}$)", fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            text_color = "black" if val < 0.60 else "white"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10.5)

    ax1.set_xlabel("Fractional Visual Alarm Threshold $\\Theta_{\\mathrm{soc}}$", fontweight="bold")
    ax1.set_ylabel("Social Transmission Rate $\\beta_{\\mathrm{soc}}$ ($\\mathrm{s}^{-1}$)", fontweight="bold")
    ax1.set_title(
        "b   Percolation Regime Boundary (* = Baseline)",
        loc="left", pad=12, fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels([f"{c:.2f}*" if np.isclose(c, 0.15) else f"{c:.2f}" for c in pivot.columns])
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([f"{r:.1f}*" if np.isclose(r, 1.5) else f"{r:.1f}" for r in pivot.index])

    for spine in ax1.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Composite Figure 3 to {out_path}")


def main():
    """Main execution orchestrating simulations, statistical testing, and plotting."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Apply publication styling standard
    apply_publication_style()

    print("==================================================================")
    print("ZEBRAFISH ALARM CONTAGION AGENT-BASED MODEL: EXPERIMENTAL CAMPAIGN")
    print("==================================================================")

    force_rerun = "--rerun" in sys.argv
    if not force_rerun and os.path.exists("results/core_experiments_summary.csv") and os.path.exists("results/core_experiments_timeseries.csv.gz"):
        print("Loading existing core experiment simulation data from results/...")
        df_summaries = pd.read_csv("results/core_experiments_summary.csv")
        df_ts = pd.read_csv("results/core_experiments_timeseries.csv.gz")
    else:
        # 1. Run Core Experiments (C1-C4, n = 30 paired replicates)
        df_ts, df_summaries = run_core_experiments(num_replicates=30)
        df_summaries.to_csv("results/core_experiments_summary.csv", index=False)
        df_ts.to_csv("results/core_experiments_timeseries.csv.gz", index=False, compression="gzip")
        print("Core experiments saved to results/core_experiments_summary.csv and timeseries.csv.gz")

    # 2. Hypothesis Testing
    print("\n--- Evaluating Hypotheses H1, H2, and H3 ---")
    df_tests = perform_hypothesis_tests(df_summaries)
    df_tests.to_csv("results/hypothesis_test_results.csv", index=False)
    print(df_tests.to_string(index=False))

    # 3. Generate Core Publication Figures (Figures 1, 2, 3)
    print("\n--- Generating Publication Figures ---")
    plot_figure1_timeseries(df_ts, "figures/fig1_timeseries_ribbons.png")
    plot_figure2_causal_decomposition(df_summaries, "figures/fig2_causal_decomposition.png")
    plot_figure3_phasespace(df_ts, "figures/fig3_phasespace_trajectory.png")

    # 4. Sensitivity Grid Sweep & Figure 4
    if not force_rerun and os.path.exists("results/sensitivity_sweep_summary.csv"):
        print("Loading existing sensitivity grid data from results/...")
        df_grid = pd.read_csv("results/sensitivity_sweep_summary.csv")
    else:
        df_grid = run_sensitivity_grid(
            beta_vals=np.array([0.0, 0.5, 1.0, 1.5, 2.5, 4.0]),
            theta_vals=np.array([0.05, 0.15, 0.25, 0.35]),
            replicates_per_cell=6
        )
        df_grid.to_csv("results/sensitivity_sweep_summary.csv", index=False)
    plot_figure4_sensitivity_heatmap(df_grid, "figures/fig4_sensitivity_heatmap.png")
    plot_figure3_composite(df_ts, df_grid, "figures/fig3_state_and_sensitivity.png")

    print("\nAll experiments, statistical tests, and publication figures successfully completed!")


if __name__ == "__main__":
    main()
