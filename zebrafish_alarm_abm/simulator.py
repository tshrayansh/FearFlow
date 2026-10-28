"""
Simulator module for the Zebrafish Alarm Contagion ABM.
Executes single stochastic replicates with:
  - Exact common-random-number paired warmup seeding across conditions
  - Dynamic relative chemical cue anchoring at t_on = 20.0s
  - Full Couzin kinematics and dual-channel contagion loop
  - Time-series and replicate summary metric extraction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from .config import (
    SimulationConfig,
    STATE_S,
    STATE_A_CHEM,
    STATE_A_SOC,
    STATE_R
)
from .kinematics import (
    compute_pairwise_geometry,
    compute_couzin_vectors,
    compute_wall_avoidance,
    update_kinematics
)
from .contagion import (
    check_chemical_hazard,
    compute_social_hazard,
    update_states
)
from .metrics import (
    compute_frame_metrics,
    summarize_replicate
)


def run_replicate(
    cfg: SimulationConfig,
    seed: int,
    condition_id: str = "C3"
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[np.ndarray]]:
    """
    Executes a single simulation replicate of duration cfg.t_total.

    Parameters:
      cfg: SimulationConfig containing all parameters
      seed: Master random seed (1000 + replicate_id for warmup pairing)
      condition_id: Identifying label ('C1', 'C2', 'C3', 'C4', etc.)

    Returns:
      df_metrics: DataFrame of frame-by-frame time series
      summary: Dictionary of replicate-level summary outcomes
      cue_center: (2,) array of chemical plume location (or None if no cue)
    """
    rng = np.random.default_rng(seed)
    num_agents = cfg.num_agents
    num_steps = int(round(cfg.t_total / cfg.dt))

    # 1. Initialization (t = 0s)
    # Agents initialized in compact disk of radius init_disk_radius
    r_rand = cfg.init_disk_radius * np.sqrt(rng.uniform(0.0, 1.0, size=num_agents))
    theta_rand = rng.uniform(0.0, 2.0 * np.pi, size=num_agents)
    positions = np.zeros((num_agents, 2), dtype=np.float64)
    positions[:, 0] = cfg.init_center_x + r_rand * np.cos(theta_rand)
    positions[:, 1] = cfg.init_center_y + r_rand * np.sin(theta_rand)

    # Initial headings pointing generally eastward (+x, 0 rad)
    headings = rng.normal(0.0, cfg.init_heading_sd, size=num_agents)

    # Initial states all Susceptible (S)
    states = np.full(num_agents, STATE_S, dtype=np.int32)
    trigger_times = np.full(num_agents, np.nan, dtype=np.float64)

    # Primary trigger tracking (mutually exclusive primary infection channel)
    # 0 = Uninfected, 1 = Direct Chemical (A_chem), 2 = Visual Social (A_soc)
    first_trigger = np.zeros(num_agents, dtype=np.int32)

    cue_center: Optional[np.ndarray] = None
    step_records = []

    # 2. Main Simulation Time Loop
    for step in range(num_steps):
        current_time = step * cfg.dt

        # At t = t_warmup (20.0s), anchor the chemical cue relative to the shoal centroid
        if step == int(round(cfg.t_warmup / cfg.dt)):
            centroid = np.mean(positions, axis=0)
            heading_vecs = np.column_stack([np.cos(headings), np.sin(headings)])
            group_v = np.sum(heading_vecs, axis=0)
            group_speed = np.linalg.norm(group_v)
            if group_speed > 1e-6:
                v_unit = group_v / group_speed
            else:
                v_unit = np.array([1.0, 0.0])

            # Perpendicular vector v_perp = (-v_y, v_x)
            v_perp = np.array([-v_unit[1], v_unit[0]])

            # Offset plume 5.0 BL forward along path and 3.0 BL laterally (flank grazing)
            if cfg.alpha_chem > 0.0:
                raw_cue = centroid + (5.0 * v_unit) + (3.0 * v_perp)
                # Keep plume safely inside arena
                safe_margin = cfg.cue_radius + 1.0
                cue_center = np.clip(raw_cue, safe_margin, cfg.arena_size - safe_margin)
            else:
                cue_center = None

        # Check whether chemical cue is actively releasing substance
        cue_active = (
            (cfg.alpha_chem > 0.0)
            and (cue_center is not None)
            and (cfg.t_warmup <= current_time <= cfg.t_cue_end)
        )

        # Pairwise geometry & visual FOV
        dist, unit_r, in_visual_cone = compute_pairwise_geometry(
            positions, headings, cfg.blind_angle
        )

        # Contagion Hazards & State Updates (active only after warmup)
        if current_time >= cfg.t_warmup:
            chem_hazard = check_chemical_hazard(
                positions, cue_center, cfg.cue_radius, cue_active, cfg.alpha_chem
            )
            soc_hazard = compute_social_hazard(
                dist, in_visual_cone, states, cfg.r_soc, cfg.theta_soc, cfg.beta_soc
            )

            old_states = states.copy()
            states, trigger_times = update_states(
                states=states,
                trigger_times=trigger_times,
                chem_hazard=chem_hazard,
                soc_hazard=soc_hazard,
                gamma_recovery=cfg.gamma_recovery,
                kappa_refractory=cfg.kappa_refractory,
                dt=cfg.dt,
                current_time=current_time,
                rng=rng
            )

            # Record mutually exclusive primary trigger
            new_chem = (old_states == STATE_S) & (states == STATE_A_CHEM) & (first_trigger == 0)
            new_soc = (old_states == STATE_S) & (states == STATE_A_SOC) & (first_trigger == 0)
            first_trigger[new_chem] = 1
            first_trigger[new_soc] = 2

        # Couzin Kinematics: interaction vectors, wall repulsion, and update
        social_vecs = compute_couzin_vectors(
            dist=dist,
            unit_r=unit_r,
            headings=headings,
            in_visual_cone=in_visual_cone,
            states=states,
            cfg=cfg,
            frozen_kinematics=cfg.frozen_kinematics
        )
        wall_vecs = compute_wall_avoidance(
            positions=positions,
            arena_size=cfg.arena_size,
            wall_margin=cfg.wall_margin
        )

        positions, headings = update_kinematics(
            positions=positions,
            headings=headings,
            social_vecs=social_vecs,
            wall_vecs=wall_vecs,
            states=states,
            cfg=cfg,
            rng=rng,
            frozen_kinematics=cfg.frozen_kinematics
        )

        # Frame Metrics
        frame_dict = compute_frame_metrics(
            positions=positions,
            headings=headings,
            states=states,
            r_cluster_threshold=cfg.r_attraction
        )
        frame_dict["time"] = current_time
        frame_dict["step"] = step
        frame_dict["condition"] = condition_id
        frame_dict["seed"] = seed
        step_records.append(frame_dict)

    df_metrics = pd.DataFrame(step_records)
    summary = summarize_replicate(
        df=df_metrics,
        ever_chem_count=int(np.count_nonzero(first_trigger == 1)),
        ever_soc_count=int(np.count_nonzero(first_trigger == 2)),
        num_agents=num_agents,
        cue_start=cfg.t_warmup,
        cue_end=cfg.t_cue_end
    )
    summary["condition"] = condition_id
    summary["seed"] = seed

    return df_metrics, summary, cue_center
