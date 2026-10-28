"""
Contagion module for the Zebrafish Alarm Contagion ABM.
Implements:
  - Localized chemical cue detection (direct olfactory hazard)
  - Fractional visual social contagion (Rosenthal et al., 2015)
  - Competing-risk epidemiological state machine:
      S -> A_chem (direct) vs. S -> A_soc (social) -> R -> S
"""

import numpy as np
from typing import Tuple, Optional
from .config import STATE_S, STATE_A_CHEM, STATE_A_SOC, STATE_R


def check_chemical_hazard(
    positions: np.ndarray,
    cue_center: Optional[np.ndarray],
    cue_radius: float,
    cue_active: bool,
    alpha_chem: float
) -> np.ndarray:
    """
    Computes direct chemical hazard rate lambda_chem for each agent.
    If agent is within cue_radius of cue_center during active window,
    hazard is alpha_chem; otherwise 0.0.
    """
    num_agents = positions.shape[0]
    hazard = np.zeros(num_agents, dtype=np.float64)

    if not cue_active or cue_center is None or alpha_chem <= 0.0:
        return hazard

    dist_to_cue = np.linalg.norm(positions - cue_center, axis=1)
    in_plume = dist_to_cue <= cue_radius
    hazard[in_plume] = alpha_chem

    return hazard


def compute_social_hazard(
    dist: np.ndarray,
    in_visual_cone: np.ndarray,
    states: np.ndarray,
    r_soc: float,
    theta_soc: float,
    beta_soc: float
) -> np.ndarray:
    """
    Computes visual social contagion hazard rate lambda_soc for each agent.
    Evaluates the fraction f_{A,i} of visible neighbours within r_soc in an Alarmed state.
    lambda_{soc,i} = beta_soc * max(0, f_{A,i} - theta_soc).
    """
    num_agents = dist.shape[0]
    hazard = np.zeros(num_agents, dtype=np.float64)

    if beta_soc <= 0.0:
        return hazard

    # Visible neighbours within social perception distance r_soc
    vis_neighbors = (dist <= r_soc) & in_visual_cone
    np.fill_diagonal(vis_neighbors, False)

    # Boolean array indicating which agents are currently alarmed (chem or soc)
    is_alarmed = np.isin(states, (STATE_A_CHEM, STATE_A_SOC))

    for i in range(num_agents):
        neighbors_i = np.where(vis_neighbors[i])[0]
        num_vis = len(neighbors_i)

        if num_vis > 0:
            num_alarmed_vis = np.count_nonzero(is_alarmed[neighbors_i])
            f_alarmed = num_alarmed_vis / float(num_vis)

            # Threshold-linear contagion response
            if f_alarmed > theta_soc:
                hazard[i] = beta_soc * (f_alarmed - theta_soc)

    return hazard


def update_states(
    states: np.ndarray,
    trigger_times: np.ndarray,
    chem_hazard: np.ndarray,
    soc_hazard: np.ndarray,
    gamma_recovery: float,
    kappa_refractory: float,
    dt: float,
    current_time: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the discrete behavioural states according to the competing-risk framework:
      1. S -> A_chem or A_soc based on relative hazard proportions
      2. A -> R with recovery rate gamma
      3. R -> S with refractory return rate kappa

    Returns:
      updated_states: (N,) new discrete state array
      trigger_times: (N,) timestamps of first transition into alarm
    """
    num_agents = states.shape[0]
    new_states = states.copy()

    # Uniform random draws for transitions
    u1 = rng.uniform(0.0, 1.0, size=num_agents)
    u2 = rng.uniform(0.0, 1.0, size=num_agents)

    # 1. Transitions out of Susceptible (S)
    s_mask = (states == STATE_S)
    total_hazard = chem_hazard + soc_hazard

    # Probability of leaving S during dt: P = 1 - exp(-lambda_tot * dt)
    p_leave_s = 1.0 - np.exp(-total_hazard * dt)
    trans_s = s_mask & (total_hazard > 0.0) & (u1 < p_leave_s)

    # Unbiased causal attribution via competing hazard ratio
    # chem_ratio = lambda_chem / lambda_tot
    chem_ratio = np.zeros(num_agents, dtype=np.float64)
    valid_hazard = trans_s & (total_hazard > 0.0)
    chem_ratio[valid_hazard] = chem_hazard[valid_hazard] / total_hazard[valid_hazard]

    trans_to_chem = trans_s & (u2 < chem_ratio)
    trans_to_soc = trans_s & (~trans_to_chem)

    new_states[trans_to_chem] = STATE_A_CHEM
    new_states[trans_to_soc] = STATE_A_SOC

    # Record first alarm onset time if not already recorded
    alarm_onset = trans_s & np.isnan(trigger_times)
    trigger_times[alarm_onset] = current_time

    # 2. Recovery transitions from Alarmed (A_chem, A_soc) to Refractory (R)
    a_mask = np.isin(states, (STATE_A_CHEM, STATE_A_SOC))
    p_recover = 1.0 - np.exp(-gamma_recovery * dt)
    trans_to_r = a_mask & (u1 < p_recover)
    new_states[trans_to_r] = STATE_R

    # 3. Normalization transitions from Refractory (R) back to Susceptible (S)
    r_mask = (states == STATE_R)
    p_return_s = 1.0 - np.exp(-kappa_refractory * dt)
    trans_to_s = r_mask & (u1 < p_return_s)
    new_states[trans_to_s] = STATE_S

    return new_states, trigger_times
