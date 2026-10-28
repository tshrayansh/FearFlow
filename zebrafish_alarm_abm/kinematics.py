"""
Kinematics module for the Zebrafish Alarm Contagion ABM.
Implements the 2D Couzin metric-zonal collective movement framework:
  - Repulsion (omnidirectional), Orientation, and Attraction zones
  - Rear visual blind angle masking
  - Soft repulsive arena wall avoidance
  - Turn-rate clamped heading and position integration
"""

import numpy as np
from typing import Tuple
from .config import SimulationConfig, STATE_A_CHEM, STATE_A_SOC


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wraps angles in radians to the interval [-pi, pi)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def compute_pairwise_geometry(
    positions: np.ndarray,
    headings: np.ndarray,
    blind_angle: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes pairwise relative displacement vectors, Euclidean distances,
    and visual field-of-view boolean mask for all N agents.

    Parameters:
      positions: (N, 2) array of Cartesian coordinates [BL]
      headings: (N,) array of heading angles [rad]
      blind_angle: rear blind cone angle phi_blind [rad]

    Returns:
      dist: (N, N) pairwise distance matrix
      unit_r: (N, N, 2) unit direction vectors from agent i to agent j
      in_visual_cone: (N, N) boolean mask; True if neighbour j is in agent i's FOV
    """
    num_agents = positions.shape[0]

    # Relative displacement vector r_ij = x_j - x_i
    # dr[i, j, :] = positions[j] - positions[i]
    dr = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(dr, axis=-1)

    # Unit displacement vector from i to j (safe from zero-division on diagonal)
    safe_dist = np.maximum(dist, 1e-9)[:, :, np.newaxis]
    unit_r = dr / safe_dist

    # Agent heading unit vectors v_i = (cos(theta_i), sin(theta_i))
    heading_vecs = np.column_stack([np.cos(headings), np.sin(headings)])

    # Dot product v_i . r_ij to compute bearing angle psi_ij
    # heading_vecs[:, None, :] has shape (N, 1, 2); unit_r has shape (N, N, 2)
    dot = np.sum(heading_vecs[:, np.newaxis, :] * unit_r, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    psi = np.arccos(dot)

    # In visual field if bearing psi <= pi - phi_blind / 2
    max_visible_angle = np.pi - (blind_angle / 2.0)
    in_visual_cone = (psi <= max_visible_angle)

    # Exclude self-interaction
    np.fill_diagonal(in_visual_cone, False)

    return dist, unit_r, in_visual_cone


def compute_couzin_vectors(
    dist: np.ndarray,
    unit_r: np.ndarray,
    headings: np.ndarray,
    in_visual_cone: np.ndarray,
    states: np.ndarray,
    cfg: SimulationConfig,
    frozen_kinematics: bool = False
) -> np.ndarray:
    """
    Evaluates concentric Couzin interaction zones (Repulsion, Orientation, Attraction)
    with biological priority: Repulsion overrides Orientation and Attraction.

    Parameters:
      dist: (N, N) distance matrix
      unit_r: (N, N, 2) unit vectors from i to j
      headings: (N,) heading angles [rad]
      in_visual_cone: (N, N) boolean mask for visual FOV
      states: (N,) discrete behavioural state array
      cfg: SimulationConfig parameter dataclass
      frozen_kinematics: If True, all agents use baseline movement parameters

    Returns:
      desired_social_vecs: (N, 2) desired heading vectors from social interaction
    """
    num_agents = dist.shape[0]
    heading_vecs = np.column_stack([np.cos(headings), np.sin(headings)])

    # Zone masks
    # Repulsion is omnidirectional (lateral line + vision)
    repulsion_mask = (dist <= cfg.r_repulsion)
    np.fill_diagonal(repulsion_mask, False)

    # Orientation and Attraction are gated by visual cone
    orientation_mask = (dist > cfg.r_repulsion) & (dist <= cfg.r_orientation) & in_visual_cone
    attraction_mask = (dist > cfg.r_orientation) & (dist <= cfg.r_attraction) & in_visual_cone

    desired_social_vecs = np.zeros((num_agents, 2), dtype=np.float64)

    for i in range(num_agents):
        rep_neighbors = np.where(repulsion_mask[i])[0]

        if len(rep_neighbors) > 0:
            # Case 1: Repulsion active - strictly prioritize collision avoidance
            # d_soc = - sum_j (r_ij / ||r_ij||)
            desired_social_vecs[i] = -np.sum(unit_r[i, rep_neighbors], axis=0)
        else:
            # Case 2: Repulsion empty - evaluate state-dependent alignment and attraction
            is_alarmed = (states[i] in (STATE_A_CHEM, STATE_A_SOC)) and (not frozen_kinematics)
            w_o = cfg.w_orient_alarm if is_alarmed else cfg.w_orient_baseline
            w_a = cfg.w_attract_alarm if is_alarmed else cfg.w_attract_baseline

            ori_neighbors = np.where(orientation_mask[i])[0]
            att_neighbors = np.where(attraction_mask[i])[0]

            u_o = np.zeros(2, dtype=np.float64)
            if len(ori_neighbors) > 0:
                sum_ori = np.sum(heading_vecs[ori_neighbors], axis=0)
                norm_ori = np.linalg.norm(sum_ori)
                if norm_ori > 1e-9:
                    u_o = sum_ori / norm_ori

            u_a = np.zeros(2, dtype=np.float64)
            if len(att_neighbors) > 0:
                sum_att = np.sum(unit_r[i, att_neighbors], axis=0)
                norm_att = np.linalg.norm(sum_att)
                if norm_att > 1e-9:
                    u_a = sum_att / norm_att

            if len(ori_neighbors) > 0 or len(att_neighbors) > 0:
                desired_social_vecs[i] = (w_o * u_o) + (w_a * u_a)
            else:
                # Maintain current swimming heading when isolated
                desired_social_vecs[i] = heading_vecs[i]

    return desired_social_vecs


def compute_wall_avoidance(
    positions: np.ndarray,
    arena_size: float,
    wall_margin: float
) -> np.ndarray:
    """
    Computes soft boundary repulsion forces directed inward away from tank walls.
    Proportional to penetration depth within wall_margin.
    """
    wall_vecs = np.zeros_like(positions)
    x = positions[:, 0]
    y = positions[:, 1]

    # Left wall (x = 0)
    mask_left = x < wall_margin
    wall_vecs[mask_left, 0] += (wall_margin - x[mask_left]) / wall_margin

    # Right wall (x = arena_size)
    mask_right = x > (arena_size - wall_margin)
    wall_vecs[mask_right, 0] -= (x[mask_right] - (arena_size - wall_margin)) / wall_margin

    # Bottom wall (y = 0)
    mask_bottom = y < wall_margin
    wall_vecs[mask_bottom, 1] += (wall_margin - y[mask_bottom]) / wall_margin

    # Top wall (y = arena_size)
    mask_top = y > (arena_size - wall_margin)
    wall_vecs[mask_top, 1] -= (y[mask_top] - (arena_size - wall_margin)) / wall_margin

    return wall_vecs


def update_kinematics(
    positions: np.ndarray,
    headings: np.ndarray,
    social_vecs: np.ndarray,
    wall_vecs: np.ndarray,
    states: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    frozen_kinematics: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines social vectors, soft wall avoidance, state-dependent Gaussian noise,
    and physiological turn-rate limits to update agent headings and positions.

    Returns:
      new_positions: (N, 2) updated Cartesian coordinates [BL]
      new_headings: (N,) updated heading angles [rad]
    """
    num_agents = positions.shape[0]

    # Normalize social vectors
    social_norm = np.linalg.norm(social_vecs, axis=-1, keepdims=True)
    safe_social_norm = np.maximum(social_norm, 1e-9)
    u_soc = social_vecs / safe_social_norm

    # Combine social vector with soft wall avoidance
    combined_vecs = u_soc + (cfg.wall_weight * wall_vecs)
    target_angles = np.arctan2(combined_vecs[:, 1], combined_vecs[:, 0])

    # Assign state-dependent noise standard deviation, turn rate, and speed
    is_alarmed = np.isin(states, (STATE_A_CHEM, STATE_A_SOC))
    if frozen_kinematics:
        is_alarmed = np.zeros(num_agents, dtype=bool)

    noise_sd = np.where(is_alarmed, cfg.noise_sd_alarm, cfg.noise_sd_baseline)
    omega_max = np.where(is_alarmed, cfg.omega_max_alarm, cfg.omega_max_baseline)
    speed = np.where(is_alarmed, cfg.v_alarm, cfg.v_baseline)

    # Add Gaussian angular noise (erratic zig-zagging in alarm state)
    noise = rng.normal(loc=0.0, scale=noise_sd, size=num_agents)
    desired_angles = target_angles + noise

    # Compute angular difference and clamp by maximum turning rate
    angle_diff = wrap_to_pi(desired_angles - headings)
    max_turn_step = omega_max * cfg.dt
    clamped_turn = np.clip(angle_diff, -max_turn_step, max_turn_step)

    # New headings
    new_headings = wrap_to_pi(headings + clamped_turn)

    # Position integration: x(t + dt) = x(t) + v * [cos(theta), sin(theta)] * dt
    step_vecs = np.column_stack([np.cos(new_headings), np.sin(new_headings)])
    new_positions = positions + (speed[:, np.newaxis] * step_vecs * cfg.dt)

    # Soft boundary safeguard to prevent numerical boundary clipping
    new_positions[:, 0] = np.clip(new_positions[:, 0], 0.1, cfg.arena_size - 0.1)
    new_positions[:, 1] = np.clip(new_positions[:, 1], 0.1, cfg.arena_size - 0.1)

    return new_positions, new_headings
