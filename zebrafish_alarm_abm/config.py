"""
Configuration module for the Zebrafish Alarm Contagion Agent-Based Model (ABM).
Defines all parameters, physical constants, Couzin zones, behavioural states,
and experimental condition overrides following the scientific plan.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


# Discrete Behavioural States
STATE_S = 0        # Susceptible / Baseline
STATE_A_CHEM = 1   # Alarmed via Direct Chemical Cue
STATE_A_SOC = 2    # Alarmed via Visual Social Contagion
STATE_R = 3        # Recovering / Refractory


@dataclass
class SimulationConfig:
    """Complete parameter configuration for the 2D zebrafish alarm ABM."""

    # Arena & Simulation Time
    num_agents: int = 50                 # Shoal size N
    arena_size: float = 40.0             # Arena side length L (BL)
    dt: float = 0.05                     # Time step (seconds, 20 Hz update)
    t_total: float = 90.0                # Total simulation duration (seconds, 1800 steps)
    t_warmup: float = 20.0               # Warm-up duration before cue activation (s)
    t_cue_end: float = 26.0              # Cue deactivation timestamp (s, 6s acute pulse)

    # Initial Placement
    init_disk_radius: float = 4.0        # Initial compact shoal disk radius (BL)
    init_center_x: float = 12.0          # Initial shoal centroid X (BL)
    init_center_y: float = 20.0          # Initial shoal centroid Y (BL)
    init_heading_sd: float = 0.2         # Heading noise around eastwards (+x) heading (rad)

    # Metric Interaction Zones & Visual FOV (Couzin et al., 2002)
    r_repulsion: float = 1.0             # Zone of Repulsion r_r (BL, omnidirectional)
    r_orientation: float = 4.0           # Zone of Orientation r_o (BL, visual FOV)
    r_attraction: float = 10.0           # Zone of Attraction r_a (BL, visual FOV)
    blind_angle: float = 1.0471975512    # Rear blind angle phi_blind = 60 deg = pi/3 rad
    wall_margin: float = 3.0             # Soft wall repulsion margin r_wall (BL)
    wall_weight: float = 2.0             # Wall repulsion vector weight w_wall

    # Baseline Kinematic Parameters (State S and State R)
    v_baseline: float = 3.0              # Cruising speed v_S (BL/s)
    w_orient_baseline: float = 1.0       # Orientation weight w_{o,S}
    w_attract_baseline: float = 0.35     # Attraction weight w_{a,S} (prevents repulsion floor collapse)
    noise_sd_baseline: float = 0.08      # Angular noise SD sigma_S (rad per step)
    omega_max_baseline: float = 4.0      # Max turning rate (rad/s)

    # Alarmed Kinematic Parameters (State A_chem and State A_soc)
    v_alarm: float = 4.5                 # Darting speed v_A (BL/s, 1.5x baseline)
    w_orient_alarm: float = 0.25         # Orientation weight w_{o,A} (reduced alignment)
    w_attract_alarm: float = 2.5         # Attraction weight w_{a,A} (>7x attraction pull, shoal tightening)
    noise_sd_alarm: float = 0.45         # Angular noise SD sigma_A (rad per step, zig-zag)
    omega_max_alarm: float = 6.0         # Max turning rate (rad/s)

    # Chemical Cue Subsystem
    cue_radius: float = 4.0              # Circular cue zone radius R_cue (BL)
    alpha_chem: float = 2.0              # Direct chemical hazard rate (s^-1)

    # Social Contagion Subsystem (Rosenthal et al., 2015)
    r_soc: float = 5.0                   # Local social perception radius (BL)
    theta_soc: float = 0.15              # Fractional visual alarm threshold Theta_soc
    beta_soc: float = 1.5                # Social transmission rate (s^-1)

    # Behavioural Recovery & Refractory Return (Kermack-McKendrick style)
    gamma_recovery: float = 0.15         # Alarm recovery rate A -> R (s^-1, mean bout ~6.7s)
    kappa_refractory: float = 0.04       # Refractory return rate R -> S (s^-1, mean ~25s, prevents loop)

    # Kinematic Coupling Toggle (Condition C4 control)
    frozen_kinematics: bool = False      # If True, alarmed agents keep baseline movement


def get_condition_config(condition_name: str, **overrides: Any) -> SimulationConfig:
    """
    Factory creating SimulationConfig customized for the standard experimental conditions:
      C1: Undisturbed Baseline (alpha_chem=0, beta_soc=0)
      C2: Direct Cue-Only Control (alpha_chem=2.0, beta_soc=0, full kinematics)
      C3: Full Coupled Contagion (alpha_chem=2.0, beta_soc=1.5, full kinematics)
      C4: Frozen-Kinematics Control (alpha_chem=2.0, beta_soc=1.5, frozen_kinematics=True)
    """
    cfg = SimulationConfig()

    if condition_name == "C1":
        cfg.alpha_chem = 0.0
        cfg.beta_soc = 0.0
        cfg.frozen_kinematics = False
    elif condition_name == "C2":
        cfg.alpha_chem = 2.0
        cfg.beta_soc = 0.0
        cfg.frozen_kinematics = False
    elif condition_name == "C3":
        cfg.alpha_chem = 2.0
        cfg.beta_soc = 1.5
        cfg.frozen_kinematics = False
    elif condition_name == "C4":
        cfg.alpha_chem = 2.0
        cfg.beta_soc = 1.5
        cfg.frozen_kinematics = True
    else:
        # Default or custom condition
        pass

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise AttributeError(f"Unknown configuration parameter: {key}")

    return cfg
