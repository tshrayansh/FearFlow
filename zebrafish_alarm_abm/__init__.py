"""
Zebrafish Alarm Contagion Agent-Based Model (ABM) Package.
Decoupling Direct Chemical Alarm Sensing from Visual Behavioural Contagion.
"""

from .config import (
    SimulationConfig,
    get_condition_config,
    STATE_S,
    STATE_A_CHEM,
    STATE_A_SOC,
    STATE_R
)
from .kinematics import (
    wrap_to_pi,
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
from .simulator import run_replicate

__all__ = [
    "SimulationConfig",
    "get_condition_config",
    "STATE_S",
    "STATE_A_CHEM",
    "STATE_A_SOC",
    "STATE_R",
    "wrap_to_pi",
    "compute_pairwise_geometry",
    "compute_couzin_vectors",
    "compute_wall_avoidance",
    "update_kinematics",
    "check_chemical_hazard",
    "compute_social_hazard",
    "update_states",
    "compute_frame_metrics",
    "summarize_replicate",
    "run_replicate",
]
