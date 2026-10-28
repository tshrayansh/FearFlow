"""
Deterministic Unit Tests for the Zebrafish Alarm Contagion ABM.
Validates:
  1. Angle wrapping and geometry calculations
  2. Couzin repulsion, orientation, attraction, and blind angle masking
  3. Chemical plume detection and fractional social contagion thresholding
  4. Competing-risk state transitions and causal attribution
  5. Simulation pipeline integrity
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest

from zebrafish_alarm_abm.config import (
    SimulationConfig,
    STATE_S,
    STATE_A_CHEM,
    STATE_A_SOC,
    STATE_R
)
from zebrafish_alarm_abm.kinematics import (
    wrap_to_pi,
    compute_pairwise_geometry,
    compute_couzin_vectors,
    compute_wall_avoidance,
    update_kinematics
)
from zebrafish_alarm_abm.contagion import (
    check_chemical_hazard,
    compute_social_hazard,
    update_states
)
from zebrafish_alarm_abm.metrics import (
    compute_frame_metrics,
    summarize_replicate
)
from zebrafish_alarm_abm.simulator import run_replicate


def test_angle_wrapping():
    """Verify wrap_to_pi maps arbitrary angles strictly to [-pi, pi)."""
    angles = np.array([0.0, np.pi, -np.pi, 3.0 * np.pi, -3.0 * np.pi, 2.0 * np.pi])
    wrapped = wrap_to_pi(angles)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert np.isclose(wrapped[0], 0.0)
    assert np.isclose(wrapped[3], -np.pi)  # 3pi wraps to -pi


def test_visual_blind_angle():
    """
    Agent 0 is at (10, 10) heading East (0 rad).
    Neighbour 1 is at (12, 10) directly ahead (East, bearing 0).
    Neighbour 2 is at (8, 10) directly behind (West, bearing pi).
    Rear blind cone phi_blind = pi/3 (60 deg).
    Neighbour 1 should be visible; Neighbour 2 must be in blind cone.
    """
    positions = np.array([[10.0, 10.0], [12.0, 10.0], [8.0, 10.0]])
    headings = np.array([0.0, 0.0, 0.0])
    blind_angle = np.pi / 3.0  # 60 degrees

    dist, unit_r, in_visual_cone = compute_pairwise_geometry(positions, headings, blind_angle)

    # Neighbour 1 is visible to Agent 0
    assert bool(in_visual_cone[0, 1]) is True
    # Neighbour 2 is in blind cone of Agent 0
    assert bool(in_visual_cone[0, 2]) is False
    # No self-interaction
    assert bool(in_visual_cone[0, 0]) is False


def test_couzin_repulsion_priority():
    """
    Agent 0 is at (10, 10), Agent 1 is at (10.5, 10) [dist = 0.5 < r_r = 1.0].
    Repulsion must strictly repel Agent 0 to the left (negative x direction).
    """
    cfg = SimulationConfig()
    positions = np.array([[10.0, 10.0], [10.5, 10.0]])
    headings = np.array([0.0, 0.0])
    states = np.array([STATE_S, STATE_S])

    dist, unit_r, in_visual_cone = compute_pairwise_geometry(positions, headings, cfg.blind_angle)
    social_vecs = compute_couzin_vectors(dist, unit_r, headings, in_visual_cone, states, cfg)

    # Agent 0 repulsion vector points in -x direction
    assert social_vecs[0, 0] < 0.0
    assert np.isclose(social_vecs[0, 1], 0.0)


def test_chemical_hazard_detection():
    """Agent inside plume gets alpha_chem; agent outside gets 0.0."""
    positions = np.array([[20.0, 20.0], [30.0, 30.0]])
    cue_center = np.array([20.0, 20.0])
    cue_radius = 4.0
    alpha_chem = 2.0

    # Active window
    hazard_active = check_chemical_hazard(positions, cue_center, cue_radius, True, alpha_chem)
    assert np.isclose(hazard_active[0], 2.0)
    assert np.isclose(hazard_active[1], 0.0)

    # Inactive window
    hazard_inactive = check_chemical_hazard(positions, cue_center, cue_radius, False, alpha_chem)
    assert np.all(hazard_inactive == 0.0)


def test_social_hazard_threshold():
    """
    Fractional threshold Theta_soc = 0.15.
    If 0 out of 4 visible neighbours are alarmed -> hazard = 0.0.
    If 2 out of 4 visible neighbours are alarmed (f = 0.50) -> hazard = beta_soc * (0.50 - 0.15).
    """
    # 5 agents arranged in a line within distance r_soc
    dist = np.array([
        [0.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 0.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 0.0],
    ])
    in_visual_cone = np.ones((5, 5), dtype=bool)
    np.fill_diagonal(in_visual_cone, False)

    # Subtest A: No neighbours alarmed
    states_none = np.array([STATE_S, STATE_S, STATE_S, STATE_S, STATE_S])
    hazard_none = compute_social_hazard(
        dist, in_visual_cone, states_none, r_soc=10.0, theta_soc=0.15, beta_soc=1.5
    )
    assert hazard_none[0] == 0.0

    # Subtest B: 2 neighbours alarmed (agents 1 and 2) -> f = 2/4 = 0.50
    states_half = np.array([STATE_S, STATE_A_CHEM, STATE_A_SOC, STATE_S, STATE_S])
    hazard_half = compute_social_hazard(
        dist, in_visual_cone, states_half, r_soc=10.0, theta_soc=0.15, beta_soc=1.5
    )
    expected_hazard = 1.5 * (0.50 - 0.15)
    assert np.isclose(hazard_half[0], expected_hazard)


def test_causal_state_attribution():
    """
    Deterministic transition check:
    If only chem hazard is positive -> agent transitions to STATE_A_CHEM.
    If only soc hazard is positive -> agent transitions to STATE_A_SOC.
    """
    rng = np.random.default_rng(42)
    states = np.array([STATE_S, STATE_S])
    trigger_times = np.array([np.nan, np.nan])

    chem_hazard = np.array([100.0, 0.0])  # Extremely high hazard guarantees transition
    soc_hazard = np.array([0.0, 100.0])

    new_states, new_triggers = update_states(
        states=states,
        trigger_times=trigger_times,
        chem_hazard=chem_hazard,
        soc_hazard=soc_hazard,
        gamma_recovery=0.0,
        kappa_refractory=0.0,
        dt=0.05,
        current_time=25.0,
        rng=rng
    )

    assert new_states[0] == STATE_A_CHEM
    assert new_states[1] == STATE_A_SOC
    assert np.isclose(new_triggers[0], 25.0)
    assert np.isclose(new_triggers[1], 25.0)


def test_simulation_replicate_execution():
    """Run a fast replicate (short duration) and verify metrics consistency."""
    cfg = SimulationConfig(t_total=5.0, dt=0.1)  # 50 steps
    df_metrics, summary, _ = run_replicate(cfg, seed=1001, condition_id="C1")

    assert len(df_metrics) == 50
    assert "polarisation" in df_metrics.columns
    assert "nnd" in df_metrics.columns
    assert "p_alarm" in df_metrics.columns

    # In baseline C1, no alarms should trigger
    assert df_metrics["p_alarm"].max() == 0.0
    assert summary["peak_pa"] == 0.0
    # Group polarisation in undisturbed shoal should be positive
    assert df_metrics["polarisation"].mean() > 0.5
