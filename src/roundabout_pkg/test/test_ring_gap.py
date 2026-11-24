"""Unit tests for gap rule + state machine (no ROS required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Allow running without sourcing the ROS workspace.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roundabout_pkg.ring_gap import (  # noqa: E402
    GapParams,
    RingObservation,
    TrackGeometry,
    feedforward_steering,
    following_speed,
    gap_rule_go,
    gap_time,
    remaining_angle_to_conflict,
)
from roundabout_pkg.state_machine import RightOfWayLogic, RowState  # noqa: E402
from roundabout_pkg.sim.scenario_sim import ScenarioSim, run_batch  # noqa: E402


def test_gap_time_basic():
    # R=1.3, phi=pi, v=1.0 → ~4.08 s
    gt = gap_time(math.pi, 1.0, 1.3)
    assert abs(gt - 1.3 * math.pi) < 1e-6


def test_gap_rule_requires_clear_zone():
    params = GapParams(t_clear=1.5, margin=0.5)
    obs = RingObservation(zone_busy=True, gap_time=10.0)
    assert gap_rule_go(obs, params) is False
    obs.zone_busy = False
    obs.gap_time = 1.0  # too small
    assert gap_rule_go(obs, params) is False
    obs.gap_time = 3.0
    assert gap_rule_go(obs, params) is True


def test_remaining_angle_ccw():
    # Car at pi, conflict at 0 → remaining = pi
    phi = remaining_angle_to_conflict(math.pi, 0.0)
    assert abs(phi - math.pi) < 1e-9


def test_following_speed_ramp():
    params = GapParams(follow_min_gap=0.4, follow_full_gap=1.2, cruise_speed=0.8)
    assert following_speed(0.2, params) == 0.0
    assert following_speed(1.5, params) == 0.8
    mid = following_speed(0.8, params)
    assert 0.0 < mid < 0.8


def test_feedforward_about_14_deg():
    ff = feedforward_steering(0.33, 1.3)
    assert abs(math.degrees(ff) - math.degrees(math.atan(0.33 / 1.3))) < 1e-9
    assert 12.0 < math.degrees(ff) < 16.0


def test_state_machine_yield_then_merge():
    logic = RightOfWayLogic(params=GapParams(t_clear=1.0, margin=0.5, go_scans_required=3))
    # Approach to line with small gap → YIELD
    obs = RingObservation(at_yield_line=True, gap_time=0.5, zone_busy=False, front_clear=True)
    cmd = logic.step(obs)
    assert cmd.state == RowState.YIELD
    assert cmd.speed == 0.0

    # Three consecutive good gaps → MERGE
    obs.gap_time = 5.0
    for _ in range(2):
        cmd = logic.step(obs)
        assert cmd.state == RowState.YIELD
    cmd = logic.step(obs)
    assert cmd.state == RowState.MERGE
    assert cmd.speed > 0.0


def test_estop_overrides():
    logic = RightOfWayLogic()
    obs = RingObservation(front_clear=False, at_yield_line=True, gap_time=10.0)
    cmd = logic.step(obs)
    assert cmd.state == RowState.STOP
    assert cmd.speed == 0.0


def test_scenario_batch_zero_collisions_on_emptyish():
    # With a conservative margin, cruising human and staggered angles should
    # produce zero collisions in the default batch.
    stats = run_batch(12, behaviours=["cruise", "slow"], seed=1)
    assert stats["collisions"] == 0
    assert stats["entries"] >= 1


def test_scenario_waits_for_fast_human():
    sim = ScenarioSim(params=GapParams(t_clear=1.5, margin=0.8, cruise_speed=0.8))
    # Human almost at the conflict point, moving fast.
    sim.reset(human_behaviour="fast", human_angle=0.3, ego_dist=0.0)
    saw_yield = False
    for _ in range(30):
        _, _, done, info = sim.step()
        if info["state"] == "YIELD":
            saw_yield = True
            break
    assert saw_yield
