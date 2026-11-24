"""Ring geometry, gap-time rule, and shared feature helpers.

gap_time = R * phi / v
go  <=>  zone empty AND gap_time > T_clear + margin

These functions are pure NumPy / Python so the scenario simulator and the
ROS2 nodes run the exact same decision code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


@dataclass
class TrackGeometry:
    """Example track from the Team 12 build guide (measure and replace)."""

    island_radius: float = 0.95  # m
    lane_center_radius: float = 1.30  # m  (R in gap_time = R * phi / v)
    outer_radius: float = 1.65  # m
    lane_width: float = 0.70  # m
    # Ring centre in the LiDAR frame when the car is stopped at the yield line.
    # Measure once with a tape measure after the car stops on the orange tape.
    center_x: float = 0.0  # m, lidar frame
    center_y: float = 1.30  # m, lidar forward ≈ lane-centre radius
    # Conflict point: where the entry path crosses the circulating lane.
    # Angle 0 is +x in the ring frame; increase counter-clockwise (traffic CCW).
    conflict_angle: float = 0.0  # rad
    # A car closer than this remaining angle that is nearly stopped counts as busy.
    stopped_busy_angle: float = math.radians(90.0)


@dataclass
class GapParams:
    t_clear: float = 1.5  # s — measure on the real car (slowest of 10 runs)
    margin: float = 0.5  # s — covers LiDAR period + processing
    go_scans_required: int = 3  # ~0.3 s at 10 Hz
    min_cluster_points: int = 3
    # Bumper-to-bumper stop gap when following on the ring.
    follow_min_gap: float = 0.40  # m
    follow_full_gap: float = 1.20  # m
    cruise_speed: float = 0.80  # m/s  (also used to normalise features)
    # Soft stop buffer used by the interactive sim's continuous following law.
    follow_stop_buffer: float = 0.35  # m
    follow_headway: float = 0.80  # s
    # Emergency stop distance straight ahead.
    estop_distance: float = 0.25  # m


@dataclass
class RingObservation:
    """Everything the right-of-way logic needs from one scan / time step."""

    at_yield_line: bool = False
    zone_busy: bool = False
    gap_time: float = float("inf")
    lead_distance: float = float("inf")
    phi: float = float("nan")  # remaining angle of nearest circulating car [rad]
    car_speed: float = float("nan")  # m/s along the lane
    own_speed: float = 0.0  # m/s
    exit_mark_seen: bool = False
    past_conflict: bool = False
    front_clear: bool = True  # nothing within estop_distance ahead


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def remaining_angle_to_conflict(
    car_angle: float,
    conflict_angle: float,
    *,
    circulating_ccw: bool = True,
) -> float:
    """Angle the circulating car still has to travel to the conflict point.

    Traffic circulates counter-clockwise. Remaining angle is in [0, 2pi).
    """
    if circulating_ccw:
        delta = wrap_angle(conflict_angle - car_angle)
        if delta < 0.0:
            delta += 2.0 * math.pi
        return delta
    delta = wrap_angle(car_angle - conflict_angle)
    if delta < 0.0:
        delta += 2.0 * math.pi
    return delta


def gap_time(phi: float, speed: float, lane_radius: float) -> float:
    """Time until a circulating car reaches the conflict point.

    gap_time = R * phi / v.  If speed is near zero, return +inf unless the
    caller already marked the zone busy via the stopped-car check.
    """
    if not math.isfinite(phi) or phi < 0.0:
        return float("inf")
    if speed is None or not math.isfinite(speed) or abs(speed) < 1e-3:
        return float("inf")
    return (lane_radius * phi) / abs(speed)


def gap_rule_go(
    obs: RingObservation,
    params: GapParams,
) -> bool:
    """Hand-written acceptance rule. Learned models may only veto this."""
    if obs.zone_busy:
        return False
    if not math.isfinite(obs.gap_time):
        # Empty ring (no tracked car) → go.
        return True
    return obs.gap_time > (params.t_clear + params.margin)


def scale_free_features(
    obs: RingObservation,
    params: GapParams,
) -> np.ndarray:
    """Three numbers shared by sim, Waymax mining, and the real car.

    [gap_ratio, zone_busy, own_speed / cruise]
    """
    if math.isfinite(obs.gap_time) and params.t_clear > 0.0:
        gap_ratio = obs.gap_time / params.t_clear
    else:
        gap_ratio = 10.0  # empty / stopped far away
    busy = 1.0 if obs.zone_busy else 0.0
    speed_norm = obs.own_speed / max(params.cruise_speed, 1e-6)
    return np.array([gap_ratio, busy, speed_norm], dtype=np.float64)


def following_speed(
    lead_distance: float,
    params: GapParams,
    *,
    continuous: bool = False,
) -> float:
    """Speed while circulating behind a lead car.

    Piecewise (floor-test version from step 2.2):
      d < follow_min_gap  → 0
      d > follow_full_gap → cruise
      else linear ramp

    continuous=True uses the sim law:
      min(cruise, (gap - stop_buffer) / headway)
    """
    if not math.isfinite(lead_distance):
        return params.cruise_speed

    if continuous:
        if lead_distance <= params.follow_stop_buffer:
            return 0.0
        return min(
            params.cruise_speed,
            (lead_distance - params.follow_stop_buffer) / max(params.follow_headway, 1e-6),
        )

    if lead_distance < params.follow_min_gap:
        return 0.0
    if lead_distance >= params.follow_full_gap:
        return params.cruise_speed
    span = params.follow_full_gap - params.follow_min_gap
    return params.cruise_speed * (lead_distance - params.follow_min_gap) / span


def feedforward_steering(wheelbase: float, lane_radius: float) -> float:
    """Fixed steering bias for a constant-radius curve: atan(L / R) [rad]."""
    if lane_radius <= 0.0:
        return 0.0
    return math.atan(wheelbase / lane_radius)


def zone_busy_from_track(
    phi: Optional[float],
    speed: Optional[float],
    track: TrackGeometry,
    *,
    conflict_occupied: bool = False,
) -> bool:
    """Conflict zone occupied, or a nearly-stopped car still within 90°."""
    if conflict_occupied:
        return True
    if phi is None or not math.isfinite(phi):
        return False
    if speed is None or not math.isfinite(speed):
        return False
    if abs(speed) < 0.05 and phi < track.stopped_busy_angle:
        return True
    return False
