"""Hindsight labels for go/wait training — no hand labelling.

At each wait scan, ask: if we had gone then, would the human car have
stayed out of the conflict zone until we cleared (T_clear)?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import math
import numpy as np

from roundabout_pkg.ring_gap import GapParams, gap_time, scale_free_features


@dataclass
class WaitSample:
    stamp: float
    phi: float
    car_speed: float
    gap_time: float
    zone_busy: bool
    own_speed: float
    # Future trajectory of the circulating car (angles, speeds, stamps).
    # Filled by the labeller from the rest of the bag / log.


@dataclass
class LabelledSample:
    features: np.ndarray  # [gap_ratio, busy, speed_norm]
    label_safe: int  # 1 = safe to go, 0 = unsafe
    stamp: float


def would_have_been_safe(
    phi0: float,
    speed0: float,
    future_angles: Sequence[float],
    future_speeds: Sequence[float],
    future_stamps: Sequence[float],
    t0: float,
    params: GapParams,
    lane_radius: float,
    conflict_angle: float = 0.0,
    conflict_half_angle: float = math.radians(20.0),
) -> bool:
    """True if the conflict zone stays clear for T_clear after t0."""
    from roundabout_pkg.ring_gap import wrap_angle

    t_end = t0 + params.t_clear
    for ang, spd, t in zip(future_angles, future_speeds, future_stamps):
        if t < t0:
            continue
        if t > t_end:
            break
        if abs(wrap_angle(ang - conflict_angle)) < conflict_half_angle:
            return False
    # Also reject if current gap time already shorter than clear time.
    gt = gap_time(phi0, speed0, lane_radius)
    if math.isfinite(gt) and gt < params.t_clear:
        return False
    return True


def label_waits(
    samples: Iterable[WaitSample],
    futures: Iterable[tuple],
    params: GapParams,
    lane_radius: float,
) -> List[LabelledSample]:
    """futures[i] = (angles, speeds, stamps) aligned with samples[i]."""
    out: List[LabelledSample] = []
    for sample, (angs, spds, stamps) in zip(samples, futures):
        safe = would_have_been_safe(
            sample.phi,
            sample.car_speed,
            angs,
            spds,
            stamps,
            sample.stamp,
            params,
            lane_radius,
        )
        # Build a RingObservation-like feature vector.
        from roundabout_pkg.ring_gap import RingObservation

        obs = RingObservation(
            zone_busy=sample.zone_busy,
            gap_time=sample.gap_time,
            own_speed=sample.own_speed,
            phi=sample.phi,
            car_speed=sample.car_speed,
        )
        feats = scale_free_features(obs, params)
        out.append(LabelledSample(features=feats, label_safe=int(safe), stamp=sample.stamp))
    return out


def score_rule(
    labelled: Sequence[LabelledSample],
    params: GapParams,
) -> dict:
    """Count unsafe goes / needless waits for the hand-written rule."""
    unsafe_goes = 0
    needless_waits = 0
    for s in labelled:
        gap_ratio, busy, _ = s.features
        rule_go = busy < 0.5 and gap_ratio > (1.0 + params.margin / max(params.t_clear, 1e-6))
        if rule_go and s.label_safe == 0:
            unsafe_goes += 1
        if (not rule_go) and s.label_safe == 1:
            needless_waits += 1
    return {
        "n": len(labelled),
        "unsafe_goes": unsafe_goes,
        "needless_waits": needless_waits,
    }
