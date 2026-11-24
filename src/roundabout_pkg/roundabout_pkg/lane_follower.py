"""OpenCV colour-mask lane follower with feedforward + PID steering.

Steering only — throttle is owned by the right-of-way logic.
On a constant-radius ring, feedforward = atan(wheelbase / R) keeps the
PID from settling off-centre toward the outer tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np

from roundabout_pkg.ring_gap import feedforward_steering

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import for pure unit tests
    cv2 = None


@dataclass
class LaneFollowerConfig:
    # HSV mask for lane-edge tape (tune on the floor).
    h_low: int = 0
    s_low: int = 0
    v_low: int = 180
    h_high: int = 180
    s_high: int = 60
    v_high: int = 255
    # Use the bottom fraction of the image as the ROI.
    roi_top_frac: float = 0.45
    # PID on normalised lateral error in [-1, 1].
    kp: float = 0.8
    ki: float = 0.0
    kd: float = 0.05
    wheelbase: float = 0.33  # m
    lane_radius: float = 1.30  # m
    # Output steering is angular.z in approx. [-1, 1] for VESC twist.
    max_steer: float = 1.0
    use_feedforward: bool = True


class LaneFollower:
    def __init__(self, config: Optional[LaneFollowerConfig] = None):
        self.cfg = config or LaneFollowerConfig()
        self._integral = 0.0
        self._prev_error = 0.0
        self.ff = (
            feedforward_steering(self.cfg.wheelbase, self.cfg.lane_radius)
            if self.cfg.use_feedforward
            else 0.0
        )
        # Map feedforward radians into roughly the same [-1, 1] steering units
        # the VESC expects. ~25° full lock ≈ 1.0 on many course cars — tune.
        self.ff_norm = math.degrees(self.ff) / 25.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def _require_cv(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for LaneFollower")

    def mask_lane(self, bgr: np.ndarray) -> np.ndarray:
        self._require_cv()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = np.array([self.cfg.h_low, self.cfg.s_low, self.cfg.v_low])
        high = np.array([self.cfg.h_high, self.cfg.s_high, self.cfg.v_high])
        mask = cv2.inRange(hsv, low, high)
        h, _ = mask.shape
        y0 = int(h * self.cfg.roi_top_frac)
        roi = np.zeros_like(mask)
        roi[y0:, :] = mask[y0:, :]
        return roi

    def lane_center_error(self, bgr: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """Return normalised lateral error in [-1, 1] (positive = lane right of centre)."""
        self._require_cv()
        mask = self.mask_lane(bgr)
        ys, xs = np.where(mask > 0)
        if xs.size < 20:
            return 0.0, mask
        cx = float(np.mean(xs))
        width = bgr.shape[1]
        error = (cx - (width / 2.0)) / (width / 2.0)
        return float(np.clip(error, -1.0, 1.0)), mask

    def compute_steering(self, bgr: np.ndarray, dt: float = 0.05) -> Tuple[float, dict]:
        error, mask = self.lane_center_error(bgr)
        self._integral += error * dt
        derivative = (error - self._prev_error) / max(dt, 1e-3)
        self._prev_error = error
        pid = (
            self.cfg.kp * error
            + self.cfg.ki * self._integral
            + self.cfg.kd * derivative
        )
        # Positive image error (tape right of centre) → steer right (negative
        # angular.z on many differential / ackermann conventions used here:
        # we treat +angular.z as left. Flip the sign if your car mirrors.)
        steer = float(np.clip(-(pid) + self.ff_norm, -self.cfg.max_steer, self.cfg.max_steer))
        debug = {
            "error": error,
            "pid": pid,
            "feedforward": self.ff_norm,
            "steering": steer,
            "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
        }
        return steer, debug
