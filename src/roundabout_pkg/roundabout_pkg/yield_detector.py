"""Detect the orange yield-line tape and a distinct exit-mark colour.

Stop when the orange band reaches a fixed image row near the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class TapeDetectorConfig:
    # Orange yield line (HSV). Tune under your lighting.
    orange_h_low: int = 5
    orange_s_low: int = 100
    orange_v_low: int = 100
    orange_h_high: int = 25
    orange_s_high: int = 255
    orange_v_high: int = 255
    # Exit mark — default green tape; change if you use another colour.
    exit_h_low: int = 40
    exit_s_low: int = 80
    exit_v_low: int = 80
    exit_h_high: int = 90
    exit_s_high: int = 255
    exit_v_high: int = 255
    # Fraction of image height: band centre must be below this to count as "at line".
    stop_row_frac: float = 0.78
    min_pixels: int = 80
    search_bottom_frac: float = 0.35  # only look in the bottom portion


class YieldLineDetector:
    def __init__(self, config: Optional[TapeDetectorConfig] = None):
        self.cfg = config or TapeDetectorConfig()

    def _require_cv(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for YieldLineDetector")

    def _mask(
        self,
        bgr: np.ndarray,
        h_low: int,
        s_low: int,
        v_low: int,
        h_high: int,
        s_high: int,
        v_high: int,
    ) -> np.ndarray:
        self._require_cv()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = np.array([h_low, s_low, v_low])
        high = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, low, high)
        h, _ = mask.shape
        y0 = int(h * (1.0 - self.cfg.search_bottom_frac))
        roi = np.zeros_like(mask)
        roi[y0:, :] = mask[y0:, :]
        return roi

    def _band_row(self, mask: np.ndarray) -> Optional[float]:
        ys, xs = np.where(mask > 0)
        if xs.size < self.cfg.min_pixels:
            return None
        return float(np.mean(ys))

    def detect_yield(self, bgr: np.ndarray) -> Tuple[bool, Optional[float], np.ndarray]:
        """Return (at_line, mean_row_frac, mask)."""
        c = self.cfg
        mask = self._mask(
            bgr,
            c.orange_h_low,
            c.orange_s_low,
            c.orange_v_low,
            c.orange_h_high,
            c.orange_s_high,
            c.orange_v_high,
        )
        row = self._band_row(mask)
        if row is None:
            return False, None, mask
        frac = row / float(bgr.shape[0])
        return frac >= c.stop_row_frac, frac, mask

    def detect_exit(self, bgr: np.ndarray) -> Tuple[bool, Optional[float], np.ndarray]:
        c = self.cfg
        mask = self._mask(
            bgr,
            c.exit_h_low,
            c.exit_s_low,
            c.exit_v_low,
            c.exit_h_high,
            c.exit_s_high,
            c.exit_v_high,
        )
        row = self._band_row(mask)
        if row is None:
            return False, None, mask
        frac = row / float(bgr.shape[0])
        # Exit mark counts as soon as it is visible in the lower ROI.
        return True, frac, mask
