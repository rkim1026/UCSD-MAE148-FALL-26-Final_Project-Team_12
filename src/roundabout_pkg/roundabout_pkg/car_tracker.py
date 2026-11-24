"""LiDAR → circulating-car angle, speed, and lead distance.

When the RoboCar sits at the yield line, the ring centre is fixed in the
LiDAR frame. Points inside the lane annulus are clustered; each car-sized
cluster becomes a tracked vehicle whose angle gives phi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
import numpy as np

from roundabout_pkg.ring_gap import (
    TrackGeometry,
    remaining_angle_to_conflict,
    wrap_angle,
)


@dataclass
class TrackedCar:
    angle: float  # absolute angle around ring centre [rad]
    phi: float  # remaining angle to conflict [rad]
    speed: float  # m/s along lane centre
    radius: float  # distance from ring centre [m]
    n_points: int


@dataclass
class TrackerState:
    last_angle: Optional[float] = None
    last_stamp: Optional[float] = None
    smoothed_speed: float = 0.0
    alpha: float = 0.4  # exponential smoothing of Δφ/Δt


@dataclass
class CarTracker:
    track: TrackGeometry
    min_points: int = 3
    max_cluster_gap: float = 0.25  # m between neighbouring points in a cluster
    speed_filter: TrackerState = field(default_factory=TrackerState)

    def lidar_to_xy(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
    ) -> np.ndarray:
        """Convert a LaserScan to Nx2 points in the LiDAR frame."""
        n = len(ranges)
        angles = angle_min + np.arange(n, dtype=np.float64) * angle_increment
        r = np.asarray(ranges, dtype=np.float64)
        valid = np.isfinite(r) & (r >= range_min) & (r <= range_max) & (r > 0.05)
        x = r[valid] * np.cos(angles[valid])
        y = r[valid] * np.sin(angles[valid])
        if x.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack([x, y])

    def points_in_lane(self, points_xy: np.ndarray) -> np.ndarray:
        """Keep points whose distance from the ring centre is inside the lane."""
        if points_xy.size == 0:
            return points_xy
        dx = points_xy[:, 0] - self.track.center_x
        dy = points_xy[:, 1] - self.track.center_y
        radii = np.hypot(dx, dy)
        # Slightly pad the annulus so clusters near the tape edges survive.
        inner = self.track.island_radius + 0.05
        outer = self.track.outer_radius - 0.05
        mask = (radii >= inner) & (radii <= outer)
        return points_xy[mask]

    def cluster_points(self, points_xy: np.ndarray) -> List[np.ndarray]:
        """Greedy nearest-neighbour clustering along arc order."""
        if points_xy.shape[0] == 0:
            return []
        dx = points_xy[:, 0] - self.track.center_x
        dy = points_xy[:, 1] - self.track.center_y
        angles = np.arctan2(dy, dx)
        order = np.argsort(angles)
        pts = points_xy[order]
        clusters: List[List[np.ndarray]] = [[pts[0]]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - clusters[-1][-1]) <= self.max_cluster_gap:
                clusters[-1].append(pts[i])
            else:
                clusters.append([pts[i]])
        # Also merge first/last if they wrap around the circle.
        if len(clusters) > 1 and np.linalg.norm(clusters[0][0] - clusters[-1][-1]) <= self.max_cluster_gap:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
        return [np.asarray(c) for c in clusters if len(c) >= self.min_points]

    def cluster_to_car(self, cluster: np.ndarray) -> TrackedCar:
        cx = float(np.mean(cluster[:, 0]))
        cy = float(np.mean(cluster[:, 1]))
        dx = cx - self.track.center_x
        dy = cy - self.track.center_y
        angle = math.atan2(dy, dx)
        radius = math.hypot(dx, dy)
        phi = remaining_angle_to_conflict(angle, self.track.conflict_angle)
        return TrackedCar(
            angle=angle,
            phi=phi,
            speed=0.0,
            radius=radius,
            n_points=len(cluster),
        )

    def update_speed(self, car: TrackedCar, stamp: float) -> TrackedCar:
        """Exponentially-smoothed speed from Δangle / Δt along the lane."""
        st = self.speed_filter
        if st.last_angle is not None and st.last_stamp is not None:
            dt = stamp - st.last_stamp
            if dt > 1e-3:
                dtheta = wrap_angle(car.angle - st.last_angle)
                # Circulating CCW → positive dtheta when progressing toward conflict
                # in our convention; use absolute arc speed.
                inst = abs(dtheta) * self.track.lane_center_radius / dt
                st.smoothed_speed = (
                    st.alpha * inst + (1.0 - st.alpha) * st.smoothed_speed
                )
        st.last_angle = car.angle
        st.last_stamp = stamp
        car.speed = st.smoothed_speed
        return car

    def track_nearest(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        stamp: float,
    ) -> Tuple[Optional[TrackedCar], np.ndarray]:
        """Return the nearest (smallest phi) car and the lane-filtered points."""
        xy = self.lidar_to_xy(ranges, angle_min, angle_increment, range_min, range_max)
        lane = self.points_in_lane(xy)
        clusters = self.cluster_points(lane)
        if not clusters:
            return None, lane
        cars = [self.cluster_to_car(c) for c in clusters]
        cars.sort(key=lambda c: c.phi)
        nearest = self.update_speed(cars[0], stamp)
        return nearest, lane

    def lead_distance_ahead(
        self,
        points_xy: np.ndarray,
        ego_angle: float,
        half_width_rad: float = math.radians(25.0),
    ) -> float:
        """Bumper-ish distance to nearest lane points ahead along the curve."""
        if points_xy.size == 0:
            return float("inf")
        dx = points_xy[:, 0] - self.track.center_x
        dy = points_xy[:, 1] - self.track.center_y
        angles = np.arctan2(dy, dx)
        # Remaining arc from ego heading (approx. ego pose on the ring).
        deltas = np.array(
            [remaining_angle_to_conflict(a, ego_angle + math.pi) for a in angles]
        )
        # Simpler: points whose angle is just CCW of ego.
        ahead = []
        for i, a in enumerate(angles):
            d = wrap_angle(a - ego_angle)
            if 0.0 < d < half_width_rad * 4.0:
                arc = d * self.track.lane_center_radius
                ahead.append(arc)
        if not ahead:
            return float("inf")
        return float(min(ahead))

    def conflict_zone_occupied(
        self,
        points_xy: np.ndarray,
        half_angle: float = math.radians(20.0),
    ) -> bool:
        """True if any lane point sits near the conflict angle."""
        if points_xy.size == 0:
            return False
        dx = points_xy[:, 0] - self.track.center_x
        dy = points_xy[:, 1] - self.track.center_y
        angles = np.arctan2(dy, dx)
        for a in angles:
            if abs(wrap_angle(a - self.track.conflict_angle)) < half_angle:
                return True
        return False
