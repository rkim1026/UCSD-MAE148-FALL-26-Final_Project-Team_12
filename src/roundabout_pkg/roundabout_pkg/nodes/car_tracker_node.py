#!/usr/bin/env python3
"""/scan → ring observations (phi, speed, gap_time, lead_distance, zone_busy)."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension

from roundabout_pkg.car_tracker import CarTracker
from roundabout_pkg.ring_gap import (
    GapParams,
    TrackGeometry,
    gap_time,
    zone_busy_from_track,
)


class CarTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("car_tracker_node")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("island_radius", 0.95)
        self.declare_parameter("lane_center_radius", 1.30)
        self.declare_parameter("outer_radius", 1.65)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 1.30)
        self.declare_parameter("conflict_angle", 0.0)
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("estop_distance", 0.25)
        # Ego angle on the ring while following (approx). At the yield line
        # the car is not yet on the ring; lead distance then uses conflict.
        self.declare_parameter("ego_angle_on_ring", 0.0)

        track = TrackGeometry(
            island_radius=float(self.get_parameter("island_radius").value),
            lane_center_radius=float(self.get_parameter("lane_center_radius").value),
            outer_radius=float(self.get_parameter("outer_radius").value),
            center_x=float(self.get_parameter("center_x").value),
            center_y=float(self.get_parameter("center_y").value),
            conflict_angle=float(self.get_parameter("conflict_angle").value),
        )
        self.track = track
        self.params = GapParams(
            estop_distance=float(self.get_parameter("estop_distance").value),
        )
        self.tracker = CarTracker(
            track=track,
            min_points=int(self.get_parameter("min_cluster_points").value),
        )
        self.ego_angle = float(self.get_parameter("ego_angle_on_ring").value)

        scan_topic = self.get_parameter("scan_topic").value
        self.ring_pub = self.create_publisher(Float32MultiArray, "/ring_state", 10)
        self.gap_pub = self.create_publisher(Float32, "/gap_time", 10)
        self.phi_pub = self.create_publisher(Float32, "/phi", 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.get_logger().info(f"Car tracker listening on {scan_topic}")

    def _on_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        car, lane_pts = self.tracker.track_nearest(
            ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            stamp,
        )

        # Forward sector clearance for E-stop (robot frame, +x forward).
        front_clear = True
        half = math.radians(20.0)
        for i, r in enumerate(ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            ang = msg.angle_min + i * msg.angle_increment
            if abs(ang) < half and r < self.params.estop_distance:
                front_clear = False
                break

        if car is None:
            phi = float("nan")
            speed = float("nan")
            gt = float("inf")
            busy = False
            lead = float("inf")
        else:
            phi = car.phi
            speed = car.speed
            gt = gap_time(phi, speed, self.track.lane_center_radius)
            occupied = self.tracker.conflict_zone_occupied(lane_pts)
            busy = zone_busy_from_track(phi, speed, self.track, conflict_occupied=occupied)
            lead = self.tracker.lead_distance_ahead(lane_pts, self.ego_angle)

        # Layout: [phi, speed, gap_time, lead_distance, zone_busy, front_clear]
        arr = Float32MultiArray()
        arr.layout.dim = [MultiArrayDimension(label="ring", size=6, stride=6)]
        arr.data = [
            float(phi) if math.isfinite(phi) else -1.0,
            float(speed) if math.isfinite(speed) else -1.0,
            float(gt) if math.isfinite(gt) else -1.0,
            float(lead) if math.isfinite(lead) else -1.0,
            1.0 if busy else 0.0,
            1.0 if front_clear else 0.0,
        ]
        self.ring_pub.publish(arr)

        g = Float32()
        g.data = float(gt) if math.isfinite(gt) else -1.0
        self.gap_pub.publish(g)
        p = Float32()
        p.data = float(phi) if math.isfinite(phi) else -1.0
        self.phi_pub.publish(p)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
