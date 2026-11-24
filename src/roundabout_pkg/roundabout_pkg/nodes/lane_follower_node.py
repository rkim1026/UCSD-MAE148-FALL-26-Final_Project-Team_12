#!/usr/bin/env python3
"""Camera → steering angle. Publishes /lane_steering (Float32)."""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from roundabout_pkg.lane_follower import LaneFollower, LaneFollowerConfig


class LaneFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_follower_node")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("steering_topic", "/lane_steering")
        self.declare_parameter("kp", 0.8)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.05)
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("lane_radius", 1.30)
        self.declare_parameter("use_feedforward", True)

        cfg = LaneFollowerConfig(
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
            wheelbase=float(self.get_parameter("wheelbase").value),
            lane_radius=float(self.get_parameter("lane_radius").value),
            use_feedforward=bool(self.get_parameter("use_feedforward").value),
        )
        self.follower = LaneFollower(cfg)
        self.bridge = CvBridge()
        self._last_t = self.get_clock().now()

        image_topic = self.get_parameter("image_topic").value
        steering_topic = self.get_parameter("steering_topic").value
        self.pub = self.create_publisher(Float32, steering_topic, 10)
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.get_logger().info(f"Lane follower listening on {image_topic}")

    def _on_image(self, msg: Image) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_t).nanoseconds * 1e-9
        self._last_t = now
        if dt <= 0.0 or dt > 1.0:
            dt = 0.05
        try:
            # OAK-D / intel often publish rgb8; fall back to bgr8.
            encoding = msg.encoding.lower()
            if "rgb" in encoding:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return
        steer, debug = self.follower.compute_steering(frame, dt=dt)
        out = Float32()
        out.data = float(steer)
        self.pub.publish(out)
        if self.get_logger().get_effective_level() <= 10:  # DEBUG
            self.get_logger().debug(
                f"steer={steer:.3f} err={debug['error']:.3f} ff={debug['feedforward']:.3f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
