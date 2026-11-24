#!/usr/bin/env python3
"""Camera → yield-line / exit-mark flags."""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from roundabout_pkg.yield_detector import TapeDetectorConfig, YieldLineDetector


class YieldDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("yield_detector_node")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("stop_row_frac", 0.78)

        cfg = TapeDetectorConfig(
            stop_row_frac=float(self.get_parameter("stop_row_frac").value),
        )
        self.detector = YieldLineDetector(cfg)
        self.bridge = CvBridge()

        image_topic = self.get_parameter("image_topic").value
        self.at_line_pub = self.create_publisher(Bool, "/at_yield_line", 10)
        self.line_row_pub = self.create_publisher(Float32, "/yield_line_row", 10)
        self.exit_pub = self.create_publisher(Bool, "/exit_mark", 10)
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.get_logger().info(f"Yield detector listening on {image_topic}")

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return

        at_line, row_frac, _ = self.detector.detect_yield(frame)
        exit_seen, _, _ = self.detector.detect_exit(frame)

        m = Bool()
        m.data = bool(at_line)
        self.at_line_pub.publish(m)

        if row_frac is not None:
            r = Float32()
            r.data = float(row_frac)
            self.line_row_pub.publish(r)

        e = Bool()
        e.data = bool(exit_seen)
        self.exit_pub.publish(e)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YieldDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
