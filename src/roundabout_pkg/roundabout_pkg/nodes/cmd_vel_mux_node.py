#!/usr/bin/env python3
"""Lane steering + ROW speed → /cmd_vel for the UCSD VESC twist node.

linear.x  = throttle in [-1, 1]  (scaled from m/s via cruise_speed)
angular.z = steering in [-1, 1]
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class CmdVelMuxNode(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_mux_node")
        self.declare_parameter("cruise_speed", 0.80)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("steering_topic", "/lane_steering")
        self.declare_parameter("speed_topic", "/row_speed")
        self.declare_parameter("hold_to_run_topic", "/joy_enable")
        self.declare_parameter("require_enable", True)
        self.declare_parameter("publish_hz", 20.0)

        self.cruise = float(self.get_parameter("cruise_speed").value)
        self.require_enable = bool(self.get_parameter("require_enable").value)
        self._steer = 0.0
        self._speed_mps = 0.0
        self._enabled = not self.require_enable

        cmd_topic = self.get_parameter("cmd_vel_topic").value
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(
            Float32, self.get_parameter("steering_topic").value, self._on_steer, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("speed_topic").value, self._on_speed, 10
        )
        self.create_subscription(
            Bool, self.get_parameter("hold_to_run_topic").value, self._on_enable, 10
        )

        period = 1.0 / float(self.get_parameter("publish_hz").value)
        self.create_timer(period, self._tick)
        self.get_logger().info(f"Publishing Twist on {cmd_topic}")

    def _on_steer(self, msg: Float32) -> None:
        self._steer = float(msg.data)

    def _on_speed(self, msg: Float32) -> None:
        self._speed_mps = float(msg.data)

    def _on_enable(self, msg: Bool) -> None:
        self._enabled = bool(msg.data)

    def _tick(self) -> None:
        twist = Twist()
        if not self._enabled:
            self.pub.publish(twist)
            return
        throttle = 0.0
        if self.cruise > 1e-6:
            throttle = max(-1.0, min(1.0, self._speed_mps / self.cruise))
        twist.linear.x = float(throttle)
        twist.angular.z = float(max(-1.0, min(1.0, self._steer)))
        self.pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
