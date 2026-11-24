#!/usr/bin/env python3
"""Combine ring state + tape flags → desired speed via the 5-state machine."""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from roundabout_pkg.ring_gap import GapParams, RingObservation
from roundabout_pkg.state_machine import RightOfWayLogic, RowState


class RightOfWayNode(Node):
    def __init__(self) -> None:
        super().__init__("right_of_way_node")
        self.declare_parameter("t_clear", 1.5)
        self.declare_parameter("margin", 0.5)
        self.declare_parameter("go_scans_required", 3)
        self.declare_parameter("cruise_speed", 0.80)
        self.declare_parameter("follow_min_gap", 0.40)
        self.declare_parameter("follow_full_gap", 1.20)
        self.declare_parameter("merge_speed", 0.50)
        self.declare_parameter("speed_topic", "/row_speed")
        self.declare_parameter("classifier_path", "")
        self.declare_parameter("use_classifier_veto", False)

        params = GapParams(
            t_clear=float(self.get_parameter("t_clear").value),
            margin=float(self.get_parameter("margin").value),
            go_scans_required=int(self.get_parameter("go_scans_required").value),
            cruise_speed=float(self.get_parameter("cruise_speed").value),
            follow_min_gap=float(self.get_parameter("follow_min_gap").value),
            follow_full_gap=float(self.get_parameter("follow_full_gap").value),
        )
        self.logic = RightOfWayLogic(
            params=params,
            merge_speed=float(self.get_parameter("merge_speed").value),
        )

        if bool(self.get_parameter("use_classifier_veto").value):
            path = str(self.get_parameter("classifier_path").value)
            self._load_veto(path)

        self._at_line = False
        self._exit_mark = False
        self._past_conflict = False
        self._obs = RingObservation()

        speed_topic = self.get_parameter("speed_topic").value
        self.speed_pub = self.create_publisher(Float32, speed_topic, 10)
        self.state_pub = self.create_publisher(String, "/row_state", 10)

        self.create_subscription(Float32MultiArray, "/ring_state", self._on_ring, 10)
        self.create_subscription(Bool, "/at_yield_line", self._on_line, 10)
        self.create_subscription(Bool, "/exit_mark", self._on_exit, 10)
        self.create_subscription(Bool, "/past_conflict", self._on_past, 10)

        self.get_logger().info("Right-of-way logic ready")

    def _load_veto(self, path: str) -> None:
        if not path:
            self.get_logger().warn("use_classifier_veto=True but classifier_path empty")
            return
        try:
            from roundabout_pkg.ml.go_wait_classifier import GoWaitClassifier

            clf = GoWaitClassifier.load(Path(path))

            def veto(obs: RingObservation) -> bool:
                return clf.allow_go(obs, self.logic.params)

            self.logic.veto = veto
            self.get_logger().info(f"Loaded go/wait veto from {path}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to load classifier: {exc}")

    def _on_line(self, msg: Bool) -> None:
        self._at_line = bool(msg.data)

    def _on_exit(self, msg: Bool) -> None:
        self._exit_mark = bool(msg.data)

    def _on_past(self, msg: Bool) -> None:
        self._past_conflict = bool(msg.data)

    def _on_ring(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 6:
            return
        phi, speed, gt, lead, busy, front = data[:6]

        def _pos(v: float, inf_like: float = float("inf")) -> float:
            return inf_like if v < 0.0 else float(v)

        obs = RingObservation(
            at_yield_line=self._at_line,
            zone_busy=busy >= 0.5,
            gap_time=_pos(gt),
            lead_distance=_pos(lead),
            phi=float("nan") if phi < 0.0 else float(phi),
            car_speed=float("nan") if speed < 0.0 else float(speed),
            own_speed=0.0,
            exit_mark_seen=self._exit_mark,
            past_conflict=self._past_conflict,
            front_clear=front >= 0.5,
        )
        # Heuristic: once we have merged and lead distance is finite on-ring,
        # treat as past conflict if the yield line has left the frame.
        if (
            self.logic.state in (RowState.MERGE, RowState.FOLLOW)
            and not self._at_line
            and math.isfinite(obs.lead_distance)
        ):
            obs.past_conflict = True

        cmd = self.logic.step(obs)
        self._obs = obs

        s = Float32()
        s.data = float(cmd.speed)
        self.speed_pub.publish(s)

        st = String()
        st.data = cmd.state.name
        self.state_pub.publish(st)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RightOfWayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
