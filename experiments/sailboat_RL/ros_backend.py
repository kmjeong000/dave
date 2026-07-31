from __future__ import annotations

import math
import time
from typing import Any, Sequence

from .core import RawState


class Ros2AttachBackend:
    """Attach to an already running, armed BO trial and publish RL residuals.

    This backend intentionally does not restart Gazebo or ArduPilot. Its
    reset() zeros residuals and resets the local waypoint tracker. A later
    lifecycle backend can implement full process/mission reset behind the same
    environment protocol.
    """

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        namespace: str = "sailboat",
        wind_world_xyz_mps: Sequence[float] = (0.0, 8.0, 0.0),
        waypoint_capture_radius_m: float = 5.0,
        telemetry_timeout_s: float = 5.0,
        adapter_node_name: str = "/sailboat_command_adapter",
        manage_adapter_enabled: bool = True,
    ):
        if not waypoints:
            raise ValueError("at least one waypoint is required")
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.parameter import Parameter
            from rclpy.parameter_client import AsyncParameterClient
            from std_msgs.msg import Float64
        except ImportError as exc:
            raise RuntimeError(
                "Ros2AttachBackend requires rclpy, nav_msgs, and std_msgs"
            ) from exc

        self._rclpy = rclpy
        self._Float64 = Float64
        self._Parameter = Parameter
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self.namespace = namespace.strip("/")
        self.waypoints = [
            (float(waypoint[0]), float(waypoint[1])) for waypoint in waypoints
        ]
        self.wind_x_mps = float(wind_world_xyz_mps[0])
        self.wind_y_mps = float(wind_world_xyz_mps[1])
        self.capture_radius_m = float(waypoint_capture_radius_m)
        self.telemetry_timeout_s = float(telemetry_timeout_s)
        self.manage_adapter_enabled = bool(manage_adapter_enabled)
        self._node = rclpy.create_node(
            f"sailboat_rl_env_{int(time.time() * 1000)}"
        )
        self._adapter_parameters = AsyncParameterClient(
            self._node,
            adapter_node_name,
        )

        joint_prefix = f"/model/{self.namespace}/joint"
        residual_prefix = f"/{self.namespace}/rl/residual"
        self._residual_publishers = {
            "rudder": self._node.create_publisher(
                Float64, f"{residual_prefix}/rudder", 10
            ),
            "sail": self._node.create_publisher(
                Float64, f"{residual_prefix}/sail", 10
            ),
        }
        self._node.create_subscription(
            Odometry,
            f"/model/{self.namespace}/odometry",
            self._on_odometry,
            10,
        )
        self._node.create_subscription(
            Float64,
            f"{joint_prefix}/rudder_joint/base_cmd_pos",
            lambda msg: self._on_base("rudder", msg),
            10,
        )
        self._node.create_subscription(
            Float64,
            f"{joint_prefix}/sail_joint/base_cmd_pos",
            lambda msg: self._on_base("sail", msg),
            10,
        )

        self._pose: dict[str, float] | None = None
        self._base = {"rudder": None, "sail": None}
        self._residual = {"rudder": 0.0, "sail": 0.0}
        self._waypoint_index = 0
        self._closed = False

    def _set_adapter_enabled(self, enabled: bool) -> None:
        if not self.manage_adapter_enabled:
            return
        if not self._adapter_parameters.wait_for_service(
            timeout_sec=self.telemetry_timeout_s
        ):
            raise TimeoutError(
                "timed out waiting for sailboat command-adapter parameter service"
            )
        future = self._adapter_parameters.set_parameters(
            [
                self._Parameter(
                    "residual_enabled",
                    self._Parameter.Type.BOOL,
                    bool(enabled),
                )
            ]
        )
        self._rclpy.spin_until_future_complete(
            self._node,
            future,
            timeout_sec=self.telemetry_timeout_s,
        )
        if not future.done() or future.result() is None:
            raise TimeoutError("timed out setting residual_enabled")
        results = future.result()
        if not results or not all(result.successful for result in results):
            reasons = ", ".join(
                result.reason for result in results if not result.successful
            )
            raise RuntimeError(
                f"command adapter rejected residual_enabled={enabled}: {reasons}"
            )

    def _on_odometry(self, msg: Any) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        siny_cosp = 2.0 * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        )
        cosy_cosp = 1.0 - 2.0 * (
            float(orientation.y) ** 2 + float(orientation.z) ** 2
        )
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        sinr_cosp = 2.0 * (
            float(orientation.w) * float(orientation.x)
            + float(orientation.y) * float(orientation.z)
        )
        cosr_cosp = 1.0 - 2.0 * (
            float(orientation.x) ** 2 + float(orientation.y) ** 2
        )
        roll_deg = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        linear = msg.twist.twist.linear
        stamp = msg.header.stamp
        sim_time_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        self._pose = {
            "sim_time_s": sim_time_s,
            "x_m": float(position.x),
            "y_m": float(position.y),
            "yaw_rad": yaw_rad,
            "speed_mps": math.hypot(float(linear.x), float(linear.y)),
            "roll_deg": roll_deg,
        }

    def _on_base(self, surface: str, msg: Any) -> None:
        self._base[surface] = float(msg.data)

    def _spin_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            if (
                self._pose is not None
                and self._base["rudder"] is not None
                and self._base["sail"] is not None
            ):
                return
        raise TimeoutError(
            "timed out waiting for odometry and BO base actuator commands"
        )

    def _publish_residuals(self, rudder: float, sail: float) -> None:
        self._residual["rudder"] = float(rudder)
        self._residual["sail"] = float(sail)
        for surface, value in self._residual.items():
            message = self._Float64()
            message.data = value
            self._residual_publishers[surface].publish(message)

    def _update_waypoint(self) -> None:
        if self._pose is None:
            return
        target_x, target_y = self.waypoints[self._waypoint_index]
        distance = math.hypot(
            target_x - self._pose["x_m"],
            target_y - self._pose["y_m"],
        )
        if (
            distance <= self.capture_radius_m
            and self._waypoint_index < len(self.waypoints) - 1
        ):
            self._waypoint_index += 1

    def _state(self) -> RawState:
        if self._pose is None:
            raise RuntimeError("odometry is not available")
        self._update_waypoint()
        target_x, target_y = self.waypoints[self._waypoint_index]
        final_distance = math.hypot(
            target_x - self._pose["x_m"],
            target_y - self._pose["y_m"],
        )
        mission_complete = (
            self._waypoint_index == len(self.waypoints) - 1
            and final_distance <= self.capture_radius_m
        )
        return RawState(
            **self._pose,
            target_x_m=target_x,
            target_y_m=target_y,
            wind_x_mps=self.wind_x_mps,
            wind_y_mps=self.wind_y_mps,
            base_rudder_rad=float(self._base["rudder"]),
            base_sail_rad=float(self._base["sail"]),
            residual_rudder_rad=self._residual["rudder"],
            residual_sail_rad=self._residual["sail"],
            waypoint_index=self._waypoint_index,
            waypoint_count=len(self.waypoints),
            mission_complete=mission_complete,
        )

    def reset(self, *, seed: int | None, options: dict[str, Any]) -> RawState:
        del seed
        if options.get("waypoint_index") is not None:
            requested = int(options["waypoint_index"])
            self._waypoint_index = min(max(requested, 0), len(self.waypoints) - 1)
        else:
            self._waypoint_index = 0
        self._set_adapter_enabled(True)
        self._publish_residuals(0.0, 0.0)
        self._spin_until_ready(self.telemetry_timeout_s)
        return self._state()

    def step(
        self,
        rudder_residual_rad: float,
        sail_residual_rad: float,
        control_period_s: float,
    ) -> RawState:
        self._publish_residuals(rudder_residual_rad, sail_residual_rad)
        deadline = time.monotonic() + float(control_period_s)
        while time.monotonic() < deadline:
            self._rclpy.spin_once(
                self._node,
                timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())),
            )
        return self._state()

    def close(self) -> None:
        if self._closed:
            return
        self._publish_residuals(0.0, 0.0)
        self._rclpy.spin_once(self._node, timeout_sec=0.05)
        self._set_adapter_enabled(False)
        self._node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
        self._closed = True
