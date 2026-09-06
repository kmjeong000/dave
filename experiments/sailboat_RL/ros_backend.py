from __future__ import annotations

import math
import time
from typing import Any, Sequence

from experiments.sailboat_BO.run_trial import segment_metrics

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
        spawn_xy_m: Sequence[float] = (0.0, 0.0),
        namespace: str = "sailboat",
        wind_world_xyz_mps: Sequence[float] = (0.0, 8.0, 0.0),
        waypoint_capture_radius_m: float = 5.0,
        waypoint_capture_hold_s: float = 1.0,
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
        spawn = tuple(spawn_xy_m)
        if len(spawn) < 2:
            raise ValueError("spawn_xy_m must contain x and y")
        self.path_points = [
            (float(spawn[0]), float(spawn[1])),
            *self.waypoints,
        ]
        self.wind_x_mps = float(wind_world_xyz_mps[0])
        self.wind_y_mps = float(wind_world_xyz_mps[1])
        self.capture_radius_m = float(waypoint_capture_radius_m)
        self.capture_hold_s = max(0.0, float(waypoint_capture_hold_s))
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
        # Observe the command adapter's real bounded outputs rather than
        # reconstructing them from the requested residual. This preserves
        # rate-limit and clamp evidence after a failed RL episode.
        self._node.create_subscription(
            Float64,
            f"{joint_prefix}/rudder_joint/cmd_pos",
            lambda msg: self._on_final("rudder", msg),
            10,
        )
        self._node.create_subscription(
            Float64,
            f"{joint_prefix}/sail_joint/cmd_pos",
            lambda msg: self._on_final("sail", msg),
            10,
        )

        self._pose: dict[str, float] | None = None
        self._base = {"rudder": None, "sail": None}
        self._final = {"rudder": None, "sail": None}
        self._residual = {"rudder": 0.0, "sail": 0.0}
        self._waypoint_index = 0
        self._within_capture_radius_since_s: float | None = None
        self._mission_complete = False
        self._closed = False

    def _set_adapter_enabled(
        self,
        enabled: bool,
        *,
        allow_missing_service: bool = False,
    ) -> bool:
        if not self.manage_adapter_enabled:
            return False
        if not self._adapter_parameters.wait_for_services(
            timeout_sec=self.telemetry_timeout_s
        ):
            if allow_missing_service:
                return False
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
        if not future.done():
            raise TimeoutError("timed out setting residual_enabled")
        response = future.result()
        if response is None:
            raise TimeoutError("timed out setting residual_enabled")
        results = response.results
        if not results or not all(result.successful for result in results):
            reasons = ", ".join(
                result.reason for result in results if not result.successful
            )
            raise RuntimeError(
                f"command adapter rejected residual_enabled={enabled}: {reasons}"
            )
        return True

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

    def _on_final(self, surface: str, msg: Any) -> None:
        self._final[surface] = float(msg.data)

    def _spin_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            if (
                self._pose is not None
                and self._base["rudder"] is not None
                and self._base["sail"] is not None
                and self._final["rudder"] is not None
                and self._final["sail"] is not None
            ):
                return
        raise TimeoutError(
            "timed out waiting for odometry and adapter actuator commands"
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
        if distance > self.capture_radius_m:
            self._within_capture_radius_since_s = None
            return

        sim_time_s = float(self._pose["sim_time_s"])
        if self._within_capture_radius_since_s is None:
            self._within_capture_radius_since_s = sim_time_s
        held_s = max(0.0, sim_time_s - self._within_capture_radius_since_s)
        if held_s + 1e-9 < self.capture_hold_s:
            return

        if self._waypoint_index < len(self.waypoints) - 1:
            self._waypoint_index += 1
            self._within_capture_radius_since_s = None
        else:
            self._mission_complete = True

    def _cross_track_error_m(self) -> float:
        """Return BO-compatible distance to the active finite path segment."""
        if self._pose is None:
            raise RuntimeError("odometry is not available")
        start_x, start_y = self.path_points[self._waypoint_index]
        end_x, end_y = self.path_points[self._waypoint_index + 1]
        cross_track_m, _along_track_m = segment_metrics(
            self._pose["x_m"],
            self._pose["y_m"],
            start_x,
            start_y,
            end_x,
            end_y,
        )
        return float(cross_track_m)

    def _state(self) -> RawState:
        if self._pose is None:
            raise RuntimeError("odometry is not available")
        self._update_waypoint()
        target_x, target_y = self.waypoints[self._waypoint_index]
        base_rudder = float(self._base["rudder"])
        base_sail = float(self._base["sail"])
        final_rudder = float(self._final["rudder"])
        final_sail = float(self._final["sail"])
        return RawState(
            **self._pose,
            target_x_m=target_x,
            target_y_m=target_y,
            wind_x_mps=self.wind_x_mps,
            wind_y_mps=self.wind_y_mps,
            base_rudder_rad=base_rudder,
            base_sail_rad=base_sail,
            # Preserve the existing observation semantics: this is the
            # requested residual published by the environment. Per-step CSV
            # telemetry derives the actual adapter contribution from the
            # observed final and base commands below.
            residual_rudder_rad=self._residual["rudder"],
            residual_sail_rad=self._residual["sail"],
            cross_track_error_m=self._cross_track_error_m(),
            waypoint_index=self._waypoint_index,
            waypoint_count=len(self.waypoints),
            mission_complete=self._mission_complete,
            final_rudder_rad=final_rudder,
            final_sail_rad=final_sail,
        )

    def reset(self, *, seed: int | None, options: dict[str, Any]) -> RawState:
        del seed
        if options.get("waypoint_index") is not None:
            requested = int(options["waypoint_index"])
            self._waypoint_index = min(max(requested, 0), len(self.waypoints) - 1)
        else:
            self._waypoint_index = 0
        self._within_capture_radius_since_s = None
        self._mission_complete = False
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
        cleanup_error: Exception | None = None
        try:
            self._publish_residuals(0.0, 0.0)
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        except Exception as exc:
            cleanup_error = exc

        try:
            # A naturally completed BO trial may already have removed the
            # command-adapter node before Gymnasium calls the next reset().
            # In that case there is no live adapter left to disable, so a
            # missing service is an already-clean state rather than an error.
            self._set_adapter_enabled(False, allow_missing_service=True)
        except TimeoutError:
            # The service can disappear between discovery and the parameter
            # response while run_trial.py tears down the ROS launch process.
            pass
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        finally:
            try:
                self._node.destroy_node()
            finally:
                if self._owns_rclpy and self._rclpy.ok():
                    self._rclpy.shutdown()
                self._closed = True

        if cleanup_error is not None:
            raise cleanup_error
