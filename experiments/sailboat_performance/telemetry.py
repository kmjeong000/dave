"""Synchronized ROS and MAVLink telemetry for sailing-performance trials."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from experiments.sailboat_physics.frames import (
    gazebo_quaternion_to_frd_attitude_deg,
)

from .core import (
    angle_error_deg,
    course_over_ground_deg,
    heading_rate_deg_s,
    model_velocity_to_world_enu,
    normalize_degrees_360,
)


def message_stamp_s(msg: Any) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    return float(getattr(stamp, "sec", 0)) + 1e-9 * float(
        getattr(stamp, "nanosec", 0)
    )


def extract_joint_state(
    names: list[str],
    positions: list[float],
    velocities: list[float],
    joint_name: str,
) -> tuple[float, float | None] | None:
    """Extract one named joint without assuming message ordering."""

    matches = [
        index
        for index, name in enumerate(names)
        if name == joint_name or name.endswith(f"::{joint_name}")
    ]
    if len(matches) != 1:
        return None
    index = matches[0]
    if index >= len(positions):
        return None
    velocity = float(velocities[index]) if index < len(velocities) else None
    return float(positions[index]), velocity


def telemetry_ages_are_fresh(
    ages_s: tuple[float | None, ...], maximum_age_s: float
) -> bool:
    """Return whether every source age is finite and inside the stale gate."""

    maximum = float(maximum_age_s)
    return math.isfinite(maximum) and maximum >= 0.0 and all(
        age is not None
        and math.isfinite(float(age))
        and 0.0 <= float(age) <= maximum
        for age in ages_s
    )


@dataclass(frozen=True)
class RosPerformanceSnapshot:
    sim_time_s: float | None = None
    x_east_m: float | None = None
    y_north_m: float | None = None
    z_up_m: float | None = None
    roll_frd_deg: float | None = None
    pitch_frd_deg: float | None = None
    heading_deg: float | None = None
    derived_yaw_rate_deg_s: float | None = None
    gazebo_angular_yaw_rate_deg_s: float | None = None
    model_vx_mps: float | None = None
    model_vy_mps: float | None = None
    model_vz_mps: float | None = None
    velocity_east_mps: float | None = None
    velocity_north_mps: float | None = None
    velocity_up_mps: float | None = None
    speed_over_ground_mps: float | None = None
    course_over_ground_deg: float | None = None
    leeway_deg: float | None = None
    actual_rudder_angle_rad: float | None = None
    actual_rudder_velocity_rad_s: float | None = None
    actual_boom_angle_rad: float | None = None
    actual_boom_velocity_rad_s: float | None = None
    base_rudder_rad: float | None = None
    final_rudder_rad: float | None = None
    base_sheet_allowance_rad: float | None = None
    final_sheet_allowance_rad: float | None = None
    odom_frame_id: str = ""
    odom_child_frame_id: str = ""
    odom_age_s: float | None = None
    rudder_age_s: float | None = None
    boom_age_s: float | None = None
    actuator_age_s: float | None = None


class PerformanceRosCollector:
    """Collect pose, body twist, joint states and actuator commands."""

    def __init__(self, namespace: str, world_name: str):
        self.namespace = namespace.strip("/")
        self.world_name = world_name
        self._lock = threading.Lock()
        self._snapshot = RosPerformanceSnapshot()
        self._received_wall: dict[str, float] = {}
        self._node: Any = None
        self._rclpy: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._did_init = False
        self._previous_heading_sample: tuple[float, float] | None = None

    def start(self) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64
        except ImportError as exc:  # pragma: no cover - live dependency guard
            raise RuntimeError("performance telemetry requires ROS 2 Python messages") from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._did_init = True
        self._node = rclpy.create_node(
            f"sailboat_performance_{os.getpid()}_{int(time.time() * 1000)}"
        )
        topics = {
            "odom": f"/model/{self.namespace}/odometry",
            "joint": f"/world/{self.world_name}/model/{self.namespace}/joint_state",
            "base_rudder": f"/model/{self.namespace}/joint/rudder_joint/base_cmd_pos",
            "final_rudder": f"/model/{self.namespace}/joint/rudder_joint/cmd_pos",
            "base_sail": f"/model/{self.namespace}/joint/sail_joint/base_cmd_pos",
            "final_sail": f"/model/{self.namespace}/joint/sail_joint/cmd_pos",
        }
        self._node.create_subscription(Odometry, topics["odom"], self._on_odom, 20)
        self._node.create_subscription(JointState, topics["joint"], self._on_joint, 20)
        self._node.create_subscription(
            Float64, topics["base_rudder"], lambda msg: self._set_command("base_rudder_rad", msg), 20
        )
        self._node.create_subscription(
            Float64, topics["final_rudder"], lambda msg: self._set_command("final_rudder_rad", msg), 20
        )
        self._node.create_subscription(
            Float64, topics["base_sail"], lambda msg: self._set_command("base_sheet_allowance_rad", msg), 20
        )
        self._node.create_subscription(
            Float64, topics["final_sail"], lambda msg: self._set_command("final_sheet_allowance_rad", msg), 20
        )
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print("[polar_trial] ROS telemetry: " + ", ".join(topics.values()))

    def _spin(self) -> None:
        while not self._stop.is_set():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def _on_odom(self, msg: Any) -> None:
        pose = msg.pose.pose
        linear_twist = msg.twist.twist.linear
        angular_twist = msg.twist.twist.angular
        sim_time_s = message_stamp_s(msg)
        quaternion = (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        attitude = gazebo_quaternion_to_frd_attitude_deg(*quaternion)
        east, north, up = model_velocity_to_world_enu(
            float(linear_twist.x),
            float(linear_twist.y),
            float(linear_twist.z),
            quaternion,
        )
        speed = math.hypot(east, north)
        course = course_over_ground_deg(east, north)
        leeway = (
            angle_error_deg(course, attitude.heading_deg)
            if course is not None
            else None
        )
        yaw_rate = None
        if sim_time_s is not None and self._previous_heading_sample is not None:
            previous_time_s, previous_heading_deg = self._previous_heading_sample
            if sim_time_s > previous_time_s:
                yaw_rate = heading_rate_deg_s(
                    previous_heading_deg,
                    attitude.heading_deg,
                    sim_time_s - previous_time_s,
                )
        if sim_time_s is not None:
            self._previous_heading_sample = (sim_time_s, attitude.heading_deg)
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                sim_time_s=sim_time_s,
                x_east_m=float(pose.position.x),
                y_north_m=float(pose.position.y),
                z_up_m=float(pose.position.z),
                roll_frd_deg=attitude.roll_deg,
                pitch_frd_deg=attitude.pitch_deg,
                heading_deg=attitude.heading_deg,
                derived_yaw_rate_deg_s=yaw_rate,
                # The 3-D Gazebo OdometryPublisher independently derives and
                # publishes body/model angular velocity from successive poses.
                # Model +Z is up while body FRD +Z is down, so this conversion
                # is clockwise-positive like MAVLink yawspeed.
                gazebo_angular_yaw_rate_deg_s=math.degrees(
                    -float(angular_twist.z)
                ),
                model_vx_mps=float(linear_twist.x),
                model_vy_mps=float(linear_twist.y),
                model_vz_mps=float(linear_twist.z),
                velocity_east_mps=east,
                velocity_north_mps=north,
                velocity_up_mps=up,
                speed_over_ground_mps=speed,
                course_over_ground_deg=course,
                leeway_deg=leeway,
                odom_frame_id=str(msg.header.frame_id),
                odom_child_frame_id=str(msg.child_frame_id),
            )
            self._received_wall["odom"] = time.monotonic()

    def _on_joint(self, msg: Any) -> None:
        names = list(msg.name)
        positions = list(msg.position)
        velocities = list(msg.velocity)
        boom = extract_joint_state(names, positions, velocities, "sail_joint")
        rudder = extract_joint_state(names, positions, velocities, "rudder_joint")
        if boom is None and rudder is None:
            return
        received_at = time.monotonic()
        updates: dict[str, float | None] = {}
        if boom is not None:
            updates["actual_boom_angle_rad"] = boom[0]
            updates["actual_boom_velocity_rad_s"] = boom[1]
        if rudder is not None:
            updates["actual_rudder_angle_rad"] = rudder[0]
            updates["actual_rudder_velocity_rad_s"] = rudder[1]
        with self._lock:
            self._snapshot = replace(self._snapshot, **updates)
            if boom is not None:
                self._received_wall["boom"] = received_at
            if rudder is not None:
                self._received_wall["rudder"] = received_at

    def _set_command(self, field_name: str, msg: Any) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, **{field_name: float(msg.data)})
            self._received_wall[field_name] = time.monotonic()

    def snapshot(self) -> RosPerformanceSnapshot:
        # Copy values and receipt times under one lock. Read the reference
        # clock only after that copy, so an asynchronous callback cannot give
        # copied telemetry a timestamp newer than its snapshot reference.
        with self._lock:
            snapshot = self._snapshot
            received_wall = dict(self._received_wall)
        now = time.monotonic()
        commands = (
            "base_rudder_rad",
            "final_rudder_rad",
            "base_sheet_allowance_rad",
            "final_sheet_allowance_rad",
        )
        command_times = [received_wall.get(name) for name in commands]
        actuator_age = (
            max(now - stamp for stamp in command_times if stamp is not None)
            if all(stamp is not None for stamp in command_times)
            else None
        )
        return replace(
            snapshot,
            odom_age_s=(
                now - received_wall["odom"] if "odom" in received_wall else None
            ),
            rudder_age_s=(
                now - received_wall["rudder"]
                if "rudder" in received_wall
                else None
            ),
            boom_age_s=(
                now - received_wall["boom"] if "boom" in received_wall else None
            ),
            actuator_age_s=actuator_age,
        )

    def wait_ready(self, timeout_s: float, maximum_age_s: float) -> RosPerformanceSnapshot:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if (
                snapshot.sim_time_s is not None
                and telemetry_ages_are_fresh(
                    (
                        snapshot.odom_age_s,
                        snapshot.rudder_age_s,
                        snapshot.boom_age_s,
                        snapshot.actuator_age_s,
                    ),
                    maximum_age_s,
                )
            ):
                return snapshot
            time.sleep(0.1)
        raise TimeoutError(
            "timed out waiting for odometry, rudder, boom and actuator telemetry"
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._did_init and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


@dataclass
class MavlinkPerformanceState:
    wind_from_deg: float | None = None
    wind_speed_mps: float | None = None
    wind_speed_z_mps: float | None = None
    wind_received_monotonic_s: float | None = None
    yaw_deg: float | None = None
    yaw_rate_deg_s: float | None = None
    attitude_received_monotonic_s: float | None = None
    heartbeat_custom_mode: int | None = None
    armed: bool = False
    status_text: str = ""

    def update(self, msg: Any, mavlink: Any, now_monotonic_s: float) -> None:
        msg_type = msg.get_type()
        if msg_type == "WIND":
            self.wind_from_deg = float(msg.direction) % 360.0
            self.wind_speed_mps = float(msg.speed)
            self.wind_speed_z_mps = float(getattr(msg, "speed_z", 0.0))
            self.wind_received_monotonic_s = float(now_monotonic_s)
        elif msg_type == "ATTITUDE":
            # MAVLink ATTITUDE is aeronautical FRD: yaw is the NED/compass
            # angle and yawspeed is positive about body +Z (down).
            self.yaw_deg = normalize_degrees_360(math.degrees(float(msg.yaw)))
            self.yaw_rate_deg_s = math.degrees(float(msg.yawspeed))
            self.attitude_received_monotonic_s = float(now_monotonic_s)
        elif msg_type == "HEARTBEAT":
            self.heartbeat_custom_mode = int(msg.custom_mode)
            self.armed = bool(
                int(msg.base_mode) & int(mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            )
        elif msg_type == "STATUSTEXT":
            self.status_text = str(getattr(msg, "text", "")).strip()

    def wind_age_s(self, now_monotonic_s: float) -> float | None:
        if self.wind_received_monotonic_s is None:
            return None
        return float(now_monotonic_s) - self.wind_received_monotonic_s

    def attitude_age_s(self, now_monotonic_s: float) -> float | None:
        if self.attitude_received_monotonic_s is None:
            return None
        return float(now_monotonic_s) - self.attitude_received_monotonic_s


def drain_mavlink(master: Any, mavlink: Any, state: MavlinkPerformanceState) -> int:
    count = 0
    while True:
        msg = master.recv_match(
            type=["WIND", "ATTITUDE", "HEARTBEAT", "STATUSTEXT"],
            blocking=False,
        )
        if msg is None:
            return count
        state.update(msg, mavlink, time.monotonic())
        count += 1
