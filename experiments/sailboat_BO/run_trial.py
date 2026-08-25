from __future__ import annotations

import argparse
import codecs
import json
import math
import os
import shutil
import shlex
import signal
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    from .common import (
        DEFAULT_PARAMS_PATH,
        DEFAULT_RESULTS_DIR,
        DEFAULT_SCENARIO_PATH,
        clip,
        load_yaml,
    )
    from .score import (
        append_summary_row,
        build_summary_columns,
        build_summary_row,
        compute_constraints,
        compute_metrics,
        compute_objective,
    )
except ImportError:
    from common import (
        DEFAULT_PARAMS_PATH,
        DEFAULT_RESULTS_DIR,
        DEFAULT_SCENARIO_PATH,
        clip,
        load_yaml,
    )
    from score import (
        append_summary_row,
        build_summary_columns,
        build_summary_row,
        compute_constraints,
        compute_metrics,
        compute_objective,
    )

EARTH_RADIUS_M = 6378137.0
REQUEST_RETRIES = 5
DEFAULT_CONTAINER_REPO_ROOT = "/home/docker/sailboat_ws/src/dave"

MESSAGE_INTERVAL_HZ = {
    "SYS_STATUS": 2.0,
    "GLOBAL_POSITION_INT": 10.0,
    "ATTITUDE": 10.0,
    "VFR_HUD": 10.0,
    "SERVO_OUTPUT_RAW": 10.0,
    "NAV_CONTROLLER_OUTPUT": 10.0,
    "MISSION_CURRENT": 5.0,
    "MISSION_ITEM_REACHED": 5.0,
}

MESSAGE_NAME_TO_ID = {
    "SYS_STATUS": 1,
    "GLOBAL_POSITION_INT": 33,
    "ATTITUDE": 30,
    "VFR_HUD": 74,
    "SERVO_OUTPUT_RAW": 36,
    "NAV_CONTROLLER_OUTPUT": 62,
    "MISSION_CURRENT": 42,
    "MISSION_ITEM_REACHED": 46,
}

SUPPORTED_MISSION_FRAMES = {"local_enu_m", "gazebo_xy_m"}


@dataclass(frozen=True)
class TrialFiles:
    trial_dir: Path
    logs_dir: Path
    raw_dir: Path
    summary_dir: Path
    manifest_path: Path
    param_file: Path
    world_file: Path
    samples_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class TrialContext:
    study_name: str
    scenario_id: str
    scenario_split: str
    trial_id: str
    repeat_idx: int
    started_at_utc: str
    repo_root: Path
    workspace_root: Path
    world_template_dir: Path
    launch_file: Path
    results_dir: Path
    files: TrialFiles
    params: dict[str, float]
    param_names: list[str]
    study_cfg: dict[str, Any]
    scenario_cfg: dict[str, Any]
    termination_cfg: dict[str, Any]
    logging_cfg: dict[str, Any]


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str
    docker_container: str | None = None
    container_repo_root: str | None = None


@dataclass(frozen=True)
class LaunchPlan:
    backend: str
    launch_cmd: list[str]
    process_cmd: list[str]
    display_cmd: str
    cleanup_token: str | None = None


@dataclass(frozen=True)
class MissionWaypoint:
    seq: int
    x_m: float
    y_m: float


@dataclass(frozen=True)
class MissionUploadItem:
    raw_seq: int
    x_m: float
    y_m: float
    is_home: bool
    user_waypoint_index: int | None


@dataclass
class TelemetryState:
    sim_time_s: float | None = None
    lat_deg: float | None = None
    lon_deg: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    yaw_rad: float | None = None
    heading_deg: float | None = None
    surge_speed_mps: float | None = None
    roll_deg: float | None = None
    rudder_cmd_rad: float | None = None
    sail_cmd_rad: float | None = None
    mission_seq: int = 0
    reached_seq: int = -1
    nav_wp_dist_m: float | None = None
    nav_xtrack_error_m: float | None = None


@dataclass
class RosTelemetrySnapshot:
    odom_roll_deg: float | None = None
    odom_x_m: float | None = None
    odom_y_m: float | None = None
    odom_yaw_rad: float | None = None
    odom_speed_mps: float | None = None
    imu_roll_deg: float | None = None


@dataclass
class WaypointTracker:
    completed_waypoint_count: int = 0
    local_capture_count: int = 0
    within_capture_radius_since_s: float | None = None


class RosTelemetryCollector:
    def __init__(self, namespace: str, *, use_odom: bool, use_imu: bool):
        self.namespace = namespace
        self.use_odom = bool(use_odom)
        self.use_imu = bool(use_imu)
        self._lock = threading.Lock()
        self._snapshot = RosTelemetrySnapshot()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._node = None
        self._rclpy = None
        self._did_init = False

    def start(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError as exc:
            raise RuntimeError(
                "ROS telemetry collector requires rclpy in the current environment"
            ) from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._did_init = True

        node_name = f"sailboat_bo_{os.getpid()}_{int(time.time() * 1000)}"
        self._node = Node(node_name)
        subscribed_topics: list[str] = []
        if self.use_odom:
            try:
                from nav_msgs.msg import Odometry
            except ImportError as exc:
                raise RuntimeError(
                    "ROS telemetry collector requires nav_msgs for odometry roll collection"
                ) from exc
            odom_topic = f"/model/{self.namespace}/odometry"
            self._node.create_subscription(Odometry, odom_topic, self._on_odometry, 10)
            subscribed_topics.append(f"odom={odom_topic}")
        if self.use_imu:
            try:
                from sensor_msgs.msg import Imu
            except ImportError as exc:
                raise RuntimeError(
                    "ROS telemetry collector requires sensor_msgs for IMU roll collection"
                ) from exc
            imu_topic = f"/model/{self.namespace}/imu"
            self._node.create_subscription(Imu, imu_topic, self._on_imu, 10)
            subscribed_topics.append(f"imu={imu_topic}")
        if not subscribed_topics:
            raise RuntimeError("ROS telemetry collector was started without any requested topics")
        self._thread = threading.Thread(target=self._spin_loop, name=node_name, daemon=True)
        self._thread.start()
        print("[run_trial] subscribed to ROS telemetry: " + ", ".join(subscribed_topics))

    def _spin_loop(self) -> None:
        assert self._node is not None
        assert self._rclpy is not None
        while not self._stop_event.is_set():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def _on_odometry(self, msg: Any) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        roll_deg = quaternion_to_roll_deg(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        yaw_rad = quaternion_to_yaw_rad(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        linear = msg.twist.twist.linear
        speed_mps = math.hypot(float(linear.x), float(linear.y))
        with self._lock:
            self._snapshot.odom_roll_deg = roll_deg
            self._snapshot.odom_x_m = float(position.x)
            self._snapshot.odom_y_m = float(position.y)
            self._snapshot.odom_yaw_rad = yaw_rad
            self._snapshot.odom_speed_mps = speed_mps

    def _on_imu(self, msg: Any) -> None:
        orientation = msg.orientation
        roll_deg = quaternion_to_roll_deg(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        with self._lock:
            self._snapshot.imu_roll_deg = roll_deg

    def snapshot(self) -> RosTelemetrySnapshot:
        with self._lock:
            return RosTelemetrySnapshot(
                odom_roll_deg=self._snapshot.odom_roll_deg,
                odom_x_m=self._snapshot.odom_x_m,
                odom_y_m=self._snapshot.odom_y_m,
                odom_yaw_rad=self._snapshot.odom_yaw_rad,
                odom_speed_mps=self._snapshot.odom_speed_mps,
                imu_roll_deg=self._snapshot.imu_roll_deg,
            )

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._did_init and self._rclpy is not None and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception:
                pass


def mission_seq_home_offset_enabled(context: TrialContext) -> bool:
    # ArduPilot commonly reports mission seq with HOME as raw seq 0.
    return bool(context.study_cfg.get("mission_seq_home_offset", True))


def mission_seq_to_waypoint_index(context: TrialContext, raw_seq: int, waypoint_count: int) -> int:
    if waypoint_count <= 0:
        return 0
    offset = 1 if mission_seq_home_offset_enabled(context) else 0
    return int(clip(float(raw_seq - offset), 0.0, float(waypoint_count - 1)))


def waypoint_index_to_mission_raw_seq(context: TrialContext, waypoint_index: int, waypoint_count: int) -> int:
    if waypoint_count <= 0:
        return 0
    clamped_index = int(clip(float(waypoint_index), 0.0, float(waypoint_count - 1)))
    offset = 1 if mission_seq_home_offset_enabled(context) else 0
    return clamped_index + offset


def final_mission_raw_seq(context: TrialContext, waypoint_count: int) -> int:
    if waypoint_count <= 0:
        return -1
    return waypoint_count if mission_seq_home_offset_enabled(context) else waypoint_count - 1


def normalize_mission_reached_seq(
    context: TrialContext,
    raw_reached_seq: int,
    waypoint_count: int,
) -> int | None:
    if waypoint_count <= 0 or raw_reached_seq < 0:
        return None
    max_valid_seq = final_mission_raw_seq(context, waypoint_count)
    if raw_reached_seq > max_valid_seq:
        return None
    return raw_reached_seq


def reached_final_waypoint(context: TrialContext, raw_reached_seq: int, waypoint_count: int) -> bool:
    normalized_seq = normalize_mission_reached_seq(context, raw_reached_seq, waypoint_count)
    if normalized_seq is None:
        return False
    return normalized_seq == final_mission_raw_seq(context, waypoint_count)


def completed_waypoint_count_from_reached_seq(
    context: TrialContext,
    raw_reached_seq: int,
    waypoint_count: int,
) -> int:
    normalized_seq = normalize_mission_reached_seq(context, raw_reached_seq, waypoint_count)
    if normalized_seq is None:
        return 0
    offset = 0 if mission_seq_home_offset_enabled(context) else 1
    return int(clip(float(normalized_seq + offset), 0.0, float(waypoint_count)))


def effective_waypoint_index(waypoint_count: int, completed_waypoint_count: int) -> int:
    if waypoint_count <= 0:
        return 0
    return int(clip(float(completed_waypoint_count), 0.0, float(waypoint_count - 1)))


def sync_waypoint_tracker(
    tracker: WaypointTracker,
    *,
    raw_waypoint_index: int,
    completed_waypoint_count_from_reached: int,
    waypoint_count: int,
) -> None:
    if waypoint_count <= 0:
        return
    tracker.completed_waypoint_count = max(
        tracker.completed_waypoint_count,
        int(clip(float(raw_waypoint_index), 0.0, float(waypoint_count - 1))),
        int(clip(float(completed_waypoint_count_from_reached), 0.0, float(waypoint_count))),
    )


def maybe_capture_local_waypoint(
    tracker: WaypointTracker,
    *,
    waypoint_count: int,
    sim_time_s: float,
    distance_to_wp_m: float,
    capture_radius_m: float,
    capture_hold_s: float,
) -> bool:
    if waypoint_count <= 0 or tracker.completed_waypoint_count >= waypoint_count:
        return False
    if distance_to_wp_m > capture_radius_m:
        tracker.within_capture_radius_since_s = None
        return False

    if tracker.within_capture_radius_since_s is None:
        tracker.within_capture_radius_since_s = sim_time_s
        if capture_hold_s > 0.0:
            return False

    held_time_s = sim_time_s - tracker.within_capture_radius_since_s
    if capture_hold_s > 0.0 and held_time_s < capture_hold_s:
        return False

    tracker.completed_waypoint_count += 1
    tracker.local_capture_count += 1
    tracker.within_capture_radius_since_s = None
    return True


def compute_capture_distance_m(
    *,
    local_distance_to_wp_m: float,
    nav_wp_dist_m: float | None,
    raw_waypoint_index: int,
    effective_target_index: int,
    use_nav_wp_dist: bool = True,
) -> float:
    capture_distance_m = max(0.0, float(local_distance_to_wp_m))
    if not use_nav_wp_dist:
        return capture_distance_m
    if nav_wp_dist_m is None:
        return capture_distance_m
    if raw_waypoint_index != effective_target_index:
        return capture_distance_m
    if float(nav_wp_dist_m) < 0.0:
        return capture_distance_m
    # Be conservative when ArduPilot is still tracking the same mission item.
    return max(capture_distance_m, float(nav_wp_dist_m))


def selected_roll_source(context: TrialContext) -> str:
    raw_value = str(context.termination_cfg.get("roll_source", "mavlink")).strip().lower()
    valid_sources = {"mavlink", "gazebo_odometry", "gazebo_imu"}
    if raw_value not in valid_sources:
        raise ValueError(f"Unsupported roll_source '{raw_value}'. Expected one of: {sorted(valid_sources)}")
    return raw_value


def ros_roll_collection_required(context: TrialContext) -> bool:
    use_odom, use_imu = ros_roll_topic_requirements(context)
    return use_odom or use_imu


def ros_roll_topic_requirements(context: TrialContext) -> tuple[bool, bool]:
    roll_source = selected_roll_source(context)
    csv_fields = {str(field) for field in context.logging_cfg.get("csv_fields", [])}
    use_odom = (
        roll_source == "gazebo_odometry"
        or selected_position_source(context) == "gazebo_odometry"
        or "roll_gz_odom_deg" in csv_fields
        or "x_gz_odom_m" in csv_fields
        or "y_gz_odom_m" in csv_fields
        or "yaw_gz_odom_rad" in csv_fields
        or "surge_speed_gz_odom_mps" in csv_fields
    )
    use_imu = roll_source == "gazebo_imu" or "roll_gz_imu_deg" in csv_fields
    return use_odom, use_imu


def quaternion_to_roll_deg(x: float, y: float, z: float, w: float) -> float:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.atan2(sinr_cosp, cosr_cosp))


def quaternion_to_yaw_rad(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def normalize_degrees_360(value: float) -> float:
    return float(value) % 360.0


def normalize_degrees_signed(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-9):
        return 180.0
    return normalized


def selected_ardupilot_home_heading_deg(spawn: dict[str, Any]) -> float:
    if "heading_deg" in spawn:
        return normalize_degrees_360(float(spawn["heading_deg"]))
    return normalize_degrees_360(90.0 - float(spawn["yaw_deg"]))


def selected_gazebo_spawn_yaw_deg(spawn: dict[str, Any]) -> float:
    if "gazebo_yaw_deg" in spawn:
        return normalize_degrees_signed(float(spawn["gazebo_yaw_deg"]))
    if "heading_deg" in spawn:
        return normalize_degrees_signed(-float(spawn["heading_deg"]))
    # Legacy scenario yaw is a Gazebo XY course angle. The sailboat hull points along model +Y.
    return normalize_degrees_signed(float(spawn["yaw_deg"]) - 90.0)


def selected_position_source(context: TrialContext) -> str:
    raw_value = str(context.termination_cfg.get("position_source", "gazebo_odometry")).strip().lower()
    valid_sources = {"mavlink", "gazebo_odometry"}
    if raw_value not in valid_sources:
        raise ValueError(
            f"Unsupported position_source '{raw_value}'. Expected one of: {sorted(valid_sources)}"
        )
    return raw_value


def trust_mavlink_reached_for_completion(context: TrialContext) -> bool:
    return selected_position_source(context) == "mavlink"


def mavlink_reached_local_gate_m(context: TrialContext) -> float:
    configured = context.termination_cfg.get("mavlink_reached_local_gate_m")
    if configured is not None:
        return max(0.0, float(configured))
    success_radius_m = float(context.termination_cfg.get("success_radius_m", 5.0))
    capture_radius_m = float(
        context.termination_cfg.get("waypoint_capture_radius_m", success_radius_m)
    )
    return max(success_radius_m, capture_radius_m) + 1.0


def accept_mavlink_reached_for_completion(
    context: TrialContext,
    *,
    raw_reached_seq: int,
    waypoint_count: int,
    distance_to_wp_m: float,
) -> bool:
    if not reached_final_waypoint(context, raw_reached_seq, waypoint_count):
        return False
    if trust_mavlink_reached_for_completion(context):
        return True
    if selected_position_source(context) != "gazebo_odometry":
        return False
    if not math.isfinite(float(distance_to_wp_m)):
        return False
    return float(distance_to_wp_m) <= mavlink_reached_local_gate_m(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one sailboat BO trial")
    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO_PATH),
        help="Path to scenario.yaml",
    )
    parser.add_argument("--scenario-id", required=True, help="Scenario id to run")
    params_group = parser.add_mutually_exclusive_group()
    params_group.add_argument(
        "--params",
        help="JSON object of parameter overrides, e.g. '{\"SAIL_ANGLE_IDEAL\": 35.0}'",
    )
    params_group.add_argument(
        "--params-file",
        help="Path to a JSON file containing parameter overrides. Defaults to baseline_params.json if omitted.",
    )
    parser.add_argument("--repeat-idx", type=int, default=0, help="Repeat index")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where trial artifacts and summary.csv are stored",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["local", "docker-exec"],
        help="Execution backend override. `local` launches directly in the current environment, "
        "`docker-exec` launches Gazebo + ArduPilot inside an already-running container while "
        "this Python process stays on the host and logs MAVLink telemetry locally.",
    )
    parser.add_argument(
        "--docker-container",
        help="Container name used when --execution-backend docker-exec is selected",
    )
    parser.add_argument(
        "--container-repo-root",
        help="Repository root path inside the container for docker-exec mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate world/parm/manifest files and print commands without launching the sim",
    )
    parser.add_argument(
        "--lifecycle-ready-file",
        help="Internal RL lifecycle marker written after mission upload, arm, and AUTO readiness",
    )
    parser.add_argument(
        "--lifecycle-stop-file",
        help="Internal RL lifecycle stop request polled while the trial is running",
    )
    parser.add_argument(
        "--lifecycle-status-file",
        help="Internal RL lifecycle marker containing the terminal trial outcome",
    )
    return parser.parse_args()


def write_json_marker(path: Path | None, payload: dict[str, Any]) -> None:
    """Atomically publish a small lifecycle marker for an external controller."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_lifecycle_stop_reason(path: Path | None) -> str | None:
    """Return a cooperative stop reason when the lifecycle controller requests one."""
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "external_stop"
    if isinstance(payload, dict):
        reason = str(payload.get("reason", "external_stop")).strip()
        return reason or "external_stop"
    return "external_stop"


def get_search_space_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in config.get("search_space", [])}


def get_scenario(config: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    for scenario in config.get("scenarios", []):
        if scenario.get("id") == scenario_id:
            return scenario
    available = ", ".join(s.get("id", "<unknown>") for s in config.get("scenarios", []))
    raise KeyError(f"Scenario '{scenario_id}' not found. Available: {available}")


def parse_param_overrides(raw_params: str) -> dict[str, float]:
    parsed = json.loads(raw_params)
    if not isinstance(parsed, dict):
        raise ValueError("--params must decode to a JSON object")
    return {str(key): float(value) for key, value in parsed.items()}


def load_param_overrides_from_file(path: str | Path) -> dict[str, float]:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("--params-file must contain a JSON object")
    return {str(key): float(value) for key, value in parsed.items()}


def validate_param_overrides(
    search_space_map: dict[str, dict[str, Any]],
    params: dict[str, float],
) -> list[str]:
    errors: list[str] = []
    for name, value in params.items():
        if name not in search_space_map:
            errors.append(f"Unknown parameter '{name}'")
            continue
        bounds = search_space_map[name]
        low = float(bounds["low"])
        high = float(bounds["high"])
        if value < low or value > high:
            errors.append(f"{name}={value} is outside [{low}, {high}]")
    return errors


def build_trial_id(scenario_id: str, repeat_idx: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}__{scenario_id}__r{repeat_idx:02d}"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_workspace_root(repo_root: Path) -> Path:
    if repo_root.name == "src":
        return repo_root.parent
    if repo_root.parent.name == "src":
        return repo_root.parent.parent
    return repo_root.parent


def resolve_repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def merge_mapping(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def selected_execution_backend(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.execution_backend:
        return args.execution_backend
    return str(config["study"].get("execution_backend", "local"))


def selected_docker_container(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.docker_container:
        return args.docker_container
    return str(config["study"].get("docker_container", "rover-debug-shell"))


def selected_container_repo_root(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.container_repo_root:
        return args.container_repo_root
    return str(config["study"].get("container_repo_root", DEFAULT_CONTAINER_REPO_ROOT))


def build_execution_config(args: argparse.Namespace, config: dict[str, Any]) -> ExecutionConfig:
    backend = selected_execution_backend(args, config)
    if backend == "local":
        return ExecutionConfig(backend="local")
    return ExecutionConfig(
        backend=backend,
        docker_container=selected_docker_container(args, config),
        container_repo_root=selected_container_repo_root(args, config),
    )


def validate_execution_backend(execution: ExecutionConfig) -> None:
    if execution.backend != "docker-exec":
        return
    if shutil.which("docker") is not None:
        return

    running_inside_container = Path("/.dockerenv").exists()
    if running_inside_container:
        raise RuntimeError(
            "execution_backend=docker-exec was selected from inside a Docker container, "
            "but the `docker` CLI is not available here.\n"
            "If you are already inside the simulation container, rerun with:\n"
            "  --execution-backend local\n"
            "and remove --docker-container / --container-repo-root."
        )

    raise RuntimeError(
        "execution_backend=docker-exec requires the `docker` CLI on the current machine, "
        "but it was not found in PATH."
    )


def host_path_to_repo_relative(repo_root: Path, path: Path) -> Path:
    resolved_repo_root = repo_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside repo root and cannot be mirrored into the container: {path}") from exc


def repo_relative_to_container_path(container_repo_root: str, relative_path: Path) -> str:
    return str(PurePosixPath(container_repo_root) / PurePosixPath(relative_path.as_posix()))


def make_trial_files(results_dir: Path, world_output_dir: Path, trial_id: str) -> TrialFiles:
    trial_dir = results_dir / trial_id
    logs_dir = trial_dir / "logs"
    raw_dir = trial_dir / "raw"
    summary_dir = trial_dir / "summary"
    for directory in (trial_dir, logs_dir, raw_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return TrialFiles(
        trial_dir=trial_dir,
        logs_dir=logs_dir,
        raw_dir=raw_dir,
        summary_dir=summary_dir,
        manifest_path=trial_dir / "trial_manifest.json",
        param_file=trial_dir / "trial.parm",
        world_file=world_output_dir / f"{trial_id}.world",
        samples_csv=raw_dir / "samples.csv",
        summary_json=summary_dir / "summary.json",
    )


def build_context(
    config: dict[str, Any],
    scenario_cfg: dict[str, Any],
    params: dict[str, float],
    repo_root: Path,
    workspace_root: Path,
    results_dir: Path,
    repeat_idx: int,
) -> TrialContext:
    study_cfg = config["study"]
    trial_id = build_trial_id(scenario_cfg["id"], repeat_idx)
    world_template_dir = resolve_repo_path(
        repo_root,
        study_cfg.get("world_template_dir", study_cfg["world_output_dir"]),
    )
    world_output_dir = resolve_repo_path(repo_root, study_cfg["world_output_dir"])
    files = make_trial_files(results_dir, world_output_dir, trial_id)
    param_names = [item["name"] for item in config.get("search_space", [])]
    termination_cfg = merge_mapping(
        dict(config.get("termination", {})),
        scenario_cfg.get("termination"),
    )
    logging_cfg = merge_mapping(
        dict(config.get("logging", {})),
        scenario_cfg.get("logging"),
    )
    return TrialContext(
        study_name=study_cfg["name"],
        scenario_id=scenario_cfg["id"],
        scenario_split=scenario_cfg["split"],
        trial_id=trial_id,
        repeat_idx=repeat_idx,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        repo_root=repo_root,
        workspace_root=workspace_root,
        world_template_dir=world_template_dir,
        launch_file=resolve_repo_path(repo_root, study_cfg["launch_file"]),
        results_dir=results_dir,
        files=files,
        params=params,
        param_names=param_names,
        study_cfg=study_cfg,
        scenario_cfg=scenario_cfg,
        termination_cfg=termination_cfg,
        logging_cfg=logging_cfg,
    )


def write_trial_manifest(context: TrialContext) -> None:
    manifest = {
        "trial_id": context.trial_id,
        "study_name": context.study_name,
        "scenario_id": context.scenario_id,
        "scenario_split": context.scenario_split,
        "repeat_idx": context.repeat_idx,
        "params": context.params,
        "files": {
            "param_file": str(context.files.param_file),
            "world_file": str(context.files.world_file),
            "samples_csv": str(context.files.samples_csv),
            "summary_json": str(context.files.summary_json),
        },
        "study": context.study_cfg,
        "scenario": context.scenario_cfg,
        "termination": context.termination_cfg,
        "logging": context.logging_cfg,
    }
    context.files.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_trial_param_file(base_param_file: Path, overrides: dict[str, float], output_path: Path) -> None:
    lines = base_param_file.read_text(encoding="utf-8").splitlines()
    remaining = dict(overrides)
    rendered: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rendered.append(line)
            continue

        parts = stripped.split()
        key = parts[0]
        if key in remaining:
            rendered.append(f"{key:<16} {remaining.pop(key)}")
        else:
            rendered.append(line)

    if remaining:
        rendered.append("")
        rendered.append("# Added by run_trial.py")
        for key, value in sorted(remaining.items()):
            rendered.append(f"{key:<16} {value}")

    output_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def installed_world_file(workspace_root: Path, world_name: str) -> Path:
    return (
        workspace_root
        / "install"
        / "dave_worlds"
        / "share"
        / "dave_worlds"
        / "worlds"
        / f"{world_name}.world"
    )


def mirror_world_file(source_world_file: Path, destination_world_file: Path) -> None:
    destination_world_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_world_file, destination_world_file)


def _decode_xml_bytes(data: bytes) -> str:
    data = data.replace(codecs.BOM_UTF8, b"", 1)
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        text = data.decode("utf-16")
    elif b"\x00" in data:
        text = None
        for encoding in ("utf-16", "utf-16-le", "utf-16-be", "utf-8"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = data.replace(b"\x00", b"").decode("utf-8", errors="ignore")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="ignore")

    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    first_tag = text.find("<")
    if first_tag > 0:
        text = text[first_tag:]
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)
    return text


def load_xml_tree(path: Path) -> ET.ElementTree:
    data = path.read_bytes()
    text = _decode_xml_bytes(data)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        preview_hex = data[:32].hex()
        preview_text = text[:120].replace("\n", "\\n")
        raise RuntimeError(
            f"Failed to parse XML file: {path} ({exc}). "
            f"first_bytes_hex={preview_hex} preview={preview_text!r}"
        ) from exc
    return ET.ElementTree(root)


def _set_element_text(root: ET.Element, path: str, value: str) -> None:
    node = root.find(path)
    if node is None:
        raise KeyError(f"Could not find XML path: {path}")
    node.text = value


def render_trial_world(context: TrialContext) -> str:
    world_cfg = context.scenario_cfg["world"]
    template_name = world_cfg["template"]
    template_path = context.world_template_dir / f"{template_name}.world"
    if not template_path.exists():
        raise FileNotFoundError(f"World template was not found: {template_path}")
    tree = load_xml_tree(template_path)
    root = tree.getroot()

    wind_xyz = world_cfg["wind_world_xyz_mps"]
    _set_element_text(root, ".//wind/linear_velocity", f"{wind_xyz[0]} {wind_xyz[1]} {wind_xyz[2]}")
    _set_element_text(
        root,
        ".//plugin[@filename='gz-sim-wind-effects-system']/horizontal/magnitude/noise/stddev",
        str(world_cfg["wind_mag_noise_stddev"]),
    )
    _set_element_text(
        root,
        ".//plugin[@filename='gz-sim-wind-effects-system']/horizontal/direction/noise/stddev",
        str(world_cfg["wind_dir_noise_stddev"]),
    )
    _set_element_text(
        root,
        ".//plugin[@filename='gz-sim-wind-effects-system']/horizontal/magnitude/sin/amplitude_percent",
        str(world_cfg["wind_mag_sin_amp_percent"]),
    )
    _set_element_text(
        root,
        ".//plugin[@filename='gz-sim-wind-effects-system']/horizontal/direction/sin/amplitude",
        str(world_cfg["wind_dir_sin_amp_deg"]),
    )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    context.files.world_file.parent.mkdir(parents=True, exist_ok=True)
    tree.write(context.files.world_file, encoding="utf-8", xml_declaration=True)
    install_world_path = installed_world_file(context.workspace_root, context.files.world_file.stem)
    mirror_world_file(context.files.world_file, install_world_path)
    return context.files.world_file.stem


def build_launch_command(
    context: TrialContext,
    trial_world_name: str,
    *,
    launch_file: str | Path | None = None,
    param_file: str | Path | None = None,
) -> list[str]:
    spawn = context.scenario_cfg["spawn"]
    study_cfg = context.study_cfg
    home_lat, home_lon, home_alt = study_cfg["home_llh"]
    gazebo_spawn_yaw_deg = selected_gazebo_spawn_yaw_deg(spawn)
    ardupilot_heading_deg = selected_ardupilot_home_heading_deg(spawn)
    ardupilot_home = f"{home_lat},{home_lon},{home_alt},{ardupilot_heading_deg}"
    launch_cmd = [
        "ros2",
        "launch",
        str(launch_file if launch_file is not None else context.launch_file),
        f"world_name:={trial_world_name}",
        f"namespace:={study_cfg['namespace']}",
        f"x:={spawn['x_m']}",
        f"y:={spawn['y_m']}",
        f"z:={spawn['z_m']}",
        f"yaw:={math.radians(gazebo_spawn_yaw_deg)}",
        f"ardupilot_params:={param_file if param_file is not None else context.files.param_file}",
        f"ardupilot_home:={ardupilot_home}",
        f"start_mavproxy:={str(study_cfg.get('use_mavproxy', False)).lower()}",
        f"start_mavros:={str(study_cfg.get('use_mavros', False)).lower()}",
    ]
    if "sitl_start_delay_s" in study_cfg:
        launch_cmd.append(f"sitl_start_delay:={float(study_cfg['sitl_start_delay_s'])}")
    if "mavproxy_start_delay_s" in study_cfg:
        launch_cmd.append(f"mavproxy_start_delay:={float(study_cfg['mavproxy_start_delay_s'])}")
    if "mavros_start_delay_s" in study_cfg:
        launch_cmd.append(f"mavros_start_delay:={float(study_cfg['mavros_start_delay_s'])}")
    if "gui" in study_cfg:
        launch_cmd.append(f"gui:={str(bool(study_cfg['gui'])).lower()}")
    if "headless" in study_cfg:
        launch_cmd.append(f"headless:={str(bool(study_cfg['headless'])).lower()}")
    if "paused" in study_cfg:
        launch_cmd.append(f"paused:={str(bool(study_cfg['paused'])).lower()}")
    if "verbose" in study_cfg:
        launch_cmd.append(f"verbose:={study_cfg['verbose']}")
    return launch_cmd


def build_launch_shell_command(
    launch_cmd: list[str],
    *,
    workspace_root: str | Path,
    extra_setup_scripts: Iterable[str | Path] = (),
) -> str:
    workspace_setup = Path(workspace_root) / "install" / "setup.bash"
    wave_plugin_dir = Path(workspace_root) / "src" / "dave" / "gazebo" / "dave_gz_world_plugins" / "ocean-waves" / "install" / "wave" / "lib"
    source_steps = [
        "export FASTDDS_BUILTIN_TRANSPORTS=\"${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}\"",
        "export XDG_RUNTIME_DIR=\"${XDG_RUNTIME_DIR:-/tmp/runtime-${USER:-docker}}\"",
        "unset SESSION_MANAGER || true",
        "if [ -d \"$HOME/ardupilot_ws/ardupilot/Tools/autotest\" ]; then "
        "export PATH=\"$HOME/ardupilot_ws/ardupilot/Tools/autotest:$PATH\"; fi",
        "if [ -d \"$HOME/ardupilot_ws/ardupilot/build/sitl/bin\" ]; then "
        "export PATH=\"$HOME/ardupilot_ws/ardupilot/build/sitl/bin:$PATH\"; fi",
        "if [ -d \"/opt/asv_sim_ws/install/lib\" ]; then "
        "export GZ_SIM_SYSTEM_PLUGIN_PATH=\"/opt/asv_sim_ws/install/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}\"; fi",
        "if [ -d \"$HOME/ardupilot_ws/ardupilot_gazebo/build\" ]; then "
        "export GZ_SIM_SYSTEM_PLUGIN_PATH=\"$HOME/ardupilot_ws/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}\"; fi",
        f"if [ -d {shlex.quote(str(wave_plugin_dir))} ]; then "
        f"export GZ_SIM_SYSTEM_PLUGIN_PATH={shlex.quote(str(wave_plugin_dir))}:${{GZ_SIM_SYSTEM_PLUGIN_PATH:-}}; "
        f"export LD_LIBRARY_PATH={shlex.quote(str(wave_plugin_dir))}:${{LD_LIBRARY_PATH:-}}; fi",
        "export GZ_SIM_RESOURCE_PATH=\"/opt/asv_sim_ws/src/asv_sim/asv_sim_gazebo/models:"
        "/opt/asv_sim_ws/src/asv_sim/asv_sim_gazebo/worlds:"
        "$HOME/ardupilot_ws/ardupilot_gazebo/models:"
        "$HOME/ardupilot_ws/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}\"",
        "if [ -n \"${ROS_DISTRO:-}\" ] && [ -f \"/opt/ros/${ROS_DISTRO}/setup.bash\" ]; then "
        "source \"/opt/ros/${ROS_DISTRO}/setup.bash\"; "
        "else "
        "for candidate in /opt/ros/*/setup.bash; do "
        "if [ -f \"$candidate\" ]; then source \"$candidate\"; break; fi; "
        "done; "
        "fi",
        "if ! compgen -G '/opt/ros/*/setup.bash' > /dev/null && "
        "[ ! -f \"/opt/ros/${ROS_DISTRO:-missing}/setup.bash\" ]; then "
        "echo 'ROS 2 setup script was not found under /opt/ros' >&2; exit 1; "
        "fi",
        f"if [ ! -f {shlex.quote(str(workspace_setup))} ]; then "
        f"echo 'Workspace overlay is missing: {workspace_setup}' >&2; exit 1; "
        "fi",
        f"source {shlex.quote(str(workspace_setup))}",
    ]
    for script in extra_setup_scripts:
        source_steps.append(f"if [ -f {shlex.quote(str(script))} ]; then source {shlex.quote(str(script))}; fi")
    source_steps.extend(
        [
            "if ! command -v ros2 >/dev/null 2>&1; then echo 'ros2 command was not found in PATH' >&2; exit 1; fi",
            "if ! command -v ardurover >/dev/null 2>&1; then echo 'ardurover command was not found in PATH' >&2; exit 1; fi",
        ]
    )
    ros_cmd = " ".join(shlex.quote(part) for part in launch_cmd)
    return " && ".join(source_steps + [ros_cmd])


def sync_file_to_container(host_path: Path, docker_container: str, container_path: str) -> None:
    container_parent = str(PurePosixPath(container_path).parent)
    mkdir_cmd = [
        "docker",
        "exec",
        "-i",
        docker_container,
        "bash",
        "-lc",
        f"mkdir -p {shlex.quote(container_parent)}",
    ]
    subprocess.run(mkdir_cmd, check=True)
    subprocess.run(
        ["docker", "cp", str(host_path), f"{docker_container}:{container_path}"],
        check=True,
    )


def build_launch_plan(
    context: TrialContext,
    execution: ExecutionConfig,
    trial_world_name: str,
    *,
    prepare_remote_inputs: bool,
) -> LaunchPlan:
    if execution.backend == "local":
        launch_cmd = build_launch_command(context, trial_world_name)
        shell_cmd = build_launch_shell_command(
            launch_cmd,
            workspace_root=context.workspace_root,
            extra_setup_scripts=(
                "/opt/dave_ws/install/setup.bash",
                "/opt/asv_sim_ws/install/setup.bash",
            ),
        )
        return LaunchPlan(
            backend="local",
            launch_cmd=launch_cmd,
            process_cmd=["bash", "-lc", shell_cmd],
            display_cmd=" ".join(launch_cmd),
            cleanup_token=context.trial_id,
        )

    if execution.docker_container is None or execution.container_repo_root is None:
        raise ValueError("docker-exec backend requires docker_container and container_repo_root")

    container_workspace_root = get_workspace_root(Path(execution.container_repo_root))
    container_launch_file = repo_relative_to_container_path(
        execution.container_repo_root,
        host_path_to_repo_relative(context.repo_root, context.launch_file),
    )
    container_world_file = repo_relative_to_container_path(
        execution.container_repo_root,
        host_path_to_repo_relative(context.repo_root, context.files.world_file),
    )
    container_install_world_file = str(
        PurePosixPath(container_workspace_root)
        / "install"
        / "dave_worlds"
        / "share"
        / "dave_worlds"
        / "worlds"
        / f"{trial_world_name}.world"
    )
    container_param_file = str(PurePosixPath("/tmp/sailboat_bo") / context.trial_id / "trial.parm")

    if prepare_remote_inputs:
        sync_file_to_container(context.files.world_file, execution.docker_container, container_world_file)
        sync_file_to_container(context.files.world_file, execution.docker_container, container_install_world_file)
        sync_file_to_container(context.files.param_file, execution.docker_container, container_param_file)

    launch_cmd = build_launch_command(
        context,
        trial_world_name,
        launch_file=container_launch_file,
        param_file=container_param_file,
    )
    shell_cmd = build_launch_shell_command(
        launch_cmd,
        workspace_root=container_workspace_root,
        extra_setup_scripts=(
            "/opt/dave_ws/install/setup.bash",
            "/opt/asv_sim_ws/install/setup.bash",
        ),
    )
    process_cmd = [
        "docker",
        "exec",
        "-i",
        execution.docker_container,
        "bash",
        "-lc",
        shell_cmd,
    ]
    return LaunchPlan(
        backend="docker-exec",
        launch_cmd=launch_cmd,
        process_cmd=process_cmd,
        display_cmd=" ".join(process_cmd),
        cleanup_token=context.trial_id,
    )


def start_launch_process(context: TrialContext, launch_plan: LaunchPlan) -> subprocess.Popen[str]:
    stdout_path = context.files.logs_dir / "launch.stdout.log"
    stderr_path = context.files.logs_dir / "launch.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        launch_plan.process_cmd,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    stdout_handle.close()
    stderr_handle.close()
    return process


def run_cleanup_token(
    execution: ExecutionConfig,
    cleanup_token: str | None,
) -> None:
    if not cleanup_token:
        return
    quoted_token = shlex.quote(cleanup_token)
    if execution.backend == "docker-exec" and execution.docker_container:
        cleanup_cmd = [
            "docker",
            "exec",
            "-i",
            execution.docker_container,
            "bash",
            "-lc",
            f"pkill -f {quoted_token} >/dev/null 2>&1 || true",
        ]
    else:
        cleanup_cmd = [
            "bash",
            "-lc",
            f"pkill -f {quoted_token} >/dev/null 2>&1 || true",
        ]
    try:
        subprocess.run(cleanup_cmd, check=False)
    except Exception:
        pass


def stop_launch_process(
    process: subprocess.Popen[str] | None,
    execution: ExecutionConfig,
    launch_plan: LaunchPlan | None = None,
) -> None:
    if process is not None and process.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except Exception:
                pass
    if launch_plan is not None:
        run_cleanup_token(execution, launch_plan.cleanup_token)


def sail_force_gate_enabled(context: TrialContext) -> bool:
    return bool(context.study_cfg.get("sail_force_gate_during_startup", True))


def sail_force_enable_topic(context: TrialContext) -> str:
    configured = str(context.study_cfg.get("sail_force_enable_topic", "")).strip()
    if configured:
        return configured
    namespace = str(context.study_cfg.get("namespace", "sailboat")).strip("/") or "sailboat"
    return f"/model/{namespace}/sail_lift_drag/enable"


def publish_sail_force_enabled(
    context: TrialContext,
    execution: ExecutionConfig,
    enabled: bool,
    *,
    reason: str,
) -> None:
    if not sail_force_gate_enabled(context):
        return

    topic = sail_force_enable_topic(context)
    if enabled:
        repeats = int(context.study_cfg.get("sail_force_enable_repeats", 3))
        interval_s = float(context.study_cfg.get("sail_force_enable_repeat_interval_s", 0.15))
    else:
        repeats = int(context.study_cfg.get("sail_force_disable_repeats", 8))
        interval_s = float(context.study_cfg.get("sail_force_disable_repeat_interval_s", 0.25))
    repeats = max(1, repeats)
    interval_s = max(0.0, interval_s)

    gz_cmd = [
        "gz",
        "topic",
        "-t",
        topic,
        "-m",
        "gz.msgs.Boolean",
        "-p",
        f"data: {'true' if enabled else 'false'}",
    ]
    shell_cmd = " ".join(shlex.quote(part) for part in gz_cmd)
    if execution.backend == "docker-exec":
        if not execution.docker_container:
            print("[run_trial] warning: cannot publish sail force gate without docker_container")
            return
        cmd = ["docker", "exec", "-i", execution.docker_container, "bash", "-lc", shell_cmd]
    else:
        cmd = ["bash", "-lc", shell_cmd]

    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt_idx in range(repeats):
        try:
            last_result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except Exception as exc:
            print(
                "[run_trial] warning: failed to publish sail force gate: "
                f"enabled={enabled}, reason={reason}, topic={topic}, error={exc}"
            )
            return
        if attempt_idx < repeats - 1 and interval_s > 0.0:
            time.sleep(interval_s)

    if last_result is not None and last_result.returncode != 0:
        stderr = (last_result.stderr or "").strip()
        print(
            "[run_trial] warning: sail force gate publish returned non-zero: "
            f"enabled={enabled}, reason={reason}, topic={topic}, rc={last_result.returncode}, stderr={stderr}"
        )
        return

    print(
        "[run_trial] sail force gate: "
        f"enabled={str(enabled).lower()}, reason={reason}, topic={topic}, repeats={repeats}"
    )


def require_pymavlink() -> Any:
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pymavlink is required for live runs. Install it with `pip install pymavlink`, "
            "or rerun with --dry-run while you finish the scaffolding."
        ) from exc
    return mavutil

def wait_for_message(master: Any, types: list[str], timeout_s: float) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = master.recv_match(type=types, blocking=True, timeout=0.5)
        if msg is not None:
            return msg
    return None


def mission_result_name(mavlink: Any, result_code: int) -> str:
    try:
        entry = mavlink.enums["MAV_MISSION_RESULT"][int(result_code)]
        return str(entry.name)
    except Exception:
        return f"UNKNOWN_{int(result_code)}"


def mav_result_name(mavlink: Any, result_code: int) -> str:
    try:
        entry = mavlink.enums["MAV_RESULT"][int(result_code)]
        return str(entry.name)
    except Exception:
        return f"UNKNOWN_{int(result_code)}"


def mav_state_name(mavlink: Any, state_code: int) -> str:
    try:
        entry = mavlink.enums["MAV_STATE"][int(state_code)]
        return str(entry.name)
    except Exception:
        return f"UNKNOWN_{int(state_code)}"


def read_param_map(param_file: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in param_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return values


def selected_mission_frame(context: TrialContext) -> str:
    raw_value = str(context.scenario_cfg.get("mission", {}).get("frame", "local_enu_m")).strip().lower()
    aliases = {
        "local_enu": "local_enu_m",
        "local_enu_m": "local_enu_m",
        "gazebo_xy": "gazebo_xy_m",
        "gazebo_xy_m": "gazebo_xy_m",
    }
    frame = aliases.get(raw_value)
    if frame not in SUPPORTED_MISSION_FRAMES:
        raise ValueError(
            f"Unsupported mission frame '{raw_value}'. Expected one of: {sorted(SUPPORTED_MISSION_FRAMES)}"
        )
    return frame


def mission_xy_to_enu(frame: str, x_m: float, y_m: float) -> tuple[float, float]:
    if frame == "local_enu_m":
        return x_m, y_m
    if frame == "gazebo_xy_m":
        # ArduPilotPlugin maps Gazebo X to NED north and Gazebo -Y to NED east.
        return -y_m, x_m
    raise ValueError(f"Unsupported mission frame '{frame}'")


def enu_to_mission_xy(frame: str, east_m: float, north_m: float) -> tuple[float, float]:
    if frame == "local_enu_m":
        return east_m, north_m
    if frame == "gazebo_xy_m":
        return north_m, -east_m
    raise ValueError(f"Unsupported mission frame '{frame}'")


def enu_to_geodetic(home_lat_deg: float, home_lon_deg: float, east_m: float, north_m: float) -> tuple[float, float]:
    lat0_rad = math.radians(home_lat_deg)
    d_lat = north_m / EARTH_RADIUS_M
    d_lon = east_m / (EARTH_RADIUS_M * max(math.cos(lat0_rad), 1e-9))
    return home_lat_deg + math.degrees(d_lat), home_lon_deg + math.degrees(d_lon)


def geodetic_to_enu(home_lat_deg: float, home_lon_deg: float, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat0_rad = math.radians(home_lat_deg)
    d_lat = math.radians(lat_deg - home_lat_deg)
    d_lon = math.radians(lon_deg - home_lon_deg)
    north_m = d_lat * EARTH_RADIUS_M
    east_m = d_lon * EARTH_RADIUS_M * math.cos(lat0_rad)
    return east_m, north_m


def mission_xy_to_geodetic(
    frame: str,
    home_lat_deg: float,
    home_lon_deg: float,
    x_m: float,
    y_m: float,
) -> tuple[float, float]:
    east_m, north_m = mission_xy_to_enu(frame, x_m, y_m)
    return enu_to_geodetic(home_lat_deg, home_lon_deg, east_m, north_m)


def pwm_to_surface_angle_rad(pwm: float | None, pwm_min: float, pwm_max: float, multiplier: float = 1.5708) -> float:
    if pwm is None or pwm_max <= pwm_min:
        return 0.0
    normalized = clip((float(pwm) - pwm_min) / (pwm_max - pwm_min), 0.0, 1.0)
    return (normalized - 0.5) * multiplier


def request_message_intervals(master: Any) -> None:
    mavlink = get_mavlink_module(master)
    for message_name, hz in MESSAGE_INTERVAL_HZ.items():
        try:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(MESSAGE_NAME_TO_ID[message_name]),
                float(int(1_000_000 / hz)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        except Exception:
            continue


def get_mavlink_module(master: Any) -> Any:
    if hasattr(master, "mavlink"):
        return master.mavlink

    mav = getattr(master, "mav", None)
    if mav is not None and hasattr(mav, "mavlink"):
        return mav.mavlink

    mavutil = require_pymavlink()
    return mavutil.mavlink


def tail_text_file(path: Path, line_count: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int] | None:
    if not endpoint.startswith("tcp:"):
        return None
    _, _, remainder = endpoint.partition("tcp:")
    host, sep, port_text = remainder.rpartition(":")
    if not sep or not host or not port_text:
        return None
    try:
        return host, int(port_text)
    except ValueError:
        return None


def is_tcp_endpoint_open(endpoint: str, *, timeout_s: float = 0.25) -> bool | None:
    parsed = parse_tcp_endpoint(endpoint)
    if parsed is None:
        return None
    host, port = parsed
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def wait_for_endpoint_closed(endpoint: str, timeout_s: float) -> bool:
    if timeout_s <= 0.0:
        return True
    state = is_tcp_endpoint_open(endpoint)
    if state is None:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_tcp_endpoint_open(endpoint):
            return True
        time.sleep(0.25)
    return not bool(is_tcp_endpoint_open(endpoint))


def connect_mavlink(
    endpoint: str,
    timeout_s: float,
    *,
    launch_process: subprocess.Popen[str] | None = None,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
) -> Any:
    mavutil = require_pymavlink()
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    attempt_idx = 0

    while time.monotonic() < deadline:
        attempt_idx += 1
        if launch_process is not None and launch_process.poll() is not None:
            log_sections: list[str] = []
            if stdout_log:
                stdout_tail = tail_text_file(stdout_log)
                if stdout_tail:
                    log_sections.append(f"Last launch stdout lines:\n{stdout_tail}")
            if stderr_log:
                stderr_tail = tail_text_file(stderr_log)
                if stderr_tail:
                    log_sections.append(f"Last launch stderr lines:\n{stderr_tail}")
            log_hint = ("\n" + "\n\n".join(log_sections)) if log_sections else ""
            raise RuntimeError(
                "Launch process exited before MAVLink became ready. "
                f"See {stdout_log} and {stderr_log} for details.{log_hint}"
            )
        master = None
        last_error = None
        try:
            master = mavutil.mavlink_connection(endpoint)
            heartbeat_timeout = max(1.0, min(5.0, deadline - time.monotonic()))
            heartbeat = master.wait_heartbeat(timeout=heartbeat_timeout)
            if heartbeat is not None:
                request_message_intervals(master)
                return master
            last_error = TimeoutError(f"No MAVLink heartbeat received yet from {endpoint}")
        except OSError as exc:
            last_error = exc
        finally:
            if master is not None and last_error is not None:
                try:
                    master.close()
                except Exception:
                    pass
        if attempt_idx % 3 == 0:
            remaining_s = max(0.0, deadline - time.monotonic())
            print(
                f"[run_trial] still waiting for MAVLink at {endpoint} "
                f"({remaining_s:.0f}s left); last_error={last_error}"
            )
            if stdout_log:
                print(f"[run_trial] inspect launch stdout: {stdout_log}")
            if stderr_log:
                print(f"[run_trial] inspect launch stderr: {stderr_log}")
        time.sleep(2.0)

    if last_error is not None:
        raise TimeoutError(f"Timed out waiting for MAVLink endpoint {endpoint}: {last_error}") from last_error
    raise TimeoutError(f"Timed out waiting for MAVLink endpoint {endpoint}")


def build_mission_waypoints(context: TrialContext) -> list[MissionWaypoint]:
    return [
        MissionWaypoint(seq=index, x_m=float(waypoint[0]), y_m=float(waypoint[1]))
        for index, waypoint in enumerate(context.scenario_cfg["mission"]["waypoints"])
    ]


def build_mission_upload_items(
    mission_waypoints: list[MissionWaypoint],
) -> list[MissionUploadItem]:
    upload_items = [
        MissionUploadItem(
            raw_seq=0,
            x_m=0.0,
            y_m=0.0,
            is_home=True,
            user_waypoint_index=None,
        )
    ]
    upload_items.extend(
        MissionUploadItem(
            raw_seq=index + 1,
            x_m=waypoint.x_m,
            y_m=waypoint.y_m,
            is_home=False,
            user_waypoint_index=waypoint.seq,
        )
        for index, waypoint in enumerate(mission_waypoints)
    )
    return upload_items


def advance_mission_current_to_waypoint(
    master: Any,
    context: TrialContext,
    state: TelemetryState,
    *,
    next_waypoint_index: int,
    waypoint_count: int,
    home_lat_deg: float,
    home_lon_deg: float,
    servo_params: dict[str, float],
    timeout_s: float = 8.0,
) -> bool:
    if next_waypoint_index < 0 or next_waypoint_index >= waypoint_count:
        return False

    mission_frame = selected_mission_frame(context)
    current_waypoint_index = mission_seq_to_waypoint_index(
        context,
        state.mission_seq,
        waypoint_count,
    )
    if current_waypoint_index >= next_waypoint_index:
        return True

    mavlink = get_mavlink_module(master)
    target_raw_seq = waypoint_index_to_mission_raw_seq(
        context,
        next_waypoint_index,
        waypoint_count,
    )
    command_id = getattr(mavlink, "MAV_CMD_DO_SET_MISSION_CURRENT", None)
    master.mav.mission_set_current_send(
        master.target_system,
        master.target_component,
        target_raw_seq,
    )
    if command_id is not None:
        try:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                command_id,
                0,
                float(target_raw_seq),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        except Exception:
            pass

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = master.recv_match(
            type=[
                "MISSION_CURRENT",
                "MISSION_ITEM_REACHED",
                "GLOBAL_POSITION_INT",
                "ATTITUDE",
                "VFR_HUD",
                "SERVO_OUTPUT_RAW",
                "NAV_CONTROLLER_OUTPUT",
            ],
            blocking=True,
            timeout=0.5,
        )
        if msg is None:
            continue
        update_state_from_message(state, msg, home_lat_deg, home_lon_deg, servo_params, mission_frame)
        current_waypoint_index = mission_seq_to_waypoint_index(
            context,
            state.mission_seq,
            waypoint_count,
        )
        if current_waypoint_index >= next_waypoint_index:
            return True
    return False


def set_vehicle_mode(master: Any, mode_name: str, timeout_s: float = 10.0) -> None:
    mavlink = get_mavlink_module(master)
    mode_map = master.mode_mapping()
    if mode_map is None or mode_name not in mode_map:
        raise RuntimeError(f"Vehicle mode mapping does not contain {mode_name}")

    target_mode = int(mode_map[mode_name])
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavlink.MAV_CMD_DO_SET_MODE,
        0,
        float(mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
        float(target_mode),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        heartbeat = master.recv_match(type=["HEARTBEAT"], blocking=True, timeout=0.5)
        if heartbeat is None:
            continue
        if int(getattr(heartbeat, "custom_mode", -1)) == target_mode:
            return
    raise TimeoutError(f"Timed out waiting for {mode_name} mode")


def retarget_mission_after_local_capture(
    master: Any,
    context: TrialContext,
    mission_waypoints: list[MissionWaypoint],
    *,
    next_waypoint_index: int,
) -> bool:
    if next_waypoint_index < 0 or next_waypoint_index >= len(mission_waypoints):
        return False

    print(
        "[run_trial] re-uploading full mission after local capture: "
        f"next_waypoint_index={next_waypoint_index}, "
        f"waypoint_count={len(mission_waypoints)}"
    )
    upload_mission(
        master,
        context,
        mission_waypoints,
        start_waypoint_index=next_waypoint_index,
    )
    set_vehicle_mode(master, "AUTO", timeout_s=10.0)
    return True


def wait_for_mission_protocol_ready(master: Any, timeout_s: float = 30.0) -> int:
    mavlink = get_mavlink_module(master)
    deadline = time.monotonic() + timeout_s
    last_observation = "no_mission_response"

    while time.monotonic() < deadline:
        master.mav.mission_request_list_send(master.target_system, master.target_component)
        msg = wait_for_message(master, ["MISSION_COUNT", "MISSION_ACK"], timeout_s=2.0)
        if msg is None:
            time.sleep(1.0)
            continue

        msg_type = msg.get_type()
        if msg_type == "MISSION_COUNT":
            return int(msg.count)

        ack_code = int(getattr(msg, "type", -1))
        last_observation = f"{ack_code} ({mission_result_name(mavlink, ack_code)})"
        time.sleep(1.0)

    raise TimeoutError(
        "Mission protocol did not become ready before timeout. "
        f"Last observation: {last_observation}"
    )


def mission_item_lat_lon_deg(msg: Any) -> tuple[float, float]:
    if msg.get_type() == "MISSION_ITEM_INT":
        return float(msg.x) * 1e-7, float(msg.y) * 1e-7
    return float(msg.x), float(msg.y)


def acknowledge_mission_download(master: Any, mavlink: Any) -> None:
    mission_ack_send = getattr(master.mav, "mission_ack_send", None)
    if mission_ack_send is None:
        return
    try:
        mission_ack_send(
            master.target_system,
            master.target_component,
            mavlink.MAV_MISSION_ACCEPTED,
            getattr(mavlink, "MAV_MISSION_TYPE_MISSION", 0),
        )
    except TypeError:
        mission_ack_send(
            master.target_system,
            master.target_component,
            mavlink.MAV_MISSION_ACCEPTED,
        )


def download_mission_items(
    master: Any,
    expected_count: int,
    *,
    timeout_s: float = 15.0,
) -> list[Any]:
    mavlink = get_mavlink_module(master)
    master.mav.mission_request_list_send(master.target_system, master.target_component)
    count_msg = wait_for_message(master, ["MISSION_COUNT"], timeout_s=min(timeout_s, 5.0))
    if count_msg is None:
        raise TimeoutError("Timed out waiting for MISSION_COUNT during mission verification")

    actual_count = int(count_msg.count)
    if actual_count != expected_count:
        raise RuntimeError(
            "Mission verification count mismatch: "
            f"expected={expected_count}, actual={actual_count}"
        )

    downloaded: list[Any] = []
    deadline = time.monotonic() + timeout_s
    for expected_seq in range(expected_count):
        item = None
        while time.monotonic() < deadline:
            master.mav.mission_request_int_send(
                master.target_system,
                master.target_component,
                expected_seq,
            )
            candidate = wait_for_message(
                master,
                ["MISSION_ITEM_INT", "MISSION_ITEM"],
                timeout_s=min(2.0, max(0.0, deadline - time.monotonic())),
            )
            if candidate is None:
                continue
            if int(candidate.seq) == expected_seq:
                item = candidate
                break
        if item is None:
            raise TimeoutError(
                f"Timed out reading mission item seq={expected_seq} during verification"
            )
        downloaded.append(item)

    acknowledge_mission_download(master, mavlink)
    return downloaded


def verify_uploaded_mission(
    master: Any,
    context: TrialContext,
    upload_items: list[MissionUploadItem],
    *,
    coordinate_tolerance_m: float = 1.0,
    home_coordinate_tolerance_m: float = 10.0,
) -> None:
    mavlink = get_mavlink_module(master)
    mission_frame = selected_mission_frame(context)
    home_lat_deg = float(context.study_cfg["home_llh"][0])
    home_lon_deg = float(context.study_cfg["home_llh"][1])
    downloaded = download_mission_items(master, len(upload_items))
    home_error_m = 0.0
    max_waypoint_error_m = 0.0

    for expected, actual in zip(upload_items, downloaded):
        actual_command = int(getattr(actual, "command", -1))
        if actual_command != int(mavlink.MAV_CMD_NAV_WAYPOINT):
            raise RuntimeError(
                "Mission verification command mismatch: "
                f"seq={expected.raw_seq}, expected={mavlink.MAV_CMD_NAV_WAYPOINT}, "
                f"actual={actual_command}"
            )

        if expected.is_home:
            expected_lat_deg = home_lat_deg
            expected_lon_deg = home_lon_deg
        else:
            expected_lat_deg, expected_lon_deg = mission_xy_to_geodetic(
                mission_frame,
                home_lat_deg,
                home_lon_deg,
                expected.x_m,
                expected.y_m,
            )
        actual_lat_deg, actual_lon_deg = mission_item_lat_lon_deg(actual)
        east_error_m, north_error_m = geodetic_to_enu(
            expected_lat_deg,
            expected_lon_deg,
            actual_lat_deg,
            actual_lon_deg,
        )
        coordinate_error_m = math.hypot(east_error_m, north_error_m)
        allowed_error_m = (
            home_coordinate_tolerance_m if expected.is_home else coordinate_tolerance_m
        )
        if expected.is_home:
            home_error_m = coordinate_error_m
        else:
            max_waypoint_error_m = max(max_waypoint_error_m, coordinate_error_m)
        if coordinate_error_m > allowed_error_m:
            raise RuntimeError(
                "Mission verification coordinate mismatch: "
                f"seq={expected.raw_seq}, error_m={coordinate_error_m:.3f}, "
                f"tolerance_m={allowed_error_m:.3f}"
            )

    print(
        "[run_trial] mission verified: "
        f"item_count={len(upload_items)}, home_seq=0, "
        f"navigation_seq=1..{len(upload_items) - 1}, "
        f"home_error_m={home_error_m:.3f}, "
        f"max_waypoint_error_m={max_waypoint_error_m:.3f}"
    )


def send_mission_item(
    master: Any,
    mavlink: Any,
    msg_type: str,
    target_system: int,
    target_component: int,
    item: MissionUploadItem,
    success_radius_m: float,
    mission_frame: str,
    home_lat_deg: float,
    home_lon_deg: float,
    home_alt_m: float,
) -> None:
    if item.is_home:
        lat_deg = home_lat_deg
        lon_deg = home_lon_deg
        frame = (
            getattr(mavlink, "MAV_FRAME_GLOBAL", mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT)
            if msg_type == "MISSION_REQUEST"
            else getattr(
                mavlink,
                "MAV_FRAME_GLOBAL_INT",
                mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            )
        )
        altitude_m = home_alt_m
        item_success_radius_m = 0.0
    else:
        lat_deg, lon_deg = mission_xy_to_geodetic(
            mission_frame,
            home_lat_deg,
            home_lon_deg,
            item.x_m,
            item.y_m,
        )
        frame = (
            mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            if msg_type == "MISSION_REQUEST"
            else mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
        )
        altitude_m = 0.0
        item_success_radius_m = success_radius_m

    current = 0
    autocontinue = 1
    hold_time_s = 0.0
    pass_radius_m = 0.0
    desired_yaw_deg = 0.0

    if msg_type == "MISSION_REQUEST":
        master.mav.mission_item_send(
            target_system,
            target_component,
            item.raw_seq,
            frame,
            mavlink.MAV_CMD_NAV_WAYPOINT,
            current,
            autocontinue,
            hold_time_s,
            item_success_radius_m,
            pass_radius_m,
            desired_yaw_deg,
            float(lat_deg),
            float(lon_deg),
            altitude_m,
        )
        return

    master.mav.mission_item_int_send(
        target_system,
        target_component,
        item.raw_seq,
        frame,
        mavlink.MAV_CMD_NAV_WAYPOINT,
        current,
        autocontinue,
        hold_time_s,
        item_success_radius_m,
        pass_radius_m,
        desired_yaw_deg,
        int(round(lat_deg * 1e7)),
        int(round(lon_deg * 1e7)),
        altitude_m,
    )


def upload_mission(
    master: Any,
    context: TrialContext,
    mission_waypoints: list[MissionWaypoint],
    *,
    start_waypoint_index: int = 0,
) -> None:
    if not mission_waypoints:
        raise ValueError("Mission is empty")
    if not mission_seq_home_offset_enabled(context):
        raise ValueError(
            "Explicit HOME mission upload requires study.mission_seq_home_offset=true"
        )
    if start_waypoint_index < 0 or start_waypoint_index >= len(mission_waypoints):
        raise ValueError(
            "start_waypoint_index is outside the mission: "
            f"index={start_waypoint_index}, waypoint_count={len(mission_waypoints)}"
        )

    home_lat_deg = float(context.study_cfg["home_llh"][0])
    home_lon_deg = float(context.study_cfg["home_llh"][1])
    home_alt_m = float(context.study_cfg["home_llh"][2])
    mission_frame = selected_mission_frame(context)
    success_radius_m = float(context.termination_cfg.get("success_radius_m", 5.0))
    mavlink = get_mavlink_module(master)
    target_system = master.target_system
    target_component = master.target_component
    upload_items = build_mission_upload_items(mission_waypoints)
    mission_item_count = len(upload_items)
    first_navigation_seq = waypoint_index_to_mission_raw_seq(
        context,
        start_waypoint_index,
        len(mission_waypoints),
    )
    current_mission_count = wait_for_mission_protocol_ready(master, timeout_s=30.0)
    upload_attempts = 3
    print(
        "[run_trial] uploading mission: "
        f"frame={mission_frame}, waypoint_count={len(mission_waypoints)}, "
        f"mission_item_count={mission_item_count}, start_seq={first_navigation_seq}"
    )

    for attempt_idx in range(upload_attempts):
        master.mav.mission_clear_all_send(target_system, target_component)
        clear_ack = wait_for_message(master, ["MISSION_ACK"], timeout_s=2.0)
        if clear_ack is not None and getattr(clear_ack, "type", mavlink.MAV_MISSION_ACCEPTED) not in (
            mavlink.MAV_MISSION_ACCEPTED,
            mavlink.MAV_MISSION_OPERATION_CANCELLED,
        ):
            clear_code = int(clear_ack.type)
            clear_name = mission_result_name(mavlink, clear_code)
            raise RuntimeError(f"MISSION_CLEAR_ALL failed with ACK type={clear_code} ({clear_name})")

        master.mav.mission_count_send(target_system, target_component, mission_item_count)
        retries_left = REQUEST_RETRIES
        retry_upload = False

        while retries_left > 0:
            msg = wait_for_message(master, ["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], timeout_s=3.0)
            if msg is None:
                retries_left -= 1
                master.mav.mission_count_send(target_system, target_component, mission_item_count)
                continue

            msg_type = msg.get_type()
            if msg_type == "MISSION_ACK":
                ack_code = int(msg.type)
                if ack_code != mavlink.MAV_MISSION_ACCEPTED:
                    ack_name = mission_result_name(mavlink, ack_code)
                    if ack_code in (mavlink.MAV_MISSION_NO_SPACE, mavlink.MAV_MISSION_ERROR) and attempt_idx < (upload_attempts - 1):
                        retry_upload = True
                        break
                    raise RuntimeError(
                        "Mission upload rejected with "
                        f"ACK type={ack_code} ({ack_name}), "
                        f"existing_mission_count={current_mission_count}, "
                        f"requested_mission_count={mission_item_count}"
                    )
                try:
                    verify_uploaded_mission(master, context, upload_items)
                except (RuntimeError, TimeoutError) as exc:
                    if attempt_idx < (upload_attempts - 1):
                        print(
                            "[run_trial] warning: mission verification failed; "
                            f"retrying upload: attempt={attempt_idx + 1}/{upload_attempts}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        retry_upload = True
                        break
                    raise
                master.mav.mission_set_current_send(
                    target_system,
                    target_component,
                    first_navigation_seq,
                )
                return

            seq = int(msg.seq)
            if seq < 0 or seq >= mission_item_count:
                raise RuntimeError(f"Vehicle requested invalid mission item seq={seq}")

            item = upload_items[seq]
            send_mission_item(
                master,
                mavlink,
                msg_type,
                target_system,
                target_component,
                item,
                success_radius_m,
                mission_frame,
                home_lat_deg,
                home_lon_deg,
                home_alt_m,
            )

        if retry_upload and attempt_idx < (upload_attempts - 1):
            current_mission_count = wait_for_mission_protocol_ready(master, timeout_s=10.0)
            time.sleep(2.0 * (attempt_idx + 1))
            continue

        break

    raise TimeoutError("Mission upload timed out before receiving final MISSION_ACK")


def send_arm_command(master: Any, force: bool = False) -> None:
    mavlink = get_mavlink_module(master)
    force_code = 21196.0 if force else 0.0
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0,
        force_code,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def attempt_arm_until_armed(
    master: Any,
    *,
    total_timeout_s: float,
    retry_interval_s: float,
    initial_delay_s: float,
    force: bool,
) -> None:
    mavlink = get_mavlink_module(master)
    start_time_s = time.monotonic()
    deadline = start_time_s + total_timeout_s
    next_arm_attempt_s = start_time_s + max(0.0, initial_delay_s)
    recent_status = deque(maxlen=12)
    arm_ack_result: tuple[int, str] | None = None
    last_system_state: tuple[int, str] | None = None
    prearm_ready_seen = False
    arm_attempt_count = 0
    last_progress_print_s = 0.0

    print(
        "[run_trial] arm stage started: "
        f"force={force}, initial_delay_s={initial_delay_s}, "
        f"retry_interval_s={retry_interval_s}, total_timeout_s={total_timeout_s}"
    )

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_arm_attempt_s:
            arm_attempt_count += 1
            print(
                "[run_trial] sending arm command: "
                f"force={force}, attempt={arm_attempt_count}, "
                f"elapsed_s={now - start_time_s:.1f}"
            )
            send_arm_command(master, force=force)
            next_arm_attempt_s = now + max(0.5, retry_interval_s)

        msg = master.recv_match(
            type=["HEARTBEAT", "COMMAND_ACK", "STATUSTEXT", "SYS_STATUS"],
            blocking=True,
            timeout=0.5,
        )
        if msg is None:
            continue
        msg_type = msg.get_type()
        if msg_type == "STATUSTEXT":
            text = str(getattr(msg, "text", "")).strip()
            if text:
                recent_status.append(text)
            continue
        if msg_type == "SYS_STATUS":
            sensors_health = int(getattr(msg, "onboard_control_sensors_health", 0))
            prearm_ready_now = bool(
                sensors_health & int(getattr(mavlink, "MAV_SYS_STATUS_PREARM_CHECK", 0))
            )
            if prearm_ready_now and not prearm_ready_seen:
                prearm_ready_seen = True
                print(
                    "[run_trial] pre-arm checks report ready; "
                    f"elapsed_s={time.monotonic() - start_time_s:.1f}"
                )
                next_arm_attempt_s = min(next_arm_attempt_s, time.monotonic())
            continue
        if msg_type == "COMMAND_ACK" and int(getattr(msg, "command", -1)) == int(mavlink.MAV_CMD_COMPONENT_ARM_DISARM):
            result_code = int(getattr(msg, "result", -1))
            arm_ack_result = (result_code, mav_result_name(mavlink, result_code))
            print(
                "[run_trial] arm ACK received: "
                f"result={arm_ack_result[0]} ({arm_ack_result[1]})"
            )
            continue
        last_system_state = (
            int(getattr(msg, "system_status", -1)),
            mav_state_name(mavlink, int(getattr(msg, "system_status", -1))),
        )
        if int(msg.base_mode) & int(mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print(
                "[run_trial] vehicle armed successfully: "
                f"elapsed_s={time.monotonic() - start_time_s:.1f}"
            )
            return
        if (now - last_progress_print_s) >= 5.0:
            remaining_s = max(0.0, deadline - now)
            next_attempt_in_s = max(0.0, next_arm_attempt_s - now)
            status_tail = recent_status[-1] if recent_status else ""
            print(
                "[run_trial] waiting for armed state: "
                f"elapsed_s={now - start_time_s:.1f}, "
                f"remaining_s={remaining_s:.1f}, "
                f"prearm_ready_seen={prearm_ready_seen}, "
                f"next_attempt_in_s={next_attempt_in_s:.1f}, "
                f"system_status={last_system_state[0]} ({last_system_state[1]})"
                + (f", last_statustext={status_tail}" if status_tail else "")
            )
            last_progress_print_s = now
    details: list[str] = []
    if arm_ack_result is not None:
        details.append(f"arm_ack={arm_ack_result[0]} ({arm_ack_result[1]})")
    if last_system_state is not None:
        details.append(f"system_status={last_system_state[0]} ({last_system_state[1]})")
    details.append(f"prearm_ready_seen={prearm_ready_seen}")
    if recent_status:
        details.append("statustext=" + " | ".join(recent_status))
    suffix = f": {'; '.join(details)}" if details else ""
    raise TimeoutError(f"Timed out waiting for armed state{suffix}")


def arm_and_set_auto(master: Any, context: TrialContext) -> None:
    arm_initial_delay_s = float(context.study_cfg.get("arm_initial_delay_s", 8.0))
    arm_total_timeout_s = float(context.study_cfg.get("arm_total_timeout_s", 60.0))
    arm_retry_interval_s = float(context.study_cfg.get("arm_retry_interval_s", 3.0))
    force_arm_timeout_s = float(context.study_cfg.get("force_arm_timeout_s", 20.0))
    force_arm_fallback = bool(context.study_cfg.get("force_arm_fallback", True))

    try:
        attempt_arm_until_armed(
            master,
            total_timeout_s=arm_total_timeout_s,
            retry_interval_s=arm_retry_interval_s,
            initial_delay_s=arm_initial_delay_s,
            force=False,
        )
    except TimeoutError as exc:
        if not force_arm_fallback:
            raise
        try:
            attempt_arm_until_armed(
                master,
                total_timeout_s=force_arm_timeout_s,
                retry_interval_s=max(1.0, arm_retry_interval_s),
                initial_delay_s=0.0,
                force=True,
            )
        except TimeoutError as force_exc:
            raise TimeoutError(
                f"{exc}. Force-arm fallback also failed: {force_exc}"
            ) from force_exc

    set_vehicle_mode(master, "AUTO", timeout_s=10.0)


def update_state_from_message(
    state: TelemetryState,
    msg: Any,
    home_lat_deg: float,
    home_lon_deg: float,
    servo_params: dict[str, float],
    mission_frame: str,
) -> None:
    msg_type = msg.get_type()
    if msg_type == "GLOBAL_POSITION_INT":
        state.sim_time_s = float(msg.time_boot_ms) / 1000.0
        state.lat_deg = float(msg.lat) / 1e7
        state.lon_deg = float(msg.lon) / 1e7
        east_m, north_m = geodetic_to_enu(home_lat_deg, home_lon_deg, state.lat_deg, state.lon_deg)
        state.x_m, state.y_m = enu_to_mission_xy(mission_frame, east_m, north_m)
        if getattr(msg, "hdg", 65535) != 65535:
            state.heading_deg = float(msg.hdg) / 100.0
            state.yaw_rad = math.radians(state.heading_deg)
        state.surge_speed_mps = math.hypot(float(msg.vx), float(msg.vy)) / 100.0
    elif msg_type == "ATTITUDE":
        state.sim_time_s = float(msg.time_boot_ms) / 1000.0
        state.yaw_rad = float(msg.yaw)
        state.roll_deg = math.degrees(float(msg.roll))
    elif msg_type == "VFR_HUD":
        state.surge_speed_mps = float(msg.groundspeed)
        state.heading_deg = float(msg.heading)
        if state.yaw_rad is None:
            state.yaw_rad = math.radians(state.heading_deg)
    elif msg_type == "SERVO_OUTPUT_RAW":
        state.rudder_cmd_rad = pwm_to_surface_angle_rad(
            float(msg.servo1_raw),
            servo_params.get("SERVO1_MIN", 1000.0),
            servo_params.get("SERVO1_MAX", 2000.0),
        )
        state.sail_cmd_rad = pwm_to_surface_angle_rad(
            float(msg.servo2_raw),
            servo_params.get("SERVO2_MIN", 1000.0),
            servo_params.get("SERVO2_MAX", 2000.0),
        )
    elif msg_type == "NAV_CONTROLLER_OUTPUT":
        state.nav_wp_dist_m = float(msg.wp_dist)
        state.nav_xtrack_error_m = float(msg.xtrack_error)
    elif msg_type == "MISSION_CURRENT":
        state.mission_seq = int(msg.seq)
    elif msg_type == "MISSION_ITEM_REACHED":
        reached_seq = int(msg.seq)
        if reached_seq != 65535:
            state.reached_seq = max(state.reached_seq, reached_seq)


def segment_metrics(
    point_x_m: float,
    point_y_m: float,
    start_x_m: float,
    start_y_m: float,
    end_x_m: float,
    end_y_m: float,
) -> tuple[float, float]:
    dx = end_x_m - start_x_m
    dy = end_y_m - start_y_m
    segment_len_sq = dx * dx + dy * dy
    if segment_len_sq <= 1e-9:
        return math.hypot(point_x_m - end_x_m, point_y_m - end_y_m), 0.0

    px = point_x_m - start_x_m
    py = point_y_m - start_y_m
    t = clip((px * dx + py * dy) / segment_len_sq, 0.0, 1.0)
    proj_x = start_x_m + t * dx
    proj_y = start_y_m + t * dy
    cross_track = math.hypot(point_x_m - proj_x, point_y_m - proj_y)
    along_track = t * math.sqrt(segment_len_sq)
    return cross_track, along_track


def compute_path_progress(
    spawn_x_m: float,
    spawn_y_m: float,
    mission_waypoints: list[MissionWaypoint],
    target_seq: int,
    x_m: float,
    y_m: float,
) -> tuple[float, float, float]:
    if not mission_waypoints:
        return 0.0, 0.0, 0.0

    points = [(spawn_x_m, spawn_y_m)] + [(wp.x_m, wp.y_m) for wp in mission_waypoints]
    segment_lengths = [
        math.hypot(points[idx + 1][0] - points[idx][0], points[idx + 1][1] - points[idx][1])
        for idx in range(len(mission_waypoints))
    ]
    total_length = max(sum(segment_lengths), 1e-6)
    current_index = int(clip(float(target_seq), 0.0, float(len(mission_waypoints) - 1)))
    start_x_m, start_y_m = points[current_index]
    end_x_m, end_y_m = points[current_index + 1]
    cross_track_m, along_track_m = segment_metrics(x_m, y_m, start_x_m, start_y_m, end_x_m, end_y_m)
    completed_length_m = sum(segment_lengths[:current_index])
    progress_along_path_m = clip(completed_length_m + along_track_m, 0.0, total_length)
    progress_ratio = clip(progress_along_path_m / total_length, 0.0, 1.0)
    return cross_track_m, progress_along_path_m, progress_ratio


def build_sample_row(
    *,
    context: TrialContext,
    state: TelemetryState,
    mission_waypoints: list[MissionWaypoint],
    ros_snapshot: RosTelemetrySnapshot | None,
    raw_waypoint_index: int,
    effective_target_index: int,
    waypoint_completed_count: int,
    waypoint_capture_count: int,
    waypoint_capture_active: bool,
) -> dict[str, Any] | None:
    odom_x_m = ros_snapshot.odom_x_m if ros_snapshot is not None else None
    odom_y_m = ros_snapshot.odom_y_m if ros_snapshot is not None else None
    odom_yaw_rad = ros_snapshot.odom_yaw_rad if ros_snapshot is not None else None
    odom_speed_mps = ros_snapshot.odom_speed_mps if ros_snapshot is not None else None
    mav_x_m = state.x_m
    mav_y_m = state.y_m
    mav_yaw_rad = state.yaw_rad
    mav_speed_mps = state.surge_speed_mps

    if state.sim_time_s is None:
        return None
    if (mav_x_m is None or mav_y_m is None) and (odom_x_m is None or odom_y_m is None):
        return None

    spawn = context.scenario_cfg["spawn"]
    roll_source = selected_roll_source(context)
    requested_position_source = selected_position_source(context)
    mav_position_valid = mav_x_m is not None and mav_y_m is not None
    odom_position_valid = odom_x_m is not None and odom_y_m is not None

    if requested_position_source == "gazebo_odometry" and odom_position_valid:
        position_source_used = "gazebo_odometry"
        position_source_valid = True
        pose_x_m = float(odom_x_m)
        pose_y_m = float(odom_y_m)
        pose_yaw_rad = odom_yaw_rad if odom_yaw_rad is not None else mav_yaw_rad
        pose_speed_mps = odom_speed_mps if odom_speed_mps is not None else mav_speed_mps
    elif requested_position_source == "mavlink" and mav_position_valid:
        position_source_used = "mavlink"
        position_source_valid = True
        pose_x_m = float(mav_x_m)
        pose_y_m = float(mav_y_m)
        pose_yaw_rad = mav_yaw_rad
        pose_speed_mps = mav_speed_mps
    elif odom_position_valid:
        position_source_used = "gazebo_odometry_fallback"
        position_source_valid = False
        pose_x_m = float(odom_x_m)
        pose_y_m = float(odom_y_m)
        pose_yaw_rad = odom_yaw_rad if odom_yaw_rad is not None else mav_yaw_rad
        pose_speed_mps = odom_speed_mps if odom_speed_mps is not None else mav_speed_mps
    else:
        position_source_used = "mavlink_fallback"
        position_source_valid = False
        pose_x_m = float(mav_x_m)
        pose_y_m = float(mav_y_m)
        pose_yaw_rad = mav_yaw_rad
        pose_speed_mps = mav_speed_mps

    normalized_reached_seq = normalize_mission_reached_seq(
        context,
        state.reached_seq,
        len(mission_waypoints),
    )
    target_wp = mission_waypoints[effective_target_index]
    distance_to_wp_m = math.hypot(pose_x_m - target_wp.x_m, pose_y_m - target_wp.y_m)
    cross_track_m, progress_along_path_m, progress_ratio = compute_path_progress(
        float(spawn["x_m"]),
        float(spawn["y_m"]),
        mission_waypoints,
        effective_target_index,
        pose_x_m,
        pose_y_m,
    )
    wind_xyz = context.scenario_cfg["world"]["wind_world_xyz_mps"]
    wind_direction_rad = (
        math.atan2(float(wind_xyz[0]), float(wind_xyz[1]))
        if wind_xyz[0] or wind_xyz[1]
        else 0.0
    )
    wind_from_direction_rad = wrap_pi(wind_direction_rad + math.pi)
    target_bearing_rad = math.atan2(target_wp.x_m - pose_x_m, target_wp.y_m - pose_y_m)
    target_wind_angle_deg = math.degrees(
        abs(wrap_pi(target_bearing_rad - wind_from_direction_rad))
    )
    roll_mav_deg = state.roll_deg
    roll_gz_odom_deg = ros_snapshot.odom_roll_deg if ros_snapshot is not None else None
    roll_gz_imu_deg = ros_snapshot.imu_roll_deg if ros_snapshot is not None else None
    selected_roll_deg = {
        "mavlink": roll_mav_deg,
        "gazebo_odometry": roll_gz_odom_deg,
        "gazebo_imu": roll_gz_imu_deg,
    }[roll_source]
    roll_source_valid = selected_roll_deg is not None

    return {
        "sim_time_s": state.sim_time_s,
        "x_m": pose_x_m,
        "y_m": pose_y_m,
        "elapsed_sim_s": 0.0,
        "elapsed_wall_s": 0.0,
        "realtime_factor": 0.0,
        "yaw_rad": pose_yaw_rad if pose_yaw_rad is not None else 0.0,
        "surge_speed_mps": pose_speed_mps if pose_speed_mps is not None else 0.0,
        "x_mav_m": mav_x_m if mav_x_m is not None else 0.0,
        "y_mav_m": mav_y_m if mav_y_m is not None else 0.0,
        "x_gz_odom_m": odom_x_m if odom_x_m is not None else 0.0,
        "y_gz_odom_m": odom_y_m if odom_y_m is not None else 0.0,
        "yaw_mav_rad": mav_yaw_rad if mav_yaw_rad is not None else 0.0,
        "yaw_gz_odom_rad": odom_yaw_rad if odom_yaw_rad is not None else 0.0,
        "surge_speed_mav_mps": mav_speed_mps if mav_speed_mps is not None else 0.0,
        "surge_speed_gz_odom_mps": odom_speed_mps if odom_speed_mps is not None else 0.0,
        "position_source_used": position_source_used,
        "position_source_valid": position_source_valid,
        "roll_deg": selected_roll_deg if selected_roll_deg is not None else 0.0,
        "roll_mav_deg": roll_mav_deg if roll_mav_deg is not None else 0.0,
        "roll_gz_odom_deg": roll_gz_odom_deg if roll_gz_odom_deg is not None else 0.0,
        "roll_gz_imu_deg": roll_gz_imu_deg if roll_gz_imu_deg is not None else 0.0,
        "roll_source_used": roll_source,
        "roll_source_valid": roll_source_valid,
        "rudder_cmd_rad": state.rudder_cmd_rad if state.rudder_cmd_rad is not None else 0.0,
        "sail_cmd_rad": state.sail_cmd_rad if state.sail_cmd_rad is not None else 0.0,
        "mission_seq_raw": state.mission_seq,
        "mission_reached_raw": state.reached_seq,
        "mission_reached_valid": normalized_reached_seq is not None,
        "waypoint_index_raw": raw_waypoint_index,
        "waypoint_index": effective_target_index,
        "waypoint_completed_count": waypoint_completed_count,
        "waypoint_capture_count": waypoint_capture_count,
        "waypoint_capture_active": waypoint_capture_active,
        "waypoint_captured_this_sample": False,
        "distance_to_wp_m": distance_to_wp_m,
        "cross_track_error_m": cross_track_m,
        "progress_ratio": progress_ratio,
        "progress_along_path_m": progress_along_path_m,
        "nav_wp_dist_m": state.nav_wp_dist_m if state.nav_wp_dist_m is not None else 0.0,
        "nav_xtrack_error_m": state.nav_xtrack_error_m if state.nav_xtrack_error_m is not None else 0.0,
        "wind_speed_mps": math.sqrt(sum(float(v) * float(v) for v in wind_xyz)),
        "wind_direction_rad": wind_direction_rad,
        "wind_from_direction_rad": wind_from_direction_rad,
        "target_bearing_rad": target_bearing_rad,
        "target_wind_angle_deg": target_wind_angle_deg,
    }


def poll_samples(
    master: Any,
    context: TrialContext,
    ros_collector: RosTelemetryCollector | None = None,
    lifecycle_stop_file: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mission_waypoints = build_mission_waypoints(context)
    mission_frame = selected_mission_frame(context)
    home_lat_deg = float(context.study_cfg["home_llh"][0])
    home_lon_deg = float(context.study_cfg["home_llh"][1])
    servo_params = read_param_map(context.files.param_file)
    state = TelemetryState()
    samples: list[dict[str, Any]] = []

    sample_hz = float(context.logging_cfg.get("sample_hz", 10.0))
    sample_period_s = 1.0 / max(sample_hz, 1e-6)
    timeout_s = float(context.termination_cfg.get("timeout_s", 240.0))
    success_radius_m = float(context.termination_cfg.get("success_radius_m", 5.0))
    min_speed_mps = float(context.termination_cfg.get("min_speed_mps", 0.25))
    stuck_window_s = float(context.termination_cfg.get("stuck_window_s", 20.0))
    stuck_min_progress_delta_m = float(
        context.termination_cfg.get("stuck_min_progress_delta_m", 0.5)
    )
    min_progress_delta_m = float(context.termination_cfg.get("min_progress_delta_m", 2.0))
    no_progress_window_s = float(context.termination_cfg.get("no_progress_window_s", 25.0))
    no_progress_margin_m = float(context.termination_cfg.get("no_progress_margin_m", 0.25))
    no_progress_confirm_s = float(context.termination_cfg.get("no_progress_confirm_s", 5.0))
    max_roll_deg = float(context.termination_cfg.get("max_roll_deg", 45.0))
    roll_grace_period_s = float(context.termination_cfg.get("roll_grace_period_s", 0.0))
    roll_violation_window_s = float(context.termination_cfg.get("roll_violation_window_s", 0.0))
    local_waypoint_capture = bool(context.termination_cfg.get("local_waypoint_capture", True))
    waypoint_capture_radius_m = float(
        context.termination_cfg.get("waypoint_capture_radius_m", success_radius_m)
    )
    waypoint_capture_hold_s = float(context.termination_cfg.get("waypoint_capture_hold_s", 1.0))

    sim_start_s: float | None = None
    wall_start_s: float | None = None
    next_sample_s: float | None = None
    last_sample_time_s: float | None = None
    low_speed_continuous_s = 0.0
    stuck_time_total_s = 0.0
    roll_violation_continuous_s = 0.0
    roll_violation_total_s = 0.0
    roll_violation_peak_continuous_s = 0.0
    max_abs_roll_deg_after_grace = 0.0
    no_progress_continuous_s = 0.0
    recent_progress: deque[tuple[float, float]] = deque()
    recent_stuck_progress: deque[tuple[float, float]] = deque()
    waypoint_tracker = WaypointTracker()
    status = "running"
    failure_reason = ""
    external_stop_requested = False
    trust_mavlink_reached = trust_mavlink_reached_for_completion(context)
    use_nav_wp_dist_for_capture = selected_position_source(context) != "gazebo_odometry"

    while True:
        external_stop_reason = read_lifecycle_stop_reason(lifecycle_stop_file)
        if external_stop_reason is not None:
            print(
                "[run_trial] terminating trial: "
                f"reason={external_stop_reason}, requested_by=external_lifecycle"
            )
            status = "failure"
            failure_reason = external_stop_reason
            external_stop_requested = True
            break

        msg = master.recv_match(blocking=True, timeout=0.2)
        if msg is not None:
            update_state_from_message(state, msg, home_lat_deg, home_lon_deg, servo_params, mission_frame)

        if state.sim_time_s is None:
            continue

        if sim_start_s is None:
            sim_start_s = state.sim_time_s
            wall_start_s = time.monotonic()
            next_sample_s = sim_start_s

        if next_sample_s is None or state.sim_time_s + 1e-9 < next_sample_s:
            continue

        waypoint_count = len(mission_waypoints)
        previous_effective_target_index = effective_waypoint_index(
            waypoint_count,
            waypoint_tracker.completed_waypoint_count,
        )
        raw_waypoint_index = mission_seq_to_waypoint_index(
            context,
            state.mission_seq,
            waypoint_count,
        )
        completed_from_reached = (
            completed_waypoint_count_from_reached_seq(
                context,
                state.reached_seq,
                waypoint_count,
            )
            if trust_mavlink_reached
            else 0
        )
        sync_waypoint_tracker(
            waypoint_tracker,
            raw_waypoint_index=raw_waypoint_index,
            completed_waypoint_count_from_reached=completed_from_reached,
            waypoint_count=waypoint_count,
        )
        current_effective_target_index = effective_waypoint_index(
            waypoint_count,
            waypoint_tracker.completed_waypoint_count,
        )
        waypoint_target_changed = current_effective_target_index != previous_effective_target_index
        if current_effective_target_index != previous_effective_target_index:
            waypoint_tracker.within_capture_radius_since_s = None
            print(
                "[run_trial] waypoint target advanced: "
                f"raw_index={raw_waypoint_index}, "
                f"effective_index={current_effective_target_index}, "
                f"completed={waypoint_tracker.completed_waypoint_count}/{waypoint_count}"
            )

        ros_snapshot = ros_collector.snapshot() if ros_collector is not None else None
        sample = build_sample_row(
            context=context,
            state=state,
            mission_waypoints=mission_waypoints,
            ros_snapshot=ros_snapshot,
            raw_waypoint_index=raw_waypoint_index,
            effective_target_index=current_effective_target_index,
            waypoint_completed_count=waypoint_tracker.completed_waypoint_count,
            waypoint_capture_count=waypoint_tracker.local_capture_count,
            waypoint_capture_active=waypoint_tracker.within_capture_radius_since_s is not None,
        )
        if sample is None:
            continue

        dt_s = 0.0

        if last_sample_time_s is not None:
            dt_s = max(0.0, sample["sim_time_s"] - last_sample_time_s)
            if sample["surge_speed_mps"] < min_speed_mps:
                low_speed_continuous_s += dt_s
                stuck_time_total_s += dt_s
            else:
                low_speed_continuous_s = 0.0

        elapsed_sim_s = sample["sim_time_s"] - sim_start_s
        elapsed_wall_s = max(0.0, time.monotonic() - wall_start_s) if wall_start_s is not None else 0.0
        realtime_factor = elapsed_sim_s / max(elapsed_wall_s, 1e-6) if elapsed_wall_s > 0.0 else 0.0
        sample["elapsed_sim_s"] = elapsed_sim_s
        sample["elapsed_wall_s"] = elapsed_wall_s
        sample["realtime_factor"] = realtime_factor

        roll_eligible = elapsed_sim_s >= roll_grace_period_s
        roll_over_limit = bool(sample["roll_source_valid"]) and abs(sample["roll_deg"]) > max_roll_deg
        if roll_eligible and bool(sample["roll_source_valid"]):
            max_abs_roll_deg_after_grace = max(max_abs_roll_deg_after_grace, abs(sample["roll_deg"]))
            if roll_over_limit:
                roll_violation_continuous_s += dt_s
                roll_violation_total_s += dt_s
            else:
                roll_violation_continuous_s = 0.0
        else:
            roll_violation_continuous_s = 0.0
        roll_violation_peak_continuous_s = max(
            roll_violation_peak_continuous_s,
            roll_violation_continuous_s,
        )

        sample["roll_violation_continuous_s"] = roll_violation_continuous_s
        sample["roll_violation_total_s"] = roll_violation_total_s

        captured_this_sample = False
        if local_waypoint_capture:
            capture_distance_m = compute_capture_distance_m(
                local_distance_to_wp_m=float(sample["distance_to_wp_m"]),
                nav_wp_dist_m=state.nav_wp_dist_m,
                raw_waypoint_index=raw_waypoint_index,
                effective_target_index=current_effective_target_index,
                use_nav_wp_dist=use_nav_wp_dist_for_capture,
            )
            previous_completed_count = waypoint_tracker.completed_waypoint_count
            previous_capture_count = waypoint_tracker.local_capture_count
            previous_capture_since_s = waypoint_tracker.within_capture_radius_since_s
            captured_this_sample = maybe_capture_local_waypoint(
                waypoint_tracker,
                waypoint_count=waypoint_count,
                sim_time_s=float(sample["sim_time_s"]),
                distance_to_wp_m=capture_distance_m,
                capture_radius_m=waypoint_capture_radius_m,
                capture_hold_s=waypoint_capture_hold_s,
            )
            if captured_this_sample:
                sample["waypoint_completed_count"] = waypoint_tracker.completed_waypoint_count
                sample["waypoint_capture_count"] = waypoint_tracker.local_capture_count
                sample["waypoint_capture_active"] = False
                sample["waypoint_captured_this_sample"] = True
                print(
                    "[run_trial] locally captured waypoint: "
                    f"target_index={current_effective_target_index}, "
                    f"completed={waypoint_tracker.completed_waypoint_count}/{waypoint_count}, "
                    f"distance_m={float(sample['distance_to_wp_m']):.2f}, "
                    f"nav_wp_dist_m={float(sample['nav_wp_dist_m']):.2f}, "
                    f"capture_distance_m={capture_distance_m:.2f}, "
                    f"sim_time_s={float(sample['sim_time_s']):.1f}"
                )
                next_waypoint_index = effective_waypoint_index(
                    waypoint_count,
                    waypoint_tracker.completed_waypoint_count,
                )
                local_capture_committed = False
                if waypoint_tracker.completed_waypoint_count < waypoint_count:
                    advanced = advance_mission_current_to_waypoint(
                        master,
                        context,
                        state,
                        next_waypoint_index=next_waypoint_index,
                        waypoint_count=waypoint_count,
                        home_lat_deg=home_lat_deg,
                        home_lon_deg=home_lon_deg,
                        servo_params=servo_params,
                    )
                    if advanced:
                        local_capture_committed = True
                        print(
                            "[run_trial] synced ArduPilot mission current after local capture: "
                            f"next_waypoint_index={next_waypoint_index}, "
                            f"mission_seq_raw={state.mission_seq}"
                        )
                    else:
                        print(
                            "[run_trial] warning: local capture advanced tracker, but ArduPilot "
                            f"MISSION_CURRENT did not confirm next_waypoint_index={next_waypoint_index} "
                            "within timeout; trying full-mission reupload fallback"
                        )
                        try:
                            retargeted = retarget_mission_after_local_capture(
                                master,
                                context,
                                mission_waypoints,
                                next_waypoint_index=next_waypoint_index,
                            )
                        except Exception as exc:
                            print(
                                "[run_trial] full-mission reupload fallback failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        else:
                            if retargeted:
                                local_capture_committed = True
                                print(
                                    "[run_trial] full-mission reupload fallback succeeded: "
                                    f"next_waypoint_index={next_waypoint_index}"
                                )
                            else:
                                print(
                                    "[run_trial] full-mission reupload fallback skipped: "
                                    f"invalid next_waypoint_index={next_waypoint_index}"
                                )
                else:
                    local_capture_committed = True

                if not local_capture_committed:
                    waypoint_tracker.completed_waypoint_count = previous_completed_count
                    waypoint_tracker.local_capture_count = previous_capture_count
                    waypoint_tracker.within_capture_radius_since_s = previous_capture_since_s
                    sample["waypoint_completed_count"] = previous_completed_count
                    sample["waypoint_capture_count"] = previous_capture_count
                    sample["waypoint_capture_active"] = previous_capture_since_s is not None
                    sample["waypoint_captured_this_sample"] = False
                    captured_this_sample = False
                    print(
                        "[run_trial] reverted provisional local capture because ArduPilot did not "
                        f"accept the waypoint advance: target_index={current_effective_target_index}, "
                        f"raw_index={raw_waypoint_index}"
                    )

        samples.append(sample)
        last_sample_time_s = sample["sim_time_s"]

        if waypoint_target_changed or captured_this_sample:
            recent_progress.clear()
            recent_stuck_progress.clear()
            no_progress_continuous_s = 0.0

        recent_progress.append((sample["sim_time_s"], sample["progress_along_path_m"]))
        while recent_progress and (sample["sim_time_s"] - recent_progress[0][0]) > no_progress_window_s:
            recent_progress.popleft()

        recent_stuck_progress.append((sample["sim_time_s"], sample["progress_along_path_m"]))
        while recent_stuck_progress and (sample["sim_time_s"] - recent_stuck_progress[0][0]) > stuck_window_s:
            recent_stuck_progress.popleft()

        mavlink_reached_completion = accept_mavlink_reached_for_completion(
            context,
            raw_reached_seq=state.reached_seq,
            waypoint_count=waypoint_count,
            distance_to_wp_m=float(sample["distance_to_wp_m"]),
        )
        if mavlink_reached_completion:
            waypoint_tracker.completed_waypoint_count = max(
                waypoint_tracker.completed_waypoint_count,
                waypoint_count,
            )
            sample["waypoint_completed_count"] = waypoint_tracker.completed_waypoint_count

        mission_complete = mavlink_reached_completion
        if local_waypoint_capture:
            mission_complete = mission_complete or waypoint_tracker.completed_waypoint_count >= waypoint_count
        else:
            mission_complete = mission_complete or (
                sample["waypoint_index"] >= waypoint_count - 1
                and sample["distance_to_wp_m"] <= success_radius_m
            )
        timeout_hit = elapsed_sim_s >= timeout_s
        stuck_hit = False
        if low_speed_continuous_s >= stuck_window_s and recent_stuck_progress:
            stuck_progress_delta_m = (
                recent_stuck_progress[-1][1] - recent_stuck_progress[0][1]
            )
            stuck_hit = stuck_progress_delta_m < stuck_min_progress_delta_m
        no_progress_hit = False
        progress_delta_m = 0.0
        if recent_progress and (sample["sim_time_s"] - recent_progress[0][0]) >= no_progress_window_s:
            progress_delta_m = recent_progress[-1][1] - recent_progress[0][1]
            if (progress_delta_m + no_progress_margin_m) < min_progress_delta_m:
                no_progress_continuous_s += dt_s
            else:
                no_progress_continuous_s = 0.0
            no_progress_hit = no_progress_continuous_s >= no_progress_confirm_s
        else:
            no_progress_continuous_s = 0.0
        excessive_roll_hit = False
        if roll_violation_window_s <= 0.0:
            excessive_roll_hit = roll_eligible and roll_over_limit
        else:
            excessive_roll_hit = roll_violation_continuous_s >= roll_violation_window_s

        if mission_complete:
            print(
                "[run_trial] mission complete: "
                f"sim_time_s={float(sample['sim_time_s']):.1f}, "
                f"completed_waypoints={waypoint_tracker.completed_waypoint_count}/{waypoint_count}, "
                f"distance_to_wp_m={float(sample['distance_to_wp_m']):.2f}, "
                f"x_m={float(sample['x_m']):.2f}, "
                f"y_m={float(sample['y_m']):.2f}, "
                f"position_source={sample['position_source_used']}, "
                f"progress_ratio={float(sample['progress_ratio']):.3f}, "
                f"realtime_percent={100.0 * float(sample['realtime_factor']):.1f}"
            )
            status = "success"
            break
        if timeout_hit:
            print(
                "[run_trial] terminating trial: reason=timeout, "
                f"sim_time_s={float(sample['sim_time_s']):.1f}, "
                f"elapsed_sim_s={elapsed_sim_s:.1f}, "
                f"distance_to_wp_m={float(sample['distance_to_wp_m']):.2f}, "
                f"x_m={float(sample['x_m']):.2f}, "
                f"y_m={float(sample['y_m']):.2f}, "
                f"position_source={sample['position_source_used']}, "
                f"progress_ratio={float(sample['progress_ratio']):.3f}, "
                f"realtime_percent={100.0 * float(sample['realtime_factor']):.1f}"
            )
            status = "failure"
            failure_reason = "timeout"
            break
        if stuck_hit:
            stuck_progress_delta_m = (
                recent_stuck_progress[-1][1] - recent_stuck_progress[0][1]
                if len(recent_stuck_progress) >= 2
                else 0.0
            )
            print(
                "[run_trial] terminating trial: reason=stuck_low_speed, "
                f"sim_time_s={float(sample['sim_time_s']):.1f}, "
                f"low_speed_continuous_s={low_speed_continuous_s:.1f}, "
                f"stuck_progress_delta_m={stuck_progress_delta_m:.2f}, "
                f"distance_to_wp_m={float(sample['distance_to_wp_m']):.2f}, "
                f"x_m={float(sample['x_m']):.2f}, "
                f"y_m={float(sample['y_m']):.2f}, "
                f"position_source={sample['position_source_used']}, "
                f"progress_ratio={float(sample['progress_ratio']):.3f}, "
                f"realtime_percent={100.0 * float(sample['realtime_factor']):.1f}"
            )
            status = "failure"
            failure_reason = "stuck_low_speed"
            break
        if no_progress_hit:
            print(
                "[run_trial] terminating trial: reason=no_progress, "
                f"sim_time_s={float(sample['sim_time_s']):.1f}, "
                f"window_progress_delta_m={progress_delta_m:.2f}, "
                f"required_progress_delta_m={min_progress_delta_m:.2f}, "
                f"margin_m={no_progress_margin_m:.2f}, "
                f"confirm_s={no_progress_continuous_s:.1f}/{no_progress_confirm_s:.1f}, "
                f"distance_to_wp_m={float(sample['distance_to_wp_m']):.2f}, "
                f"x_m={float(sample['x_m']):.2f}, "
                f"y_m={float(sample['y_m']):.2f}, "
                f"position_source={sample['position_source_used']}, "
                f"progress_ratio={float(sample['progress_ratio']):.3f}, "
                f"realtime_percent={100.0 * float(sample['realtime_factor']):.1f}"
            )
            status = "failure"
            failure_reason = "no_progress"
            break
        if excessive_roll_hit:
            print(
                "[run_trial] terminating trial: reason=excessive_roll, "
                f"sim_time_s={float(sample['sim_time_s']):.1f}, "
                f"roll_deg={float(sample['roll_deg']):.2f}, "
                f"roll_violation_continuous_s={roll_violation_continuous_s:.1f}, "
                f"distance_to_wp_m={float(sample['distance_to_wp_m']):.2f}, "
                f"x_m={float(sample['x_m']):.2f}, "
                f"y_m={float(sample['y_m']):.2f}, "
                f"position_source={sample['position_source_used']}, "
                f"realtime_percent={100.0 * float(sample['realtime_factor']):.1f}"
            )
            status = "failure"
            failure_reason = "excessive_roll"
            break

        next_sample_s += sample_period_s

    final_realtime_factor = 0.0
    min_realtime_factor = 0.0
    if samples:
        final_elapsed_sim_s = max(0.0, float(samples[-1].get("elapsed_sim_s", 0.0)))
        final_elapsed_wall_s = max(0.0, float(samples[-1].get("elapsed_wall_s", 0.0)))
        if final_elapsed_wall_s > 0.0:
            final_realtime_factor = final_elapsed_sim_s / final_elapsed_wall_s

        valid_realtime_factors = [
            float(sample.get("realtime_factor", 0.0))
            for sample in samples
            if float(sample.get("elapsed_sim_s", 0.0)) >= sample_period_s
            and float(sample.get("elapsed_wall_s", 0.0)) >= 0.5
            and float(sample.get("realtime_factor", 0.0)) > 0.0
        ]
        min_realtime_factor = (
            min(valid_realtime_factors)
            if valid_realtime_factors
            else final_realtime_factor
        )

    return samples, {
        "status": status,
        "failure_reason": failure_reason,
        "mission_complete": status == "success",
        "mission_complete_explicit": True,
        "timeout": failure_reason == "timeout",
        "stuck": failure_reason == "stuck_low_speed",
        "no_progress": failure_reason == "no_progress",
        "excessive_roll": failure_reason == "excessive_roll",
        "external_stop": external_stop_requested,
        "stuck_time_s": stuck_time_total_s,
        "roll_violation_total_s": roll_violation_total_s,
        "roll_violation_peak_continuous_s": roll_violation_peak_continuous_s,
        "max_abs_roll_deg_after_grace": max_abs_roll_deg_after_grace,
        "final_realtime_factor": final_realtime_factor,
        "mean_realtime_factor": final_realtime_factor,
        "min_realtime_factor": min_realtime_factor,
    }


def run_live_trial(
    context: TrialContext,
    execution: ExecutionConfig,
    launch_plan: LaunchPlan,
    *,
    lifecycle_ready_file: Path | None = None,
    lifecycle_stop_file: Path | None = None,
    lifecycle_status_file: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    process: subprocess.Popen[str] | None = None
    master = None
    ros_collector: RosTelemetryCollector | None = None
    mavlink_endpoint = str(context.study_cfg["mavlink_endpoint"])
    mavlink_connect_timeout_s = float(context.study_cfg.get("mavlink_connect_timeout_s", 90.0))
    mavlink_release_timeout_s = float(context.study_cfg.get("mavlink_release_timeout_s", 15.0))
    post_trial_cooldown_s = float(context.study_cfg.get("post_trial_cooldown_s", 5.0))
    try:
        print(f"[run_trial] launching trial {context.trial_id}")
        print(f"[run_trial] mission frame: {selected_mission_frame(context)}")
        print(f"[run_trial] position source: {selected_position_source(context)}")
        print(f"[run_trial] roll source: {selected_roll_source(context)}")
        print(f"[run_trial] launch stdout: {context.files.logs_dir / 'launch.stdout.log'}")
        print(f"[run_trial] launch stderr: {context.files.logs_dir / 'launch.stderr.log'}")
        process = start_launch_process(context, launch_plan)
        publish_sail_force_enabled(context, execution, False, reason="startup_prearm")
        if ros_roll_collection_required(context):
            use_odom, use_imu = ros_roll_topic_requirements(context)
            ros_collector = RosTelemetryCollector(
                str(context.study_cfg["namespace"]),
                use_odom=use_odom,
                use_imu=use_imu,
            )
            ros_collector.start()
        master = connect_mavlink(
            mavlink_endpoint,
            timeout_s=mavlink_connect_timeout_s,
            launch_process=process,
            stdout_log=context.files.logs_dir / "launch.stdout.log",
            stderr_log=context.files.logs_dir / "launch.stderr.log",
        )
        time.sleep(3.0)
        mission_waypoints = build_mission_waypoints(context)
        upload_mission(master, context, mission_waypoints)
        arm_and_set_auto(master, context)
        publish_sail_force_enabled(context, execution, True, reason="armed_auto")
        enable_settle_s = float(context.study_cfg.get("sail_force_enable_settle_s", 0.5))
        if enable_settle_s > 0.0:
            time.sleep(enable_settle_s)
        write_json_marker(
            lifecycle_ready_file,
            {
                "ready": True,
                "trial_id": context.trial_id,
                "summary_json": str(context.files.summary_json),
                "runner_pid": os.getpid(),
                "ready_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        samples, outcome = poll_samples(
            master,
            context,
            ros_collector,
            lifecycle_stop_file=lifecycle_stop_file,
        )
        write_json_marker(
            lifecycle_status_file,
            {
                "trial_id": context.trial_id,
                "summary_json": str(context.files.summary_json),
                "outcome": outcome,
                "status_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return samples, outcome
    finally:
        if ros_collector is not None:
            ros_collector.close()
        if master is not None:
            try:
                master.close()
            except Exception:
                pass
        stop_launch_process(process, execution, launch_plan)
        if not wait_for_endpoint_closed(mavlink_endpoint, mavlink_release_timeout_s):
            print(
                f"[run_trial] warning: MAVLink endpoint still appeared open after cleanup: "
                f"{mavlink_endpoint}"
            )
        if post_trial_cooldown_s > 0.0:
            print(f"[run_trial] cooldown after trial cleanup: {post_trial_cooldown_s:.1f}s")
            time.sleep(post_trial_cooldown_s)


def empty_live_outcome(reason: str) -> dict[str, Any]:
    return {
        "status": "not_started",
        "failure_reason": reason,
        "mission_complete": False,
        "mission_complete_explicit": True,
        "timeout": False,
        "stuck": False,
        "no_progress": False,
        "excessive_roll": False,
        "external_stop": False,
        "stuck_time_s": 0.0,
        "roll_violation_total_s": 0.0,
        "roll_violation_peak_continuous_s": 0.0,
        "max_abs_roll_deg_after_grace": 0.0,
        "final_realtime_factor": 0.0,
        "mean_realtime_factor": 0.0,
        "min_realtime_factor": 0.0,
    }


def build_dry_run_samples() -> list[dict[str, Any]]:
    return []


def build_metadata(
    context: TrialContext,
    finished_at_utc: str,
    duration_wall_s: float,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    spawn = context.scenario_cfg["spawn"]
    world = context.scenario_cfg["world"]
    waypoints = context.scenario_cfg["mission"]["waypoints"]
    wind_xyz = world["wind_world_xyz_mps"]
    gazebo_spawn_yaw_deg = selected_gazebo_spawn_yaw_deg(spawn)
    ardupilot_heading_deg = selected_ardupilot_home_heading_deg(spawn)
    metadata = {
        "trial_id": context.trial_id,
        "study_name": context.study_name,
        "scenario_id": context.scenario_id,
        "scenario_split": context.scenario_split,
        "repeat_idx": context.repeat_idx,
        "status": outcome["status"],
        "failure_reason": outcome["failure_reason"],
        "started_at_utc": context.started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_wall_s": duration_wall_s,
        "samples_csv": str(context.files.samples_csv),
        "param_file": str(context.files.param_file),
        "world_file": str(context.files.world_file),
        "spawn_x_m": spawn["x_m"],
        "spawn_y_m": spawn["y_m"],
        "spawn_z_m": spawn["z_m"],
        "spawn_yaw_deg": gazebo_spawn_yaw_deg,
        "spawn_course_deg": spawn["yaw_deg"],
        "spawn_heading_deg": ardupilot_heading_deg,
        "wind_x_mps": wind_xyz[0],
        "wind_y_mps": wind_xyz[1],
        "wind_z_mps": wind_xyz[2],
        "mission_waypoint_count": len(waypoints),
        "mission_frame": selected_mission_frame(context),
        "position_source": selected_position_source(context),
        "roll_source": selected_roll_source(context),
    }
    if outcome.get("exception_text"):
        metadata["exception_text"] = str(outcome["exception_text"])
    return metadata


def write_samples_csv(csv_path: Path, fields: Iterable[str], samples: list[dict[str, Any]]) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key) for key in fieldnames})


def write_summary_json(
    context: TrialContext,
    row: dict[str, Any],
    metrics: dict[str, Any],
    constraints: dict[str, Any],
    objective: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "metadata": metadata,
        "summary_row": row,
        "metrics": metrics,
        "constraints": constraints,
        "objective": objective,
    }
    context.files.summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo_root = get_repo_root()
    workspace_root = get_workspace_root(repo_root)
    lifecycle_ready_file = (
        resolve_repo_path(repo_root, args.lifecycle_ready_file)
        if args.lifecycle_ready_file
        else None
    )
    lifecycle_stop_file = (
        resolve_repo_path(repo_root, args.lifecycle_stop_file)
        if args.lifecycle_stop_file
        else None
    )
    lifecycle_status_file = (
        resolve_repo_path(repo_root, args.lifecycle_status_file)
        if args.lifecycle_status_file
        else None
    )
    scenario_path = resolve_repo_path(repo_root, args.scenario)
    config = load_yaml(scenario_path)
    execution = build_execution_config(args, config)
    validate_execution_backend(execution)
    scenario_cfg = get_scenario(config, args.scenario_id)
    if args.params is not None:
        params = parse_param_overrides(args.params)
    else:
        params_file = args.params_file or str(DEFAULT_PARAMS_PATH)
        params = load_param_overrides_from_file(resolve_repo_path(repo_root, params_file))
    search_space_map = get_search_space_map(config)
    validation_errors = validate_param_overrides(search_space_map, params)
    if validation_errors:
        raise SystemExit("\n".join(validation_errors))

    results_dir = resolve_repo_path(repo_root, args.results_dir)
    context = build_context(config, scenario_cfg, params, repo_root, workspace_root, results_dir, args.repeat_idx)
    write_trial_manifest(context)
    write_trial_param_file(
        resolve_repo_path(repo_root, context.study_cfg["base_param_file"]),
        context.params,
        context.files.param_file,
    )
    trial_world_name = render_trial_world(context)
    launch_plan = build_launch_plan(
        context,
        execution,
        trial_world_name,
        prepare_remote_inputs=not args.dry_run,
    )

    start_monotonic = time.monotonic()
    exit_code = 0
    if args.dry_run:
        samples = build_dry_run_samples()
        outcome = empty_live_outcome("dry_run")
        outcome["status"] = "dry_run"
    else:
        try:
            samples, outcome = run_live_trial(
                context,
                execution,
                launch_plan,
                lifecycle_ready_file=lifecycle_ready_file,
                lifecycle_stop_file=lifecycle_stop_file,
                lifecycle_status_file=lifecycle_status_file,
            )
        except Exception as exc:
            samples = []
            outcome = empty_live_outcome("runtime_exception")
            outcome["status"] = "failure"
            outcome["failure_reason"] = "runtime_exception"
            outcome["exception_text"] = f"{type(exc).__name__}: {exc}"
            exit_code = 1

    write_json_marker(
        lifecycle_status_file,
        {
            "trial_id": context.trial_id,
            "summary_json": str(context.files.summary_json),
            "outcome": outcome,
            "status_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    finished_at_utc = datetime.now(timezone.utc).isoformat()
    duration_wall_s = time.monotonic() - start_monotonic
    write_samples_csv(context.files.samples_csv, context.logging_cfg.get("csv_fields", []), samples)

    metrics = compute_metrics(
        samples,
        online_stats=outcome,
        sailing_cfg={
            "no_go_angle_deg": context.params.get("SAIL_NO_GO_ANGLE", 60.0),
            "no_go_grace_s": 3.0,
            "tack_min_hold_s": 3.0,
            "min_course_speed_mps": max(
                0.05,
                0.5 * float(context.termination_cfg.get("min_speed_mps", 0.25)),
            ),
        },
    )
    constraints = compute_constraints(metrics, context.termination_cfg, outcome)
    objective = compute_objective(metrics, constraints, context.termination_cfg)
    metadata = build_metadata(context, finished_at_utc, duration_wall_s, outcome)
    row = build_summary_row(
        param_names=context.param_names,
        metadata=metadata,
        params=context.params,
        metrics=metrics,
        constraints=constraints,
        objective=objective,
    )

    summary_columns = build_summary_columns(context.param_names)
    append_summary_row(context.results_dir / "summary.csv", row, summary_columns)
    write_summary_json(context, row, metrics, constraints, objective, metadata)

    print("trial_id:", context.trial_id)
    print("execution_backend:", execution.backend)
    if execution.docker_container:
        print("docker_container:", execution.docker_container)
    print("launch_cmd:", launch_plan.display_cmd)
    print("summary_csv:", context.results_dir / "summary.csv")
    print("summary_json:", context.files.summary_json)
    print("status:", row["status"])
    print("failure_reason:", row["failure_reason"])
    print("progress_ratio:", f"{float(row['progress_ratio']):.3f}")
    print("final_distance_to_wp_m:", f"{float(row['final_distance_to_wp_m']):.2f}")
    print("mission_time_s:", f"{float(row['mission_time_s']):.1f}")
    print("mean_realtime_percent:", f"{100.0 * float(row['mean_realtime_factor']):.1f}")
    print("final_realtime_percent:", f"{100.0 * float(row['final_realtime_factor']):.1f}")
    print("min_realtime_percent:", f"{100.0 * float(row['min_realtime_factor']):.1f}")
    if outcome.get("exception_text"):
        print("exception:", outcome["exception_text"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
