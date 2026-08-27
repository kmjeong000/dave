#!/usr/bin/env python3
"""Launch held-pose cases and compare Gazebo attitude with MAVLink attitude.

This command is intentionally diagnostic-only: it never uploads a mission,
arms the vehicle, enables sail force, or changes the model SDF. It launches a
fresh Gazebo / ArduPilot process for each case, repeatedly applies the requested
pose through Gazebo's ``set_pose`` service, records both attitude sources, then
uses the pure transforms in :mod:`experiments.sailboat_physics.frames`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .frames import (
    AttitudeDeg,
    FrameSample,
    frd_attitude_to_gazebo_euler_deg,
    frd_attitude_to_gazebo_quaternion,
    gazebo_quaternion_to_frd_attitude_deg,
    normalize_degrees_360,
    summarize_frame_case,
)


DEFAULT_SCENARIO = "experiments/sailboat_BO/scenario.yaml"
DEFAULT_PARAMS = "experiments/sailboat_BO/baseline_params.json"
DEFAULT_RESULTS = "experiments/sailboat_physics/results/frame_checks"


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    expected: AttitudeDeg


@dataclass(frozen=True)
class PoseSnapshot:
    quaternion_xyzw: tuple[float, float, float, float] | None = None
    position_xyz_m: tuple[float, float, float] | None = None
    received_monotonic_s: float | None = None


class RosPoseCollector:
    """Collect the complete model quaternion without interpreting its axes."""

    def __init__(self, namespace: str):
        self.namespace = namespace.strip("/")
        self._lock = threading.Lock()
        self._snapshot = PoseSnapshot()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._node = None
        self._rclpy = None
        self._did_init = False

    def start(self) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
        except ImportError as exc:
            raise RuntimeError(
                "frame diagnostic requires rclpy and nav_msgs inside the simulation container"
            ) from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._did_init = True
        node_name = f"sailboat_frame_check_{os.getpid()}_{int(time.time() * 1000)}"
        self._node = Node(node_name)
        topic = f"/model/{self.namespace}/odometry"
        self._node.create_subscription(Odometry, topic, self._on_odometry, 10)
        self._thread = threading.Thread(target=self._spin, name=node_name, daemon=True)
        self._thread.start()
        print(f"[frame_check] subscribed: {topic}")

    def _spin(self) -> None:
        assert self._node is not None
        assert self._rclpy is not None
        while not self._stop_event.is_set():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def _on_odometry(self, message: Any) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        snapshot = PoseSnapshot(
            quaternion_xyzw=(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
            position_xyz_m=(
                float(position.x),
                float(position.y),
                float(position.z),
            ),
            received_monotonic_s=time.monotonic(),
        )
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> PoseSnapshot:
        with self._lock:
            return self._snapshot

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gazebo ENU/+Y-forward attitude with ArduPilot NED/FRD attitude"
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--base-scenario-id", default="train_crosswind_straight")
    parser.add_argument("--params-file", default=DEFAULT_PARAMS)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--label", default="pre_fix")
    parser.add_argument(
        "--execution-backend",
        choices=["local"],
        default="local",
        help="Run directly in the current dave:sailboat-rl container",
    )
    parser.add_argument(
        "--headings",
        nargs="+",
        type=float,
        default=[0.0, 90.0, 180.0, 270.0],
        help="Compass headings in degrees",
    )
    parser.add_argument(
        "--roll-tests",
        nargs="*",
        type=float,
        default=[-10.0, 10.0],
        help="Body-FRD roll tests in degrees",
    )
    parser.add_argument(
        "--pitch-tests",
        nargs="*",
        type=float,
        default=[-10.0, 10.0],
        help="Body-FRD pitch tests in degrees",
    )
    parser.add_argument("--sample-duration-s", type=float, default=4.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--pose-hold-period-s", type=float, default=0.5)
    parser.add_argument("--pose-service-timeout-ms", type=int, default=3000)
    parser.add_argument("--mavlink-connect-timeout-s", type=float, default=120.0)
    parser.add_argument("--sitl-start-delay-s", type=float, default=10.0)
    parser.add_argument("--heading-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--attitude-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 when data are complete but one or more frame gates fail",
    )
    return parser.parse_args()


def safe_label(value: str) -> str:
    rendered = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return rendered.strip("_-") or "frame_check"


def build_cases(
    headings: Sequence[float],
    roll_tests: Sequence[float],
    pitch_tests: Sequence[float],
) -> list[DiagnosticCase]:
    cases: list[DiagnosticCase] = []
    for heading in headings:
        normalized = normalize_degrees_360(heading)
        cases.append(
            DiagnosticCase(
                name=f"heading_{normalized:06.1f}".replace(".", "p"),
                expected=AttitudeDeg(roll_deg=0.0, pitch_deg=0.0, heading_deg=normalized),
            )
        )
    for roll in roll_tests:
        cases.append(
            DiagnosticCase(
                name=f"roll_{roll:+05.1f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p"),
                expected=AttitudeDeg(roll_deg=float(roll), pitch_deg=0.0, heading_deg=0.0),
            )
        )
    for pitch in pitch_tests:
        cases.append(
            DiagnosticCase(
                name=f"pitch_{pitch:+05.1f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p"),
                expected=AttitudeDeg(roll_deg=0.0, pitch_deg=float(pitch), heading_deg=0.0),
            )
        )
    return cases


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def make_case_config(
    base_config: dict[str, Any],
    base_scenario: dict[str, Any],
    case: DiagnosticCase,
    *,
    sitl_start_delay_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base_config)
    scenario = copy.deepcopy(base_scenario)
    scenario["id"] = f"frame_{case.name}"
    scenario["split"] = "diagnostic"

    world = scenario.setdefault("world", {})
    world["wind_world_xyz_mps"] = [0.0, 0.0, 0.0]
    world["wind_mag_noise_stddev"] = 0.0
    world["wind_dir_noise_stddev"] = 0.0
    world["wind_mag_sin_amp_percent"] = 0.0
    world["wind_dir_sin_amp_deg"] = 0.0

    gazebo_roll_deg, gazebo_pitch_deg, _ = frd_attitude_to_gazebo_euler_deg(
        case.expected.roll_deg,
        case.expected.pitch_deg,
        case.expected.heading_deg,
    )
    spawn = scenario.setdefault("spawn", {})
    spawn["heading_deg"] = case.expected.heading_deg
    spawn["gazebo_roll_deg"] = gazebo_roll_deg
    spawn["gazebo_pitch_deg"] = gazebo_pitch_deg

    study = config["study"]
    study["gui"] = False
    study["headless"] = True
    study["paused"] = False
    study["use_mavproxy"] = False
    study["use_mavros"] = False
    study["sitl_start_delay_s"] = float(sitl_start_delay_s)
    study["post_trial_cooldown_s"] = 0.0
    return config, scenario


def set_model_pose(
    *,
    world_name: str,
    model_name: str,
    position_xyz_m: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
    timeout_ms: int,
) -> None:
    x_m, y_m, z_m = position_xyz_m
    qx, qy, qz, qw = quaternion_xyzw
    request = (
        f'name: "{model_name}", '
        f"position: {{x: {x_m:.12g}, y: {y_m:.12g}, z: {z_m:.12g}}}, "
        f"orientation: {{x: {qx:.12g}, y: {qy:.12g}, z: {qz:.12g}, w: {qw:.12g}}}"
    )
    command = [
        "gz",
        "service",
        "-s",
        f"/world/{world_name}/set_pose",
        "--reqtype",
        "gz.msgs.Pose",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        str(int(timeout_ms)),
        "--req",
        request,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    combined = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0 or "true" not in combined.lower():
        raise RuntimeError(
            "Gazebo set_pose failed: "
            f"returncode={completed.returncode}, response={combined or '<empty>'}"
        )


def collect_samples(
    *,
    master: Any,
    collector: RosPoseCollector,
    case: DiagnosticCase,
    world_name: str,
    model_name: str,
    position_xyz_m: tuple[float, float, float],
    sample_duration_s: float,
    sample_hz: float,
    pose_hold_period_s: float,
    pose_service_timeout_ms: int,
) -> list[FrameSample]:
    if sample_duration_s <= 0.0 or sample_hz <= 0.0 or pose_hold_period_s <= 0.0:
        raise ValueError("sample duration, sample rate, and pose hold period must be positive")

    target_quaternion = frd_attitude_to_gazebo_quaternion(
        case.expected.roll_deg,
        case.expected.pitch_deg,
        case.expected.heading_deg,
    )
    start = time.monotonic()
    deadline = start + sample_duration_s
    next_sample = start
    next_pose_hold = start
    latest_mav_attitude: AttitudeDeg | None = None
    latest_mav_attitude_rad: tuple[float, float, float] | None = None
    samples: list[FrameSample] = []

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_pose_hold:
            set_model_pose(
                world_name=world_name,
                model_name=model_name,
                position_xyz_m=position_xyz_m,
                quaternion_xyzw=target_quaternion,
                timeout_ms=pose_service_timeout_ms,
            )
            next_pose_hold = time.monotonic() + pose_hold_period_s

        message = master.recv_match(type="ATTITUDE", blocking=True, timeout=0.1)
        if message is not None:
            latest_mav_attitude_rad = (
                float(message.roll),
                float(message.pitch),
                float(message.yaw),
            )
            latest_mav_attitude = AttitudeDeg(
                roll_deg=math.degrees(latest_mav_attitude_rad[0]),
                pitch_deg=math.degrees(latest_mav_attitude_rad[1]),
                heading_deg=normalize_degrees_360(math.degrees(latest_mav_attitude_rad[2])),
            )

        now = time.monotonic()
        if now < next_sample:
            time.sleep(min(0.01, next_sample - now))
            continue
        next_sample += 1.0 / sample_hz

        ros_snapshot = collector.snapshot()
        if (
            ros_snapshot.quaternion_xyzw is None
            or latest_mav_attitude is None
            or latest_mav_attitude_rad is None
        ):
            continue
        if (
            ros_snapshot.received_monotonic_s is None
            or now - ros_snapshot.received_monotonic_s > 1.0
        ):
            continue
        gazebo_attitude = gazebo_quaternion_to_frd_attitude_deg(
            *ros_snapshot.quaternion_xyzw
        )
        qx, qy, qz, qw = ros_snapshot.quaternion_xyzw
        samples.append(
            FrameSample(
                elapsed_wall_s=now - start,
                gazebo_roll_deg=gazebo_attitude.roll_deg,
                gazebo_pitch_deg=gazebo_attitude.pitch_deg,
                gazebo_heading_deg=gazebo_attitude.heading_deg,
                mav_roll_deg=latest_mav_attitude.roll_deg,
                mav_pitch_deg=latest_mav_attitude.pitch_deg,
                mav_heading_deg=latest_mav_attitude.heading_deg,
                gazebo_qx=qx,
                gazebo_qy=qy,
                gazebo_qz=qz,
                gazebo_qw=qw,
                mav_roll_rad=latest_mav_attitude_rad[0],
                mav_pitch_rad=latest_mav_attitude_rad[1],
                mav_yaw_rad=latest_mav_attitude_rad[2],
            )
        )
    return samples


def write_samples(path: Path, samples: Sequence[FrameSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(samples[0]).keys()) if samples else [
        "elapsed_wall_s",
        "gazebo_roll_deg",
        "gazebo_pitch_deg",
        "gazebo_heading_deg",
        "mav_roll_deg",
        "mav_pitch_deg",
        "mav_heading_deg",
        "gazebo_qx",
        "gazebo_qy",
        "gazebo_qz",
        "gazebo_qw",
        "mav_roll_rad",
        "mav_pitch_rad",
        "mav_yaw_rad",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def write_overall_csv(path: Path, case_results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "status",
        "passed",
        "sample_count",
        "expected_roll_deg",
        "expected_pitch_deg",
        "expected_heading_deg",
        "gazebo_roll_deg",
        "gazebo_pitch_deg",
        "gazebo_heading_deg",
        "mav_roll_deg",
        "mav_pitch_deg",
        "mav_heading_deg",
        "mav_gazebo_roll_error_deg",
        "mav_gazebo_pitch_error_deg",
        "mav_gazebo_heading_error_deg",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in case_results:
            summary = result.get("summary") or {}
            expected = summary.get("expected") or {}
            mean = summary.get("mean") or {}
            error = summary.get("error") or {}
            writer.writerow(
                {
                    "case": result["case"],
                    "status": result["status"],
                    "passed": summary.get("passed", False),
                    "sample_count": summary.get("sample_count", 0),
                    "expected_roll_deg": expected.get("roll_deg"),
                    "expected_pitch_deg": expected.get("pitch_deg"),
                    "expected_heading_deg": expected.get("heading_deg"),
                    "gazebo_roll_deg": mean.get("gazebo_roll_deg"),
                    "gazebo_pitch_deg": mean.get("gazebo_pitch_deg"),
                    "gazebo_heading_deg": mean.get("gazebo_heading_deg"),
                    "mav_roll_deg": mean.get("mav_roll_deg"),
                    "mav_pitch_deg": mean.get("mav_pitch_deg"),
                    "mav_heading_deg": mean.get("mav_heading_deg"),
                    "mav_gazebo_roll_error_deg": error.get("mav_gazebo_roll_deg"),
                    "mav_gazebo_pitch_error_deg": error.get("mav_gazebo_pitch_deg"),
                    "mav_gazebo_heading_error_deg": error.get("mav_gazebo_heading_deg"),
                    "error": result.get("error", ""),
                }
            )


def main() -> int:
    args = parse_args()
    # Runtime-only imports keep pure coordinate tests independent of ROS,
    # Gazebo and pymavlink availability.
    from experiments.sailboat_BO.common import load_yaml
    from experiments.sailboat_BO.run_trial import (
        ExecutionConfig,
        build_context,
        build_launch_plan,
        connect_mavlink,
        get_repo_root,
        get_scenario,
        get_workspace_root,
        is_tcp_endpoint_open,
        load_param_overrides_from_file,
        publish_sail_force_enabled,
        render_trial_world,
        start_launch_process,
        stop_launch_process,
        wait_for_endpoint_closed,
        write_trial_manifest,
        write_trial_param_file,
    )

    repo_root = get_repo_root()
    workspace_root = get_workspace_root(repo_root)
    missing_commands = [name for name in ("gz", "ros2", "ardurover") if shutil.which(name) is None]
    if missing_commands:
        raise SystemExit(
            "Frame diagnostics must run inside the simulation container; "
            f"missing commands: {', '.join(missing_commands)}"
        )
    config = load_yaml(resolve_path(repo_root, args.scenario))
    base_scenario = get_scenario(config, args.base_scenario_id)
    params = load_param_overrides_from_file(resolve_path(repo_root, args.params_file))
    root_results = resolve_path(repo_root, args.results_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root_results / f"{stamp}__{safe_label(args.label)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run_dir: {run_dir}")

    execution = ExecutionConfig(backend=args.execution_backend)
    endpoint = str(config["study"]["mavlink_endpoint"])
    if is_tcp_endpoint_open(endpoint):
        raise SystemExit(
            f"MAVLink endpoint is already in use: {endpoint}. "
            "Stop the existing BO/Gazebo/SITL process before running frame diagnostics."
        )

    cases = build_cases(args.headings, args.roll_tests, args.pitch_tests)
    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        print(f"[frame_check] case {index + 1}/{len(cases)}: {case.name}")
        case_config, scenario = make_case_config(
            config,
            base_scenario,
            case,
            sitl_start_delay_s=args.sitl_start_delay_s,
        )
        case_results_dir = run_dir / "cases"
        context = build_context(
            case_config,
            scenario,
            params,
            repo_root,
            workspace_root,
            case_results_dir,
            index,
        )
        write_trial_manifest(context)
        write_trial_param_file(
            resolve_path(repo_root, context.study_cfg["base_param_file"]),
            context.params,
            context.files.param_file,
        )
        trial_world_name = render_trial_world(context)
        launch_plan = build_launch_plan(
            context,
            execution,
            trial_world_name,
            prepare_remote_inputs=True,
        )

        process = None
        master = None
        collector: RosPoseCollector | None = None
        samples: list[FrameSample] = []
        endpoint_closed = True
        result: dict[str, Any] = {
            "case": case.name,
            "expected": asdict(case.expected),
            "status": "runtime_error",
            "trial_id": context.trial_id,
            "trial_dir": str(context.files.trial_dir),
        }
        try:
            process = start_launch_process(context, launch_plan)
            publish_sail_force_enabled(context, execution, False, reason="frame_diagnostic")
            collector = RosPoseCollector(str(context.study_cfg["namespace"]))
            collector.start()
            master = connect_mavlink(
                endpoint,
                timeout_s=float(args.mavlink_connect_timeout_s),
                launch_process=process,
                stdout_log=context.files.logs_dir / "launch.stdout.log",
                stderr_log=context.files.logs_dir / "launch.stderr.log",
            )
            spawn = scenario["spawn"]
            samples = collect_samples(
                master=master,
                collector=collector,
                case=case,
                world_name=str(scenario["world"].get("internal_world_name", "waves")),
                model_name=str(context.study_cfg["namespace"]),
                position_xyz_m=(
                    float(spawn["x_m"]),
                    float(spawn["y_m"]),
                    float(spawn["z_m"]),
                ),
                sample_duration_s=float(args.sample_duration_s),
                sample_hz=float(args.sample_hz),
                pose_hold_period_s=float(args.pose_hold_period_s),
                pose_service_timeout_ms=int(args.pose_service_timeout_ms),
            )
            summary = summarize_frame_case(
                samples,
                expected=case.expected,
                heading_tolerance_deg=float(args.heading_tolerance_deg),
                attitude_tolerance_deg=float(args.attitude_tolerance_deg),
                minimum_samples=int(args.minimum_samples),
            )
            result["status"] = "complete"
            result["summary"] = summary
            print(
                f"[frame_check] {case.name}: passed={summary['passed']}, "
                f"samples={summary['sample_count']}, error={summary['error']}"
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[frame_check] {case.name}: ERROR: {result['error']}")
        finally:
            write_samples(context.files.raw_dir / "frame_samples.csv", samples)
            if collector is not None:
                collector.close()
            if master is not None:
                try:
                    master.close()
                except Exception:
                    pass
            stop_launch_process(process, execution, launch_plan)
            endpoint_closed = wait_for_endpoint_closed(endpoint, 15.0)
            if not endpoint_closed:
                result["status"] = "cleanup_error"
                result["error"] = f"MAVLink endpoint remained open after cleanup: {endpoint}"
        case_results.append(result)
        if not endpoint_closed:
            print(f"[frame_check] aborting remaining cases: {result['error']}")
            break

    runtime_complete = len(case_results) == len(cases) and all(
        result["status"] == "complete" for result in case_results
    )
    gates_passed = runtime_complete and all(
        bool(result.get("summary", {}).get("passed", False)) for result in case_results
    )
    payload = {
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source_scenario": str(resolve_path(repo_root, args.scenario)),
        "base_scenario_id": args.base_scenario_id,
        "params_file": str(resolve_path(repo_root, args.params_file)),
        "runtime_complete": runtime_complete,
        "all_frame_gates_passed": gates_passed,
        "tolerances": {
            "heading_deg": args.heading_tolerance_deg,
            "attitude_deg": args.attitude_tolerance_deg,
            "minimum_samples": args.minimum_samples,
        },
        "cases": case_results,
    }
    summary_json = run_dir / "frame_check_summary.json"
    summary_csv = run_dir / "frame_check_summary.csv"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_overall_csv(summary_csv, case_results)
    print(f"summary_json: {summary_json}")
    print(f"summary_csv: {summary_csv}")

    if not runtime_complete:
        print("frame_check: RUNTIME_INCOMPLETE")
        return 1
    if not gates_passed:
        print("frame_check: COMPLETED_WITH_GATE_FAILURES")
        return 2 if args.require_pass else 0
    print("frame_check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
