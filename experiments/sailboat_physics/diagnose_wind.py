#!/usr/bin/env python3
"""Validate the wind contract from Gazebo through ArduPilot and sail forces.

Each case launches a fresh, unarmed simulation, holds the boat at a known pose,
and keeps sail force disabled.  The command compares three independent views:

* the raw asv_sim anemometer ``gz.msgs.Vector3d``;
* MAVLink ``WIND`` emitted by ArduPilot's SITL wind-vane backend;
* the wind selected by ``SailLiftDragSystem::PreUpdate`` in its debug log.

Four cardinal static winds test signs and axes, a rotated-heading case separates
world-frame from sensor-frame anemometer semantics, and a sinusoidal-magnitude
case proves that the anemometer, ArduPilot and sail-force plugin all update while
the process remains running.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .diagnose_frames import resolve_path, safe_label, set_model_pose
from .frames import frd_attitude_to_gazebo_euler_deg, frd_attitude_to_gazebo_quaternion
from .wind import (
    SailPluginWindSample,
    Vector3,
    WindSample,
    summarize_wind_case,
    vector_error_mps,
)


DEFAULT_SCENARIO = "experiments/sailboat_BO/scenario.yaml"
DEFAULT_PARAMS = "experiments/sailboat_BO/baseline_params.json"
DEFAULT_RESULTS = "experiments/sailboat_physics/results/wind_checks"
WIND_MAVLINK_MESSAGE_ID = 168

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_VECTOR_FIELD = {
    axis: re.compile(rf"^\s*{axis}:\s*({_NUMBER})\s*$", re.MULTILINE)
    for axis in ("x", "y", "z")
}
_STAMP_SEC = re.compile(r"^\s*sec:\s*(-?\d+)\s*$", re.MULTILINE)
_STAMP_NSEC = re.compile(r"^\s*nsec:\s*(\d+)\s*$", re.MULTILINE)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SAIL_WIND_LINE = re.compile(
    rf"\[SailLiftDragSystem\].*?simTimeS=({_NUMBER})"
    rf".*?windSource=([^\s]+)"
    rf".*?forceScale=({_NUMBER})"
    rf".*?windWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?Vapp=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?speed=({_NUMBER})"
    rf".*?alpha_deg=({_NUMBER})"
    rf".*?cl=({_NUMBER})"
    rf".*?cd=({_NUMBER})"
)


@dataclass(frozen=True)
class WindDiagnosticCase:
    name: str
    wind_world: Vector3
    heading_deg: float
    dynamic: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gazebo wind, anemometer, MAVLink WIND, and sail-force input"
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--base-scenario-id", default="train_crosswind_straight")
    parser.add_argument("--params-file", default=DEFAULT_PARAMS)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--label", default="wind_contract")
    parser.add_argument(
        "--execution-backend",
        choices=["local"],
        default="local",
        help="Run directly in the current dave:sailboat-rl container",
    )
    parser.add_argument("--wind-speed-mps", type=float, default=8.0)
    parser.add_argument("--static-sample-duration-s", type=float, default=6.0)
    parser.add_argument("--dynamic-sample-duration-s", type=float, default=12.0)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument(
        "--wind-settle-hold-s",
        type=float,
        default=2.0,
        help=(
            "Require the sail plugin's live world wind to remain within "
            "--plugin-tolerance-mps for this many consecutive simulation seconds"
        ),
    )
    parser.add_argument(
        "--wind-settle-poll-s",
        type=float,
        default=0.2,
        help="Wall-time interval between sail-plugin log polls while settling",
    )
    parser.add_argument(
        "--wind-settle-timeout-s",
        type=float,
        default=45.0,
        help="Maximum wall time to wait for the live world wind to settle",
    )
    parser.add_argument("--pose-hold-period-s", type=float, default=0.5)
    parser.add_argument("--pose-service-timeout-ms", type=int, default=3000)
    parser.add_argument("--topic-discovery-timeout-s", type=float, default=20.0)
    parser.add_argument("--topic-read-timeout-s", type=float, default=3.0)
    parser.add_argument("--mavlink-connect-timeout-s", type=float, default=120.0)
    parser.add_argument("--sitl-start-delay-s", type=float, default=10.0)
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--minimum-plugin-samples", type=int, default=2)
    parser.add_argument("--vector-tolerance-mps", type=float, default=1.5)
    parser.add_argument("--plugin-tolerance-mps", type=float, default=0.5)
    parser.add_argument("--direction-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--speed-tolerance-mps", type=float, default=1.5)
    parser.add_argument("--heading-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--dynamic-minimum-range-mps", type=float, default=0.5)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 when runtime completes but a wind gate fails",
    )
    return parser.parse_args()


def build_cases(wind_speed_mps: float) -> list[WindDiagnosticCase]:
    speed = abs(float(wind_speed_mps))
    if speed <= 0.0:
        raise ValueError("wind speed must be positive")
    return [
        WindDiagnosticCase("north_to_heading_000", Vector3(0.0, speed, 0.0), 0.0),
        WindDiagnosticCase("east_to_heading_000", Vector3(speed, 0.0, 0.0), 0.0),
        WindDiagnosticCase("south_to_heading_000", Vector3(0.0, -speed, 0.0), 0.0),
        WindDiagnosticCase("west_to_heading_000", Vector3(-speed, 0.0, 0.0), 0.0),
        WindDiagnosticCase("north_to_heading_090", Vector3(0.0, speed, 0.0), 90.0),
        WindDiagnosticCase(
            "dynamic_north_to_heading_000",
            Vector3(0.0, speed, 0.0),
            0.0,
            dynamic=True,
        ),
    ]


def make_case_config(
    base_config: dict[str, Any],
    base_scenario: dict[str, Any],
    case: WindDiagnosticCase,
    *,
    sitl_start_delay_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base_config)
    scenario = copy.deepcopy(base_scenario)
    scenario["id"] = f"wind_{case.name}"
    scenario["split"] = "diagnostic"

    world = scenario.setdefault("world", {})
    world["wind_world_xyz_mps"] = [
        case.wind_world.x,
        case.wind_world.y,
        case.wind_world.z,
    ]
    world["wind_mag_noise_stddev"] = 0.0
    world["wind_dir_noise_stddev"] = 0.0
    world["wind_dir_sin_amp_deg"] = 0.0
    # The world template has a 60 s sinusoid.  A 50% amplitude creates an
    # observable live change during the 12 s dynamic case without reversing
    # the wind direction or making the configured mean unphysical.
    world["wind_mag_sin_amp_percent"] = 0.5 if case.dynamic else 0.0

    gazebo_roll_deg, gazebo_pitch_deg, _ = frd_attitude_to_gazebo_euler_deg(
        0.0,
        0.0,
        case.heading_deg,
    )
    spawn = scenario.setdefault("spawn", {})
    spawn["heading_deg"] = case.heading_deg
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


def parse_gz_vector3(text: str) -> tuple[float | None, Vector3]:
    values: dict[str, float] = {}
    for axis, pattern in _VECTOR_FIELD.items():
        matches = pattern.findall(text)
        if not matches:
            raise ValueError(f"gz Vector3d output is missing {axis}: {text!r}")
        values[axis] = float(matches[-1])

    sec_matches = _STAMP_SEC.findall(text)
    nsec_matches = _STAMP_NSEC.findall(text)
    sim_time_s: float | None = None
    if sec_matches:
        sim_time_s = float(sec_matches[-1])
        if nsec_matches:
            sim_time_s += float(nsec_matches[-1]) * 1e-9
    return sim_time_s, Vector3(values["x"], values["y"], values["z"])


def select_anemometer_topic(topics: Sequence[str], namespace: str) -> str:
    candidates = sorted(
        {
            topic.strip()
            for topic in topics
            if topic.strip() and "anemometer" in topic.lower()
        }
    )
    if not candidates:
        raise RuntimeError("no Gazebo anemometer topic was advertised")
    namespace_token = namespace.strip("/").lower()
    scoped = [topic for topic in candidates if namespace_token in topic.lower()]
    selected = scoped or candidates
    exact_tail = [topic for topic in selected if topic.rstrip("/").endswith("/anemometer")]
    selected = exact_tail or selected
    if len(selected) != 1:
        raise RuntimeError(
            "ambiguous Gazebo anemometer topics: " + ", ".join(selected)
        )
    return selected[0]


def discover_anemometer_topic(namespace: str, timeout_s: float) -> str:
    deadline = time.monotonic() + float(timeout_s)
    last_topics: list[str] = []
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["gz", "topic", "-l"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if completed.returncode == 0:
            last_topics = completed.stdout.splitlines()
            try:
                return select_anemometer_topic(last_topics, namespace)
            except RuntimeError:
                pass
        time.sleep(0.25)
    advertised = ", ".join(last_topics) if last_topics else "<none>"
    raise RuntimeError(
        f"anemometer topic was not discovered within {timeout_s:.1f}s; "
        f"advertised topics: {advertised}"
    )


def read_anemometer_once(topic: str, timeout_s: float) -> tuple[float | None, Vector3]:
    completed = subprocess.run(
        ["gz", "topic", "-e", "-t", topic, "-n", "1"],
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_s),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to read Gazebo anemometer topic: "
            f"topic={topic}, rc={completed.returncode}, stderr={completed.stderr.strip()}"
        )
    return parse_gz_vector3(completed.stdout)


def wait_for_wind_settle(
    log_paths: Sequence[Path],
    *,
    expected_world: Vector3,
    tolerance_mps: float,
    hold_sim_time_s: float,
    timeout_s: float,
    poll_interval_s: float,
) -> float:
    """Wait until the sail plugin's live world wind is continuously settled.

    Only ``world_component`` samples qualify: accepting the plugin's static
    ``world_seed`` fallback here would bypass the WindEffects low-pass output
    that this diagnostic is intended to validate.  The vector error is the
    same three-dimensional metric used by the
    ``sail_plugin_matches_configured_world`` result gate.
    """

    tolerance_mps = float(tolerance_mps)
    hold_sim_time_s = float(hold_sim_time_s)
    timeout_s = float(timeout_s)
    poll_interval_s = float(poll_interval_s)
    if tolerance_mps < 0.0:
        raise ValueError("wind settle tolerance must be non-negative")
    if hold_sim_time_s <= 0.0:
        raise ValueError("wind settle hold time must be positive")
    if timeout_s <= 0.0 or poll_interval_s <= 0.0:
        raise ValueError("wind settle timeout and poll interval must be positive")

    deadline = time.monotonic() + timeout_s
    stable_since_sim_time_s: float | None = None
    last_processed_sim_time_s: float | None = None
    last_sim_time_s: float | None = None
    last_source: str | None = None
    last_error_mps: float | None = None

    while time.monotonic() < deadline:
        for sample in read_sail_plugin_wind(log_paths):
            if (
                last_processed_sim_time_s is not None
                and sample.sim_time_s <= last_processed_sim_time_s
            ):
                continue

            last_processed_sim_time_s = sample.sim_time_s
            last_sim_time_s = sample.sim_time_s
            last_source = sample.source
            actual_world = Vector3(
                sample.wind_x_mps,
                sample.wind_y_mps,
                sample.wind_z_mps,
            )
            last_error_mps = vector_error_mps(actual_world, expected_world)
            qualified = (
                sample.source == "world_component"
                and last_error_mps <= tolerance_mps
            )
            if not qualified:
                stable_since_sim_time_s = None
                continue

            if stable_since_sim_time_s is None:
                stable_since_sim_time_s = sample.sim_time_s
            if sample.sim_time_s - stable_since_sim_time_s >= hold_sim_time_s:
                return sample.sim_time_s

        time.sleep(poll_interval_s)

    raise TimeoutError(
        "live world wind did not settle before the wall-time timeout: "
        f"expected={expected_world}, tolerance={tolerance_mps:.3f}m/s, "
        f"hold={hold_sim_time_s:.3f}s, last_sim_time={last_sim_time_s}, "
        f"last_source={last_source}, last_error={last_error_mps}, "
        f"wall_timeout={timeout_s:.1f}s"
    )


def request_wind_message(master: Any, hz: float = 5.0) -> None:
    mavlink = getattr(master, "mavlink", None)
    if mavlink is None:
        mavlink = getattr(getattr(master, "mav", None), "mavlink", None)
    command_id = int(getattr(mavlink, "MAV_CMD_SET_MESSAGE_INTERVAL", 511))
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        command_id,
        0,
        float(WIND_MAVLINK_MESSAGE_ID),
        float(int(1_000_000 / hz)),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _update_latest_mavlink(
    master: Any,
    latest_wind: tuple[float, float, float] | None,
    latest_heading_deg: float | None,
) -> tuple[tuple[float, float, float] | None, float | None]:
    while True:
        message = master.recv_match(type=["ATTITUDE", "WIND"], blocking=False)
        if message is None:
            return latest_wind, latest_heading_deg
        message_type = message.get_type()
        if message_type == "WIND":
            latest_wind = (
                float(message.direction),
                float(message.speed),
                float(getattr(message, "speed_z", 0.0)),
            )
        elif message_type == "ATTITUDE":
            latest_heading_deg = math.degrees(float(message.yaw)) % 360.0


def collect_wind_samples(
    *,
    master: Any,
    topic: str,
    case: WindDiagnosticCase,
    world_name: str,
    model_name: str,
    position_xyz_m: tuple[float, float, float],
    sample_duration_s: float,
    sample_hz: float,
    pose_hold_period_s: float,
    pose_service_timeout_ms: int,
    topic_read_timeout_s: float,
) -> list[WindSample]:
    if sample_duration_s <= 0.0 or sample_hz <= 0.0 or pose_hold_period_s <= 0.0:
        raise ValueError("sample duration, sample rate, and pose hold period must be positive")

    target_quaternion = frd_attitude_to_gazebo_quaternion(0.0, 0.0, case.heading_deg)
    request_wind_message(master)
    start = time.monotonic()
    deadline = start + sample_duration_s
    next_sample = start
    next_pose_hold = start
    latest_wind: tuple[float, float, float] | None = None
    latest_heading_deg: float | None = None
    samples: list[WindSample] = []

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

        latest_wind, latest_heading_deg = _update_latest_mavlink(
            master,
            latest_wind,
            latest_heading_deg,
        )
        now = time.monotonic()
        if now < next_sample:
            time.sleep(min(0.02, next_sample - now))
            continue
        next_sample += 1.0 / sample_hz

        sim_time_s, vector = read_anemometer_once(topic, topic_read_timeout_s)
        latest_wind, latest_heading_deg = _update_latest_mavlink(
            master,
            latest_wind,
            latest_heading_deg,
        )
        samples.append(
            WindSample(
                elapsed_wall_s=time.monotonic() - start,
                sim_time_s=sim_time_s,
                anemometer_x_mps=vector.x,
                anemometer_y_mps=vector.y,
                anemometer_z_mps=vector.z,
                mav_wind_direction_deg=latest_wind[0] if latest_wind else None,
                mav_wind_speed_mps=latest_wind[1] if latest_wind else None,
                mav_wind_speed_z_mps=latest_wind[2] if latest_wind else None,
                mav_heading_deg=latest_heading_deg,
            )
        )
    return samples


def parse_sail_plugin_wind(text: str) -> list[SailPluginWindSample]:
    # ros2 launch prefixes and colorizes Gazebo output.  Strip only terminal
    # control sequences so the same parser accepts raw and captured logs.
    text = _ANSI_ESCAPE.sub("", text)
    records = [
        SailPluginWindSample(
            sim_time_s=float(match.group(1)),
            source=match.group(2),
            force_scale=float(match.group(3)),
            wind_x_mps=float(match.group(4)),
            wind_y_mps=float(match.group(5)),
            wind_z_mps=float(match.group(6)),
            apparent_x_mps=float(match.group(7)),
            apparent_y_mps=float(match.group(8)),
            apparent_z_mps=float(match.group(9)),
            aerodynamic_speed_mps=float(match.group(10)),
            alpha_deg=float(match.group(11)),
            lift_coefficient=float(match.group(12)),
            drag_coefficient=float(match.group(13)),
        )
        for match in _SAIL_WIND_LINE.finditer(text)
    ]
    # Some launch configurations mirror console output into both files.  Keep
    # one copy so minimum-sample and range gates are not artificially inflated.
    unique: dict[tuple[object, ...], SailPluginWindSample] = {}
    for record in records:
        key = (
            record.sim_time_s,
            record.source,
            record.force_scale,
            record.wind_x_mps,
            record.wind_y_mps,
            record.wind_z_mps,
            record.apparent_x_mps,
            record.apparent_y_mps,
            record.apparent_z_mps,
            record.aerodynamic_speed_mps,
            record.alpha_deg,
            record.lift_coefficient,
            record.drag_coefficient,
        )
        unique[key] = record
    return sorted(unique.values(), key=lambda record: record.sim_time_s)


def read_sail_plugin_wind(log_paths: Sequence[Path]) -> list[SailPluginWindSample]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in log_paths
        if path.exists()
    )
    return parse_sail_plugin_wind(text)


def write_dataclass_csv(path: Path, records: Sequence[object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_overall_csv(path: Path, case_results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "status",
        "dynamic",
        "passed",
        "sample_count",
        "mavlink_sample_count",
        "sail_plugin_sample_count",
        "anemometer_frame_contract",
        "sail_plugin_sources",
        "anemometer_vs_world_mps",
        "anemometer_vs_sensor_mps",
        "sail_plugin_vs_world_mps",
        "sail_plugin_direction_error_deg",
        "mav_direction_error_deg",
        "mav_speed_error_mps",
        "anemometer_speed_range_mps",
        "sail_plugin_speed_range_mps",
        "mav_speed_range_mps",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in case_results:
            summary = result.get("summary") or {}
            error = summary.get("error") or {}
            value_range = summary.get("range") or {}
            writer.writerow(
                {
                    "case": result["case"],
                    "status": result["status"],
                    "dynamic": result.get("dynamic", False),
                    "passed": summary.get("passed", False),
                    "sample_count": summary.get("sample_count", 0),
                    "mavlink_sample_count": summary.get("mavlink_sample_count", 0),
                    "sail_plugin_sample_count": summary.get("sail_plugin_sample_count", 0),
                    "anemometer_frame_contract": summary.get("anemometer_frame_contract"),
                    "sail_plugin_sources": ";".join(summary.get("sail_plugin_sources", [])),
                    "anemometer_vs_world_mps": error.get("anemometer_vs_world_mps"),
                    "anemometer_vs_sensor_mps": error.get("anemometer_vs_sensor_mps"),
                    "sail_plugin_vs_world_mps": error.get("sail_plugin_vs_world_mps"),
                    "sail_plugin_direction_error_deg": error.get(
                        "sail_plugin_direction_deg"
                    ),
                    "mav_direction_error_deg": error.get("mav_direction_deg"),
                    "mav_speed_error_mps": error.get("mav_speed_mps"),
                    "anemometer_speed_range_mps": value_range.get(
                        "anemometer_horizontal_speed_mps"
                    ),
                    "sail_plugin_speed_range_mps": value_range.get(
                        "sail_plugin_horizontal_speed_mps"
                    ),
                    "mav_speed_range_mps": value_range.get("mav_wind_speed_mps"),
                    "error": result.get("error", ""),
                }
            )


def main() -> int:
    args = parse_args()
    # Runtime-only imports keep pure wind tests independent of ROS, Gazebo and
    # pymavlink availability.
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
    missing_commands = [
        name for name in ("gz", "ros2", "ardurover") if shutil.which(name) is None
    ]
    if missing_commands:
        raise SystemExit(
            "Wind diagnostics must run inside the simulation container; "
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
            "Stop the existing BO/Gazebo/SITL process before running wind diagnostics."
        )

    cases = build_cases(args.wind_speed_mps)
    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        print(f"[wind_check] case {index + 1}/{len(cases)}: {case.name}")
        case_config, scenario = make_case_config(
            config,
            base_scenario,
            case,
            sitl_start_delay_s=args.sitl_start_delay_s,
        )
        context = build_context(
            case_config,
            scenario,
            params,
            repo_root,
            workspace_root,
            run_dir / "cases",
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
        samples: list[WindSample] = []
        plugin_samples: list[SailPluginWindSample] = []
        endpoint_closed = True
        runtime_error: Exception | None = None
        result: dict[str, Any] = {
            "case": case.name,
            "dynamic": case.dynamic,
            "wind_world": asdict(case.wind_world),
            "heading_deg": case.heading_deg,
            "status": "runtime_error",
            "trial_id": context.trial_id,
            "trial_dir": str(context.files.trial_dir),
        }
        try:
            process = start_launch_process(context, launch_plan)
            publish_sail_force_enabled(context, execution, False, reason="wind_diagnostic")
            master = connect_mavlink(
                endpoint,
                timeout_s=float(args.mavlink_connect_timeout_s),
                launch_process=process,
                stdout_log=context.files.logs_dir / "launch.stdout.log",
                stderr_log=context.files.logs_dir / "launch.stderr.log",
            )
            namespace = str(context.study_cfg["namespace"])
            topic = discover_anemometer_topic(
                namespace,
                float(args.topic_discovery_timeout_s),
            )
            result["anemometer_topic"] = topic
            print(f"[wind_check] anemometer topic: {topic}")
            plugin_log_paths = [
                context.files.logs_dir / "launch.stdout.log",
                context.files.logs_dir / "launch.stderr.log",
            ]
            sampling_started_at_sim_time_s = wait_for_wind_settle(
                plugin_log_paths,
                expected_world=case.wind_world,
                tolerance_mps=float(args.plugin_tolerance_mps),
                hold_sim_time_s=float(args.wind_settle_hold_s),
                timeout_s=float(args.wind_settle_timeout_s),
                poll_interval_s=float(args.wind_settle_poll_s),
            )
            result["sampling_started_at_sim_time_s"] = sampling_started_at_sim_time_s
            print(
                "[wind_check] wind settled: "
                f"sampling_start_sim_time_s={sampling_started_at_sim_time_s:.3f}"
            )
            spawn = scenario["spawn"]
            duration_s = (
                float(args.dynamic_sample_duration_s)
                if case.dynamic
                else float(args.static_sample_duration_s)
            )
            samples = collect_wind_samples(
                master=master,
                topic=topic,
                case=case,
                world_name=str(scenario["world"].get("internal_world_name", "waves")),
                model_name=namespace,
                position_xyz_m=(
                    float(spawn["x_m"]),
                    float(spawn["y_m"]),
                    float(spawn["z_m"]),
                ),
                sample_duration_s=duration_s,
                sample_hz=float(args.sample_hz),
                pose_hold_period_s=float(args.pose_hold_period_s),
                pose_service_timeout_ms=int(args.pose_service_timeout_ms),
                topic_read_timeout_s=float(args.topic_read_timeout_s),
            )
        except Exception as exc:
            runtime_error = exc
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[wind_check] {case.name}: ERROR: {result['error']}")
        finally:
            write_dataclass_csv(
                context.files.raw_dir / "wind_samples.csv",
                samples,
                list(WindSample.__dataclass_fields__),
            )
            if master is not None:
                try:
                    master.close()
                except Exception:
                    pass
            stop_launch_process(process, execution, launch_plan)
            endpoint_closed = wait_for_endpoint_closed(endpoint, 15.0)

        plugin_samples = read_sail_plugin_wind(
            [
                context.files.logs_dir / "launch.stdout.log",
                context.files.logs_dir / "launch.stderr.log",
            ]
        )
        write_dataclass_csv(
            context.files.raw_dir / "sail_plugin_wind.csv",
            plugin_samples,
            list(SailPluginWindSample.__dataclass_fields__),
        )

        if not endpoint_closed:
            result["status"] = "cleanup_error"
            result["error"] = f"MAVLink endpoint remained open after cleanup: {endpoint}"
        elif runtime_error is None:
            summary = summarize_wind_case(
                samples,
                plugin_samples,
                expected_world=case.wind_world,
                expected_heading_deg=case.heading_deg,
                dynamic=case.dynamic,
                minimum_samples=int(args.minimum_samples),
                minimum_plugin_samples=int(args.minimum_plugin_samples),
                vector_tolerance_mps=float(args.vector_tolerance_mps),
                plugin_tolerance_mps=float(args.plugin_tolerance_mps),
                direction_tolerance_deg=float(args.direction_tolerance_deg),
                speed_tolerance_mps=float(args.speed_tolerance_mps),
                heading_tolerance_deg=float(args.heading_tolerance_deg),
                dynamic_minimum_range_mps=float(args.dynamic_minimum_range_mps),
            )
            result["status"] = "complete"
            result["summary"] = summary
            print(
                f"[wind_check] {case.name}: passed={summary['passed']}, "
                f"anemometer_contract={summary['anemometer_frame_contract']}, "
                f"plugin_sources={summary['sail_plugin_sources']}, "
                f"gates={summary['gates']}"
            )
        case_results.append(result)
        if not endpoint_closed:
            print(f"[wind_check] aborting remaining cases: {result['error']}")
            break

    runtime_complete = len(case_results) == len(cases) and all(
        result["status"] == "complete" for result in case_results
    )
    resolved_contracts = {
        result.get("summary", {}).get("anemometer_frame_contract")
        for result in case_results
        if result.get("summary", {}).get("anemometer_frame_contract")
        not in {None, "unavailable", "ambiguous_aligned"}
    }
    contract_consistent = len(resolved_contracts) == 1
    gates_passed = (
        runtime_complete
        and contract_consistent
        and all(
            bool(result.get("summary", {}).get("passed", False))
            for result in case_results
        )
    )
    payload = {
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source_scenario": str(resolve_path(repo_root, args.scenario)),
        "base_scenario_id": args.base_scenario_id,
        "params_file": str(resolve_path(repo_root, args.params_file)),
        "sampling": {
            "wind_settle_hold_s": args.wind_settle_hold_s,
            "wind_settle_poll_s": args.wind_settle_poll_s,
            "wind_settle_tolerance_mps": args.plugin_tolerance_mps,
            "wind_settle_timeout_s": args.wind_settle_timeout_s,
            "static_sample_duration_s": args.static_sample_duration_s,
            "dynamic_sample_duration_s": args.dynamic_sample_duration_s,
            "sample_hz": args.sample_hz,
        },
        "runtime_complete": runtime_complete,
        "all_wind_gates_passed": gates_passed,
        "anemometer_frame_contract_consistent": contract_consistent,
        "resolved_anemometer_frame_contracts": sorted(resolved_contracts),
        "tolerances": {
            "vector_mps": args.vector_tolerance_mps,
            "plugin_mps": args.plugin_tolerance_mps,
            "direction_deg": args.direction_tolerance_deg,
            "speed_mps": args.speed_tolerance_mps,
            "heading_deg": args.heading_tolerance_deg,
            "dynamic_minimum_range_mps": args.dynamic_minimum_range_mps,
            "minimum_samples": args.minimum_samples,
            "minimum_plugin_samples": args.minimum_plugin_samples,
        },
        "cases": case_results,
    }
    summary_json = run_dir / "wind_check_summary.json"
    summary_csv = run_dir / "wind_check_summary.csv"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_overall_csv(summary_csv, case_results)
    print(f"summary_json: {summary_json}")
    print(f"summary_csv: {summary_csv}")

    if not runtime_complete:
        print("wind_check: RUNTIME_INCOMPLETE")
        return 1
    if not gates_passed:
        print("wind_check: COMPLETED_WITH_GATE_FAILURES")
        return 2 if args.require_pass else 0
    print("wind_check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
