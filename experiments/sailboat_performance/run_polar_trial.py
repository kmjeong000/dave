"""Run one fixed-heading, wind-only sailing-performance polar cell.

The runner reuses the validated BO launch / startup-hold / sail-force lifecycle,
but deliberately does not upload a mission or enter AUTO.  It arms in GUIDED,
refreshes one velocity-and-yaw target, and measures a steady-state window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .core import (
    Vector2,
    angle_error_deg,
    bearing_unit_enu,
    circular_mean_deg,
    circular_std_deg,
    scalar_statistics,
    signed_twa_deg,
    vmg_toward_bearing_mps,
    wind_world_enu_from_condition,
)
from .guided_heading import (
    GuidedHeadingTarget,
    parse_ardupilot_parameter_file,
    validate_wind_only_sdf,
    validate_wind_only_servo_functions,
)
from .steady_state import SteadyObservation, SteadyStateConfig, SteadyStateDetector
from .telemetry import (
    MavlinkPerformanceState,
    PerformanceRosCollector,
    RosPerformanceSnapshot,
    drain_mavlink,
    telemetry_ages_are_fresh,
)


DEFAULT_CONFIG = Path("experiments/sailboat_performance/config/polar_smoke.yaml")
DEFAULT_PARAMS = Path("experiments/sailboat_BO/optimized_baseline_params.json")
DEFAULT_RESULTS = Path("experiments/sailboat_performance/results")


RAW_FIELDS = [
    "trial_id",
    "phase",
    "sample_time_utc",
    "sim_time_s",
    "elapsed_since_release_s",
    "x_east_m",
    "y_north_m",
    "z_up_m",
    "roll_frd_deg",
    "pitch_frd_deg",
    "heading_deg",
    "heading_error_deg",
    "derived_yaw_rate_deg_s",
    "gazebo_angular_yaw_rate_deg_s",
    "mavlink_yaw_deg",
    "mavlink_yaw_rate_deg_s",
    "mavlink_gazebo_heading_error_deg",
    "mavlink_gazebo_yaw_rate_error_deg_s",
    "commanded_heading_deg",
    "commanded_velocity_north_mps",
    "commanded_velocity_east_mps",
    "model_vx_mps",
    "model_vy_mps",
    "model_vz_mps",
    "velocity_east_mps",
    "velocity_north_mps",
    "velocity_up_mps",
    "speed_over_ground_mps",
    "course_over_ground_deg",
    "leeway_deg",
    "target_tws_mps",
    "target_signed_twa_deg",
    "commanded_wind_east_mps",
    "commanded_wind_north_mps",
    "measured_wind_from_deg",
    "measured_wind_speed_mps",
    "measured_wind_speed_z_mps",
    "measured_wind_to_east_mps",
    "measured_wind_to_north_mps",
    "measured_signed_twa_deg",
    "vmg_upwind_mps",
    "vmg_downwind_mps",
    "base_rudder_rad",
    "final_rudder_rad",
    "actual_rudder_angle_rad",
    "actual_rudder_velocity_rad_s",
    "rudder_tracking_error_rad",
    "base_sheet_allowance_rad",
    "final_sheet_allowance_rad",
    "actual_boom_angle_rad",
    "actual_boom_velocity_rad_s",
    "odom_frame_id",
    "odom_child_frame_id",
    "odom_age_s",
    "rudder_age_s",
    "boom_age_s",
    "actuator_age_s",
    "wind_age_s",
    "mavlink_attitude_age_s",
    "mavlink_status_text",
    "guided_mode_active",
    "wind_settled",
    "force_ramp_complete",
    "tack_detected",
    "telemetry_valid",
    "steady_qualified",
    "steady_ready",
    "steady_reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one fixed-heading wind-only sailboat polar cell"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--params-file", default=str(DEFAULT_PARAMS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--repeat-idx", type=int, default=0)
    parser.add_argument(
        "--execution-backend", choices=["local", "docker-exec"], default=None
    )
    parser.add_argument("--docker-container")
    parser.add_argument("--container-repo-root")
    return parser.parse_args()


def resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path).resolve()


def validate_performance_config(config: dict[str, Any]) -> dict[str, Any]:
    required_sections = {"study", "condition", "world", "spawn", "performance"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError("missing performance config sections: " + ", ".join(missing))
    condition = config["condition"]
    performance = config["performance"]
    if float(condition["tws_mps"]) <= 0.0:
        raise ValueError("condition.tws_mps must be positive")
    if not -180.0 <= float(condition["signed_twa_deg"]) <= 180.0:
        raise ValueError("condition.signed_twa_deg must be within [-180, 180]")
    for name in (
        "desired_speed_mps",
        "target_refresh_hz",
        "sample_hz",
        "max_sim_duration_s",
        "measurement_duration_s",
    ):
        if float(performance[name]) <= 0.0:
            raise ValueError(f"performance.{name} must be positive")
    if int(performance["minimum_measurement_samples"]) < 2:
        raise ValueError("minimum_measurement_samples must be at least 2")
    expected_samples = (
        float(performance["sample_hz"])
        * float(performance["measurement_duration_s"])
    )
    if int(performance["minimum_measurement_samples"]) > math.floor(expected_samples) + 1:
        raise ValueError(
            "minimum_measurement_samples exceeds the configured source-rate budget"
        )
    for name in ("target_heading_deg", "signed_twa_deg", "tws_mps"):
        if not math.isfinite(float(condition[name])):
            raise ValueError(f"condition.{name} must be finite")
    SteadyStateConfig.from_mapping(performance.get("steady_state"))
    return config


def make_scenario(config: dict[str, Any], wind: Vector2) -> dict[str, Any]:
    world = dict(config["world"])
    world["wind_world_xyz_mps"] = [wind.east, wind.north, 0.0]
    return {
        "id": "single_polar_cell",
        "split": "performance",
        "world": world,
        "spawn": dict(config["spawn"]),
    }


def make_bo_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    """Supply the minimum keys required by the validated BO lifecycle helpers."""

    return {
        "study": dict(config["study"]),
        "search_space": [],
        "termination": {},
        "logging": {},
    }


def execute_shell(execution: Any, shell: str) -> subprocess.CompletedProcess[str]:
    if execution.backend == "docker-exec":
        if not execution.docker_container:
            raise RuntimeError("docker-exec requires docker_container")
        command = ["docker", "exec", "-i", execution.docker_container, "bash", "-lc", shell]
    else:
        command = ["bash", "-lc", shell]
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=10.0)


def ensure_residual_disabled(execution: Any, node_name: str) -> None:
    quoted = shlex.quote(node_name)
    set_result = execute_shell(
        execution, f"ros2 param set {quoted} residual_enabled false"
    )
    if set_result.returncode != 0:
        raise RuntimeError(
            "failed to disable residual control: "
            + (set_result.stderr or set_result.stdout).strip()
        )
    get_result = execute_shell(
        execution, f"ros2 param get {quoted} residual_enabled"
    )
    combined = f"{get_result.stdout}\n{get_result.stderr}".lower()
    if get_result.returncode != 0 or "false" not in combined:
        raise RuntimeError("residual_enabled could not be verified false")
    print("[polar_trial] residual RL: disabled and verified")


def write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def guided_target_record_fields(
    target: GuidedHeadingTarget,
    measured_heading_deg: float | None,
) -> dict[str, float | None]:
    """Return the transmitted GUIDED target and its measured heading error."""

    return {
        "commanded_heading_deg": float(target.heading_deg),
        "commanded_velocity_north_mps": float(target.velocity_north_mps),
        "commanded_velocity_east_mps": float(target.velocity_east_mps),
        "heading_error_deg": (
            angle_error_deg(target.heading_deg, measured_heading_deg)
            if measured_heading_deg is not None
            else None
        ),
    }


def diagnostic_comparison_fields(
    ros: RosPerformanceSnapshot,
    mav_state: MavlinkPerformanceState,
) -> dict[str, float | None]:
    """Return independent attitude and rudder comparison diagnostics."""

    return {
        # Joint-state position and command use the same signed SDF joint axis.
        "rudder_tracking_error_rad": (
            float(ros.actual_rudder_angle_rad) - float(ros.final_rudder_rad)
            if ros.actual_rudder_angle_rad is not None
            and ros.final_rudder_rad is not None
            else None
        ),
        # Both headings are compass/NED bearings; retain a wrapped MAV-Gazebo
        # signed difference across the 0/360 degree boundary.
        "mavlink_gazebo_heading_error_deg": (
            angle_error_deg(mav_state.yaw_deg, ros.heading_deg)
            if mav_state.yaw_deg is not None and ros.heading_deg is not None
            else None
        ),
        # Both rates are body-FRD, clockwise-positive degrees per second.
        "mavlink_gazebo_yaw_rate_error_deg_s": (
            float(mav_state.yaw_rate_deg_s)
            - float(ros.gazebo_angular_yaw_rate_deg_s)
            if mav_state.yaw_rate_deg_s is not None
            and ros.gazebo_angular_yaw_rate_deg_s is not None
            else None
        ),
    }


def _sign_change_count(values: Iterable[float]) -> int:
    previous_sign = 0
    changes = 0
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric) or math.isclose(numeric, 0.0, abs_tol=1e-12):
            continue
        sign = 1 if numeric > 0.0 else -1
        if previous_sign and sign != previous_sign:
            changes += 1
        previous_sign = sign
    return changes


def summarize_diagnostic_window(
    records: list[dict[str, Any]],
    *,
    window_s: float,
    boom_limit_rad: float,
) -> dict[str, Any]:
    """Summarize the final steady-state-sized window for failed smoke runs."""

    timed = [
        record
        for record in records
        if record.get("sim_time_s") is not None
        and math.isfinite(float(record["sim_time_s"]))
    ]
    if not timed:
        return {"sample_count": 0, "window_duration_s": 0.0}
    end_s = max(float(record["sim_time_s"]) for record in timed)
    window = [
        record
        for record in timed
        if float(record["sim_time_s"]) >= end_s - float(window_s)
    ]
    start_s = min(float(record["sim_time_s"]) for record in window)

    def values(field: str) -> list[float]:
        return [
            float(record[field])
            for record in window
            if record.get(field) is not None
            and math.isfinite(float(record[field]))
        ]

    heading_errors = values("heading_error_deg")
    derived_yaw_rates = values("derived_yaw_rate_deg_s")
    gazebo_yaw_rates = values("gazebo_angular_yaw_rate_deg_s")
    mavlink_yaw_rates = values("mavlink_yaw_rate_deg_s")
    rudder = values("final_rudder_rad")
    actual_rudder = values("actual_rudder_angle_rad")
    actual_rudder_velocity = values("actual_rudder_velocity_rad_s")
    rudder_tracking_error = values("rudder_tracking_error_rad")
    boom = values("actual_boom_angle_rad")
    boom_velocity = values("actual_boom_velocity_rad_s")
    sheet_allowance = values("final_sheet_allowance_rad")
    heading_stats = scalar_statistics(heading_errors)
    heading_stats["rmse"] = (
        math.sqrt(fmean(value * value for value in heading_errors))
        if heading_errors
        else None
    )
    derived_yaw_rate_stats = scalar_statistics(derived_yaw_rates)
    derived_yaw_rate_stats["max_abs"] = max(
        (abs(value) for value in derived_yaw_rates), default=None
    )
    gazebo_yaw_rate_stats = scalar_statistics(gazebo_yaw_rates)
    gazebo_yaw_rate_stats["max_abs"] = max(
        (abs(value) for value in gazebo_yaw_rates), default=None
    )
    mavlink_yaw_rate_stats = scalar_statistics(mavlink_yaw_rates)
    mavlink_yaw_rate_stats["max_abs"] = max(
        (abs(value) for value in mavlink_yaw_rates), default=None
    )
    rudder_stats = scalar_statistics(rudder)
    rudder_stats["sign_change_count"] = _sign_change_count(rudder)
    actual_rudder_stats = scalar_statistics(actual_rudder)
    actual_rudder_velocity_stats = scalar_statistics(actual_rudder_velocity)
    actual_rudder_velocity_stats["max_abs"] = max(
        (abs(value) for value in actual_rudder_velocity), default=None
    )
    rudder_tracking_stats = scalar_statistics(rudder_tracking_error)
    rudder_tracking_stats["max_abs"] = max(
        (abs(value) for value in rudder_tracking_error), default=None
    )
    heading_source_error = values("mavlink_gazebo_heading_error_deg")
    heading_source_error_stats = scalar_statistics(heading_source_error)
    heading_source_error_stats["max_abs"] = max(
        (abs(value) for value in heading_source_error), default=None
    )
    yaw_rate_source_error = values("mavlink_gazebo_yaw_rate_error_deg_s")
    yaw_rate_source_error_stats = scalar_statistics(yaw_rate_source_error)
    yaw_rate_source_error_stats["max_abs"] = max(
        (abs(value) for value in yaw_rate_source_error), default=None
    )
    boom_stats = scalar_statistics(boom)
    limit = abs(float(boom_limit_rad))
    near_limit_threshold = 0.99 * limit
    allowance_tolerance = math.radians(0.5)
    boom_allowance_pairs = [
        (
            float(record["actual_boom_angle_rad"]),
            float(record["final_sheet_allowance_rad"]),
        )
        for record in window
        if record.get("actual_boom_angle_rad") is not None
        and record.get("final_sheet_allowance_rad") is not None
        and math.isfinite(float(record["actual_boom_angle_rad"]))
        and math.isfinite(float(record["final_sheet_allowance_rad"]))
    ]
    boom_stats.update(
        {
            "hard_limit_rad": limit,
            "near_limit_threshold_rad": near_limit_threshold,
            "near_limit_fraction": (
                sum(abs(value) >= near_limit_threshold for value in boom) / len(boom)
                if boom and limit > 0.0
                else None
            ),
            "outside_sheet_allowance_tolerance_rad": allowance_tolerance,
            "outside_sheet_allowance_fraction": (
                sum(
                    abs(actual) > abs(allowance) + allowance_tolerance
                    for actual, allowance in boom_allowance_pairs
                )
                / len(boom_allowance_pairs)
                if boom_allowance_pairs
                else None
            ),
        }
    )
    boom_velocity_stats = scalar_statistics(boom_velocity)
    boom_velocity_stats["max_abs"] = max(
        (abs(value) for value in boom_velocity), default=None
    )
    return {
        "sample_count": len(window),
        "window_duration_s": end_s - start_s,
        "commanded_heading_deg": circular_statistics(
            values("commanded_heading_deg")
        ),
        "commanded_velocity_north_mps": scalar_statistics(
            values("commanded_velocity_north_mps")
        ),
        "commanded_velocity_east_mps": scalar_statistics(
            values("commanded_velocity_east_mps")
        ),
        "heading_deg": circular_statistics(values("heading_deg")),
        "heading_error_deg": heading_stats,
        "derived_yaw_rate_deg_s": derived_yaw_rate_stats,
        "gazebo_angular_yaw_rate_deg_s": gazebo_yaw_rate_stats,
        "mavlink_yaw_deg": circular_statistics(values("mavlink_yaw_deg")),
        "mavlink_yaw_rate_deg_s": mavlink_yaw_rate_stats,
        "mavlink_gazebo_heading_error_deg": heading_source_error_stats,
        "mavlink_gazebo_yaw_rate_error_deg_s": yaw_rate_source_error_stats,
        "final_rudder_rad": rudder_stats,
        "actual_rudder_angle_rad": actual_rudder_stats,
        "actual_rudder_velocity_rad_s": actual_rudder_velocity_stats,
        "rudder_tracking_error_rad": rudder_tracking_stats,
        "final_sheet_allowance_rad": scalar_statistics(sheet_allowance),
        "actual_boom_angle_rad": boom_stats,
        "actual_boom_velocity_rad_s": boom_velocity_stats,
        "measured_signed_twa_deg": scalar_statistics(
            values("measured_signed_twa_deg")
        ),
        "roll_frd_deg": scalar_statistics(values("roll_frd_deg")),
        "speed_over_ground_mps": scalar_statistics(values("speed_over_ground_mps")),
    }


def circular_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": circular_mean_deg(values),
        "std": circular_std_deg(values),
    }


def summarize(
    records: list[dict[str, Any]],
    measurement: list[dict[str, Any]],
    *,
    context: Any,
    config_path: Path,
    params_path: Path,
    config: dict[str, Any],
    commanded_wind: Vector2,
    outcome: dict[str, Any],
    git_commit: str | None,
    boom_limit_rad: float,
) -> dict[str, Any]:
    valid = [record for record in measurement if record["telemetry_valid"]]
    raw_valid = [record for record in records if record["telemetry_valid"]]
    measurement_fields = (
        "commanded_velocity_north_mps",
        "commanded_velocity_east_mps",
        "speed_over_ground_mps",
        "velocity_east_mps",
        "velocity_north_mps",
        "vmg_upwind_mps",
        "vmg_downwind_mps",
        "heading_error_deg",
        "derived_yaw_rate_deg_s",
        "gazebo_angular_yaw_rate_deg_s",
        "mavlink_yaw_rate_deg_s",
        "mavlink_gazebo_heading_error_deg",
        "mavlink_gazebo_yaw_rate_error_deg_s",
        "roll_frd_deg",
        "leeway_deg",
        "measured_wind_speed_mps",
        "measured_wind_to_east_mps",
        "measured_wind_to_north_mps",
        "measured_signed_twa_deg",
        "base_rudder_rad",
        "final_rudder_rad",
        "actual_rudder_angle_rad",
        "actual_rudder_velocity_rad_s",
        "rudder_tracking_error_rad",
        "base_sheet_allowance_rad",
        "final_sheet_allowance_rad",
        "actual_boom_angle_rad",
        "actual_boom_velocity_rad_s",
    )
    stats = {
        field: scalar_statistics(
            record[field] for record in valid if record.get(field) is not None
        )
        for field in measurement_fields
    }
    for field in (
        "derived_yaw_rate_deg_s",
        "gazebo_angular_yaw_rate_deg_s",
        "mavlink_yaw_rate_deg_s",
        "mavlink_gazebo_heading_error_deg",
        "mavlink_gazebo_yaw_rate_error_deg_s",
        "actual_rudder_velocity_rad_s",
        "rudder_tracking_error_rad",
        "actual_boom_velocity_rad_s",
    ):
        field_values = [
            float(record[field])
            for record in valid
            if record.get(field) is not None
            and math.isfinite(float(record[field]))
        ]
        stats[field]["max_abs"] = max(
            (abs(value) for value in field_values), default=None
        )
    stats["heading_deg"] = circular_statistics(
        [float(record["heading_deg"]) for record in valid if record.get("heading_deg") is not None]
    )
    stats["commanded_heading_deg"] = circular_statistics(
        [
            float(record["commanded_heading_deg"])
            for record in valid
            if record.get("commanded_heading_deg") is not None
        ]
    )
    stats["mavlink_yaw_deg"] = circular_statistics(
        [
            float(record["mavlink_yaw_deg"])
            for record in valid
            if record.get("mavlink_yaw_deg") is not None
        ]
    )
    heading_errors = [
        float(record["heading_error_deg"])
        for record in valid
        if record.get("heading_error_deg") is not None
    ]
    stats["heading_error_deg"]["rmse"] = (
        math.sqrt(fmean(value * value for value in heading_errors))
        if heading_errors
        else None
    )
    return {
        "metadata": {
            "trial_id": context.trial_id,
            "started_at_utc": context.started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "config_path": str(config_path),
            "params_file": str(params_path),
            "params": context.params,
        },
        "condition": {
            **dict(config["condition"]),
            "commanded_wind_world_enu_mps": {
                "east": commanded_wind.east,
                "north": commanded_wind.north,
                "up": 0.0,
            },
        },
        "outcome": outcome,
        "quality": {
            "raw_sample_count": len(records),
            "raw_telemetry_valid_sample_count": len(raw_valid),
            "raw_telemetry_valid_ratio": len(raw_valid) / len(records) if records else 0.0,
            "raw_guided_mode_active_ratio": fmean(
                1.0 if record["guided_mode_active"] else 0.0 for record in records
            ) if records else 0.0,
            "measurement_sample_count": len(measurement),
            "valid_measurement_sample_count": len(valid),
            "valid_measurement_ratio": len(valid) / len(measurement) if measurement else 0.0,
            "guided_mode_active_ratio": fmean(
                1.0 if record["guided_mode_active"] else 0.0 for record in measurement
            ) if measurement else 0.0,
            "tack_detected": any(record["tack_detected"] for record in records),
            "odom_frames": sorted(
                {f"{record['odom_frame_id']}->{record['odom_child_frame_id']}" for record in records}
            ),
        },
        "diagnostics": {
            "last_steady_window": summarize_diagnostic_window(
                records,
                window_s=SteadyStateConfig.from_mapping(
                    config["performance"].get("steady_state")
                ).window_s,
                boom_limit_rad=boom_limit_rad,
            )
        },
        "measurements": stats,
    }


def git_head(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def arm_vehicle(master: Any, context: Any, attempt_arm_until_armed: Any) -> None:
    study = context.study_cfg
    try:
        attempt_arm_until_armed(
            master,
            total_timeout_s=float(study.get("arm_total_timeout_s", 120.0)),
            retry_interval_s=float(study.get("arm_retry_interval_s", 15.0)),
            initial_delay_s=float(study.get("arm_initial_delay_s", 30.0)),
            force=False,
        )
    except TimeoutError:
        if not bool(study.get("force_arm_fallback", False)):
            raise
        attempt_arm_until_armed(
            master,
            total_timeout_s=float(study.get("force_arm_timeout_s", 30.0)),
            retry_interval_s=float(study.get("arm_retry_interval_s", 15.0)),
            initial_delay_s=0.0,
            force=True,
        )


def main() -> int:
    args = parse_args()

    # Runtime-only imports keep the pure calculations and unit tests independent
    # of ROS, Gazebo and pymavlink availability.
    from experiments.sailboat_BO.common import load_yaml
    from experiments.sailboat_BO.run_trial import (
        ExecutionConfig,
        attempt_arm_until_armed,
        build_context,
        build_launch_plan,
        connect_mavlink,
        get_mavlink_module,
        get_repo_root,
        get_workspace_root,
        is_tcp_endpoint_open,
        load_param_overrides_from_file,
        publish_sail_force_enabled,
        publish_startup_state_held,
        render_trial_world,
        request_message_intervals,
        set_vehicle_mode,
        start_launch_process,
        stop_launch_process,
        validate_execution_backend,
        wait_for_endpoint_closed,
        write_trial_manifest,
        write_trial_param_file,
    )
    from experiments.sailboat_physics.diagnose_wind import wait_for_wind_settle
    from experiments.sailboat_physics.wind import Vector3

    repo_root = get_repo_root()
    workspace_root = get_workspace_root(repo_root)
    config_path = resolve_path(repo_root, args.config)
    params_path = resolve_path(repo_root, args.params_file)
    config = validate_performance_config(load_yaml(config_path))
    results_dir = resolve_path(repo_root, args.results_dir)
    performance = config["performance"]
    condition = config["condition"]

    missing_commands = [name for name in ("gz", "ros2", "ardurover") if shutil.which(name) is None]
    if missing_commands:
        raise SystemExit(
            "polar trial must run inside the simulation container; missing: "
            + ", ".join(missing_commands)
        )

    execution = ExecutionConfig(
        backend=args.execution_backend or str(config["study"].get("execution_backend", "local")),
        docker_container=args.docker_container or config["study"].get("docker_container"),
        container_repo_root=args.container_repo_root or config["study"].get("container_repo_root"),
    )
    validate_execution_backend(execution)
    params = load_param_overrides_from_file(params_path)
    base_param_path = resolve_path(repo_root, config["study"]["base_param_file"])
    base_values = parse_ardupilot_parameter_file(base_param_path)
    base_values.update(params)
    validate_wind_only_servo_functions(base_values)
    validate_wind_only_sdf(
        repo_root / "models/dave_robot_models/description/sailboat/model.sdf"
    )

    wind = wind_world_enu_from_condition(
        float(condition["tws_mps"]),
        float(condition["target_heading_deg"]),
        float(condition["signed_twa_deg"]),
    )
    scenario = make_scenario(config, wind)
    context = build_context(
        make_bo_compatible_config(config),
        scenario,
        params,
        repo_root,
        workspace_root,
        results_dir,
        int(args.repeat_idx),
    )
    write_trial_manifest(context)
    write_trial_param_file(base_param_path, params, context.files.param_file)
    trial_world_name = render_trial_world(context)
    launch_plan = build_launch_plan(
        context, execution, trial_world_name, prepare_remote_inputs=True
    )
    manifest = json.loads(context.files.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "single_polar_cell",
            "performance_config": str(config_path),
            "fixed_heading_contract": "GUIDED SET_POSITION_TARGET_LOCAL_NED vx/vy/yaw",
            "waypoint_mission": False,
            "residual_rl": False,
            "wind_only": True,
            "git_commit": git_head(repo_root),
        }
    )
    context.files.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    endpoint = str(context.study_cfg["mavlink_endpoint"])
    if is_tcp_endpoint_open(endpoint):
        raise SystemExit(f"MAVLink endpoint already in use: {endpoint}")

    process = None
    master = None
    collector: PerformanceRosCollector | None = None
    records: list[dict[str, Any]] = []
    measurement: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {
        "status": "failure",
        "failure_reason": "not_started",
        "steady_state_reached": False,
        "time_to_steady_s": None,
        "measurement_duration_s": 0.0,
        "last_steady_reasons": [],
        "steady_state_stats": None,
        "last_steady_stats": {},
    }
    exit_code = 1
    release_sim_time: float | None = None
    last_sample_sim_time: float | None = None
    measurement_start_sim_time: float | None = None
    steady_detector = SteadyStateDetector(
        SteadyStateConfig.from_mapping(performance.get("steady_state"))
    )
    tack_detected = False
    expected_tack_sign = 1 if float(condition["signed_twa_deg"]) >= 0.0 else -1

    try:
        print(f"[polar_trial] trial_id: {context.trial_id}")
        print(f"[polar_trial] launch: {launch_plan.display_cmd}")
        process = start_launch_process(context, launch_plan)
        publish_startup_state_held(context, execution, True, reason="performance_prearm")
        publish_sail_force_enabled(context, execution, False, reason="performance_prearm")
        collector = PerformanceRosCollector(
            str(context.study_cfg["namespace"]),
            str(scenario["world"].get("internal_world_name", "waves")),
        )
        collector.start()
        master = connect_mavlink(
            endpoint,
            timeout_s=float(context.study_cfg.get("mavlink_connect_timeout_s", 120.0)),
            launch_process=process,
            stdout_log=context.files.logs_dir / "launch.stdout.log",
            stderr_log=context.files.logs_dir / "launch.stderr.log",
        )
        time.sleep(3.0)
        arm_vehicle(master, context, attempt_arm_until_armed)
        set_vehicle_mode(master, "GUIDED", timeout_s=10.0)
        mavlink = get_mavlink_module(master)
        # Reuse the validated BO stream request: ATTITUDE is 10 Hz and WIND is
        # 5 Hz, which keeps both independent sources inside the stale gate.
        request_message_intervals(master)
        guided_mode = int(master.mode_mapping()["GUIDED"])
        target = GuidedHeadingTarget(
            master,
            mavlink,
            float(condition["target_heading_deg"]),
            float(performance["desired_speed_mps"]),
            float(performance["target_refresh_hz"]),
        )
        target.send_if_due(time.monotonic(), force=True)
        ensure_residual_disabled(
            execution,
            str(performance.get("command_adapter_node", "/sailboat_command_adapter")),
        )
        collector.wait_ready(
            float(performance["telemetry_startup_timeout_s"]),
            float(performance["maximum_source_age_s"]),
        )
        settled_at = wait_for_wind_settle(
            [
                context.files.logs_dir / "launch.stdout.log",
                context.files.logs_dir / "launch.stderr.log",
            ],
            expected_world=Vector3(wind.east, wind.north, 0.0),
            tolerance_mps=float(performance["wind_settle_tolerance_mps"]),
            hold_sim_time_s=float(performance["wind_settle_hold_s"]),
            timeout_s=float(performance["wind_settle_timeout_s"]),
            poll_interval_s=float(performance["wind_settle_poll_s"]),
        )
        print(f"[polar_trial] wind settled at sim_time_s={settled_at:.3f}")
        publish_startup_state_held(context, execution, False, reason="armed_guided")
        if float(context.study_cfg.get("startup_state_release_settle_s", 0.2)) > 0.0:
            time.sleep(float(context.study_cfg.get("startup_state_release_settle_s", 0.2)))
        publish_sail_force_enabled(context, execution, True, reason="armed_guided")
        if float(context.study_cfg.get("sail_force_enable_settle_s", 0.5)) > 0.0:
            time.sleep(float(context.study_cfg.get("sail_force_enable_settle_s", 0.5)))
        released_snapshot = collector.wait_ready(
            float(performance["telemetry_startup_timeout_s"]),
            float(performance["maximum_source_age_s"]),
        )
        release_sim_time = released_snapshot.sim_time_s
        if release_sim_time is None:
            raise RuntimeError("release odometry has no simulation timestamp")
        mav_state = MavlinkPerformanceState()
        period_s = 1.0 / float(performance["sample_hz"])
        # Poll faster than the requested simulation-time sample rate, then
        # accept at most one row per configured simulation-time interval.
        poll_period_s = min(0.05, 0.5 / float(performance["sample_hz"]))
        next_wall_sample = time.monotonic()
        loop_started_wall_s = next_wall_sample
        maximum_wall_duration_s = float(
            performance.get(
                "max_wall_duration_s",
                3.0 * float(performance["max_sim_duration_s"]) + 120.0,
            )
        )
        while True:
            now = time.monotonic()
            if process.poll() is not None:
                raise RuntimeError("simulation launch process exited during polar trial")
            if now - loop_started_wall_s >= maximum_wall_duration_s:
                outcome["failure_reason"] = "wall_timeout"
                break
            target.send_if_due(now)
            drain_mavlink(master, mavlink, mav_state)
            if now + 1e-9 < next_wall_sample:
                time.sleep(min(0.02, next_wall_sample - now))
                continue
            next_wall_sample = now + poll_period_s
            ros = collector.snapshot()
            if ros.sim_time_s is None or (
                last_sample_sim_time is not None
                and ros.sim_time_s - last_sample_sim_time < 0.95 * period_s
            ):
                continue
            last_sample_sim_time = ros.sim_time_s
            elapsed = ros.sim_time_s - release_sim_time
            # MAVLink state is updated synchronously above.  Read the common
            # age reference only after draining the source messages.
            mavlink_reference_now = time.monotonic()
            wind_age = mav_state.wind_age_s(mavlink_reference_now)
            mavlink_attitude_age = mav_state.attitude_age_s(
                mavlink_reference_now
            )
            measured_wind_to_east = None
            measured_wind_to_north = None
            if mav_state.wind_from_deg is not None and mav_state.wind_speed_mps is not None:
                wind_to_unit = bearing_unit_enu(mav_state.wind_from_deg + 180.0)
                measured_wind_to_east = mav_state.wind_speed_mps * wind_to_unit.east
                measured_wind_to_north = mav_state.wind_speed_mps * wind_to_unit.north
            measured_twa = (
                signed_twa_deg(mav_state.wind_from_deg, ros.heading_deg)
                if mav_state.wind_from_deg is not None and ros.heading_deg is not None
                else None
            )
            if (
                measured_twa is not None
                and abs(measured_twa) >= float(performance["tack_detection_min_twa_deg"])
                and (1 if measured_twa >= 0.0 else -1) != expected_tack_sign
            ):
                tack_detected = True
            upwind_bearing = mav_state.wind_from_deg
            downwind_bearing = (
                (upwind_bearing + 180.0) % 360.0
                if upwind_bearing is not None
                else None
            )
            vmg_upwind = (
                vmg_toward_bearing_mps(ros.velocity_east_mps, ros.velocity_north_mps, upwind_bearing)
                if ros.velocity_east_mps is not None
                and ros.velocity_north_mps is not None
                and upwind_bearing is not None
                else None
            )
            vmg_downwind = (
                vmg_toward_bearing_mps(ros.velocity_east_mps, ros.velocity_north_mps, downwind_bearing)
                if ros.velocity_east_mps is not None
                and ros.velocity_north_mps is not None
                and downwind_bearing is not None
                else None
            )
            maximum_age = float(performance["maximum_source_age_s"])
            required_values = (
                ros.heading_deg,
                ros.roll_frd_deg,
                ros.speed_over_ground_mps,
                ros.velocity_east_mps,
                ros.velocity_north_mps,
                ros.gazebo_angular_yaw_rate_deg_s,
                ros.actual_rudder_angle_rad,
                ros.actual_rudder_velocity_rad_s,
                ros.actual_boom_angle_rad,
                ros.base_rudder_rad,
                ros.final_rudder_rad,
                ros.base_sheet_allowance_rad,
                ros.final_sheet_allowance_rad,
                mav_state.wind_from_deg,
                mav_state.wind_speed_mps,
                mav_state.yaw_deg,
                mav_state.yaw_rate_deg_s,
            )
            telemetry_valid = (
                all(value is not None and math.isfinite(float(value)) for value in required_values)
                and telemetry_ages_are_fresh(
                    (
                        ros.odom_age_s,
                        ros.rudder_age_s,
                        ros.boom_age_s,
                        ros.actuator_age_s,
                        wind_age,
                        mavlink_attitude_age,
                    ),
                    maximum_age,
                )
            )
            force_ramp_complete = elapsed >= float(performance["sail_force_ramp_s"])
            wind_settled = (
                mav_state.wind_speed_mps is not None
                and abs(mav_state.wind_speed_mps - float(condition["tws_mps"]))
                <= float(performance["mavlink_wind_speed_tolerance_mps"])
                and measured_twa is not None
                and abs(angle_error_deg(measured_twa, float(condition["signed_twa_deg"])))
                <= float(performance["mavlink_wind_direction_tolerance_deg"])
            )
            guided_active = mav_state.heartbeat_custom_mode == guided_mode
            target_fields = guided_target_record_fields(target, ros.heading_deg)
            comparison_fields = diagnostic_comparison_fields(ros, mav_state)
            steady = steady_detector.update(
                SteadyObservation(
                    sim_time_s=ros.sim_time_s,
                    heading_error_deg=(
                        float(target_fields["heading_error_deg"])
                        if target_fields["heading_error_deg"] is not None
                        else 180.0
                    ),
                    heading_deg=(ros.heading_deg if ros.heading_deg is not None else 0.0),
                    speed_mps=(ros.speed_over_ground_mps if ros.speed_over_ground_mps is not None else 0.0),
                    roll_deg=(ros.roll_frd_deg if ros.roll_frd_deg is not None else 0.0),
                    telemetry_valid=telemetry_valid,
                    wind_settled=wind_settled,
                    force_ramp_complete=force_ramp_complete,
                    tack_detected=tack_detected,
                )
            )
            outcome["last_steady_reasons"] = list(steady.reasons)
            outcome["last_steady_stats"] = steady.stats
            if steady.ready and measurement_start_sim_time is None:
                measurement_start_sim_time = ros.sim_time_s
                outcome["steady_state_reached"] = True
                outcome["time_to_steady_s"] = elapsed
                outcome["steady_state_stats"] = steady.stats
                print(f"[polar_trial] steady state reached at elapsed_sim_s={elapsed:.2f}")
            phase = "measurement" if measurement_start_sim_time is not None else "settling"
            record = {
                "trial_id": context.trial_id,
                "phase": phase,
                "sample_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_since_release_s": elapsed,
                **asdict(ros),
                **target_fields,
                **comparison_fields,
                "target_tws_mps": float(condition["tws_mps"]),
                "target_signed_twa_deg": float(condition["signed_twa_deg"]),
                "commanded_wind_east_mps": wind.east,
                "commanded_wind_north_mps": wind.north,
                "measured_wind_from_deg": mav_state.wind_from_deg,
                "measured_wind_speed_mps": mav_state.wind_speed_mps,
                "measured_wind_speed_z_mps": mav_state.wind_speed_z_mps,
                "measured_wind_to_east_mps": measured_wind_to_east,
                "measured_wind_to_north_mps": measured_wind_to_north,
                "measured_signed_twa_deg": measured_twa,
                "mavlink_yaw_deg": mav_state.yaw_deg,
                "mavlink_yaw_rate_deg_s": mav_state.yaw_rate_deg_s,
                "vmg_upwind_mps": vmg_upwind,
                "vmg_downwind_mps": vmg_downwind,
                "wind_age_s": wind_age,
                "mavlink_attitude_age_s": mavlink_attitude_age,
                "mavlink_status_text": mav_state.status_text,
                "guided_mode_active": guided_active,
                "wind_settled": wind_settled,
                "force_ramp_complete": force_ramp_complete,
                "tack_detected": tack_detected,
                "telemetry_valid": telemetry_valid,
                "steady_qualified": steady.qualified,
                "steady_ready": steady.ready,
                "steady_reasons": ";".join(steady.reasons),
            }
            records.append(record)
            if phase == "measurement":
                measurement.append(record)

            if ros.roll_frd_deg is not None and abs(ros.roll_frd_deg) > float(performance["maximum_roll_deg"]):
                outcome["failure_reason"] = "excessive_roll"
                break
            if mav_state.heartbeat_custom_mode is not None and not guided_active:
                outcome["failure_reason"] = "guided_mode_lost"
                break
            if tack_detected:
                outcome["failure_reason"] = "unexpected_tack"
                break
            if measurement_start_sim_time is not None:
                outcome["measurement_duration_s"] = ros.sim_time_s - measurement_start_sim_time
                if outcome["measurement_duration_s"] >= float(performance["measurement_duration_s"]):
                    valid_count = sum(1 for item in measurement if item["telemetry_valid"])
                    if valid_count < int(performance["minimum_measurement_samples"]):
                        outcome["failure_reason"] = "insufficient_measurement_samples"
                    else:
                        outcome["status"] = "success"
                        outcome["failure_reason"] = ""
                        exit_code = 0
                    break
            if elapsed >= float(performance["max_sim_duration_s"]):
                outcome["failure_reason"] = (
                    "not_steady" if measurement_start_sim_time is None else "measurement_timeout"
                )
                break
    except Exception as exc:
        outcome["failure_reason"] = "runtime_exception"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[polar_trial] ERROR: {outcome['error']}")
    finally:
        write_csv(context.files.samples_csv, records)
        if collector is not None:
            collector.close()
        if master is not None:
            try:
                master.close()
            except Exception:
                pass
        stop_launch_process(process, execution, launch_plan)
        endpoint_closed = wait_for_endpoint_closed(
            endpoint, float(context.study_cfg.get("mavlink_release_timeout_s", 15.0))
        )
        outcome["cleanup_endpoint_closed"] = endpoint_closed
        if not endpoint_closed:
            outcome["status"] = "failure"
            outcome["failure_reason"] = "cleanup_error"
            exit_code = 1

    summary = summarize(
        records,
        measurement,
        context=context,
        config_path=config_path,
        params_path=params_path,
        config=config,
        commanded_wind=wind,
        outcome=outcome,
        git_commit=git_head(repo_root),
        boom_limit_rad=math.radians(float(base_values["SAIL_ANGLE_MAX"])),
    )
    context.files.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"samples_csv: {context.files.samples_csv}")
    print(f"summary_json: {context.files.summary_json}")
    print(f"status: {outcome['status']}")
    print(f"failure_reason: {outcome['failure_reason']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
