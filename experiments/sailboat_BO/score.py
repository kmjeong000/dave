from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .common import clip
except ImportError:
    from common import clip


BASE_SUMMARY_COLUMNS = [
    "trial_id",
    "study_name",
    "scenario_id",
    "scenario_split",
    "repeat_idx",
    "status",
    "failure_reason",
    "started_at_utc",
    "finished_at_utc",
    "duration_wall_s",
    "samples_csv",
    "param_file",
    "world_file",
    "spawn_x_m",
    "spawn_y_m",
    "spawn_z_m",
    "spawn_yaw_deg",
    "wind_x_mps",
    "wind_y_mps",
    "wind_z_mps",
    "mission_waypoint_count",
    "mission_frame",
    "position_source",
    "roll_source",
]

METRIC_COLUMNS = [
    "mission_time_s",
    "progress_ratio",
    "final_distance_to_wp_m",
    "xte_rms_m",
    "xte_integral_m_s",
    "mean_speed_mps",
    "max_speed_mps",
    "min_speed_mps",
    "max_abs_roll_deg",
    "max_abs_roll_mav_deg",
    "max_abs_roll_gz_odom_deg",
    "max_abs_roll_gz_imu_deg",
    "rudder_total_variation_rad",
    "sail_total_variation_rad",
    "rudder_rms_rad",
    "sail_rms_rad",
    "waypoint_switch_count",
    "stuck_time_s",
    "roll_source_valid_ratio",
    "roll_violation_total_s",
    "roll_violation_peak_continuous_s",
    "max_abs_roll_deg_after_grace",
    "mean_realtime_factor",
    "final_realtime_factor",
    "min_realtime_factor",
]

CONSTRAINT_COLUMNS = [
    "constraint_mission_complete",
    "constraint_timeout",
    "constraint_stuck",
    "constraint_no_progress",
    "constraint_excessive_roll",
]

OBJECTIVE_COLUMNS = [
    "penalty",
    "time_term",
    "xte_term",
    "progress_term",
    "control_term",
    "roll_term",
    "scenario_cost",
]


def build_summary_columns(param_names: Iterable[str]) -> list[str]:
    return (
        BASE_SUMMARY_COLUMNS
        + [f"param__{name}" for name in param_names]
        + METRIC_COLUMNS
        + CONSTRAINT_COLUMNS
        + OBJECTIVE_COLUMNS
    )

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _total_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(b - a) for a, b in zip(values[:-1], values[1:]))


def _trapezoid_integral(values: list[float], times: list[float]) -> float:
    if len(values) < 2 or len(times) < 2:
        return 0.0
    integral = 0.0
    for idx in range(1, min(len(values), len(times))):
        dt = max(0.0, times[idx] - times[idx - 1])
        integral += 0.5 * (values[idx] + values[idx - 1]) * dt
    return integral


def compute_metrics(
    samples: list[Mapping[str, Any]],
    online_stats: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    online_stats = dict(online_stats or {})
    if not samples:
        metrics = {column: 0.0 for column in METRIC_COLUMNS}
        metrics["stuck_time_s"] = _safe_float(online_stats.get("stuck_time_s"))
        metrics["roll_source_valid_ratio"] = _safe_float(online_stats.get("roll_source_valid_ratio"))
        metrics["roll_violation_total_s"] = _safe_float(online_stats.get("roll_violation_total_s"))
        metrics["roll_violation_peak_continuous_s"] = _safe_float(
            online_stats.get("roll_violation_peak_continuous_s")
        )
        metrics["max_abs_roll_deg_after_grace"] = _safe_float(
            online_stats.get("max_abs_roll_deg_after_grace")
        )
        metrics["mean_realtime_factor"] = _safe_float(online_stats.get("mean_realtime_factor"))
        metrics["final_realtime_factor"] = _safe_float(online_stats.get("final_realtime_factor"))
        metrics["min_realtime_factor"] = _safe_float(online_stats.get("min_realtime_factor"))
        return metrics

    times = [_safe_float(sample.get("sim_time_s")) for sample in samples]
    distances = [_safe_float(sample.get("distance_to_wp_m")) for sample in samples]
    xte_values = [abs(_safe_float(sample.get("cross_track_error_m"))) for sample in samples]
    speeds = [_safe_float(sample.get("surge_speed_mps")) for sample in samples]
    rolls = [abs(_safe_float(sample.get("roll_deg"))) for sample in samples]
    roll_mav = [abs(_safe_float(sample.get("roll_mav_deg"))) for sample in samples]
    roll_gz_odom = [abs(_safe_float(sample.get("roll_gz_odom_deg"))) for sample in samples]
    roll_gz_imu = [abs(_safe_float(sample.get("roll_gz_imu_deg"))) for sample in samples]
    roll_source_valid = [
        1.0 if bool(sample.get("roll_source_valid")) else 0.0
        for sample in samples
    ]
    rudders = [_safe_float(sample.get("rudder_cmd_rad")) for sample in samples]
    sails = [_safe_float(sample.get("sail_cmd_rad")) for sample in samples]
    progress = [clip(_safe_float(sample.get("progress_ratio")), 0.0, 1.0) for sample in samples]
    waypoint_indices = [int(_safe_float(sample.get("waypoint_index"), 0.0)) for sample in samples]
    realtime_factors = [_safe_float(sample.get("realtime_factor")) for sample in samples]
    sample_mean_realtime_factor = (
        sum(realtime_factors) / len(realtime_factors) if realtime_factors else 0.0
    )
    sample_final_realtime_factor = realtime_factors[-1] if realtime_factors else 0.0
    sample_min_realtime_factor = min(realtime_factors) if realtime_factors else 0.0

    metrics = {
        "mission_time_s": max(0.0, times[-1] - times[0]),
        "progress_ratio": max(progress) if progress else 0.0,
        "final_distance_to_wp_m": distances[-1] if distances else 0.0,
        "xte_rms_m": _rms(xte_values),
        "xte_integral_m_s": _trapezoid_integral(xte_values, times),
        "mean_speed_mps": sum(speeds) / len(speeds) if speeds else 0.0,
        "max_speed_mps": max(speeds) if speeds else 0.0,
        "min_speed_mps": min(speeds) if speeds else 0.0,
        "max_abs_roll_deg": max(rolls) if rolls else 0.0,
        "max_abs_roll_mav_deg": max(roll_mav) if roll_mav else 0.0,
        "max_abs_roll_gz_odom_deg": max(roll_gz_odom) if roll_gz_odom else 0.0,
        "max_abs_roll_gz_imu_deg": max(roll_gz_imu) if roll_gz_imu else 0.0,
        "rudder_total_variation_rad": _total_variation(rudders),
        "sail_total_variation_rad": _total_variation(sails),
        "rudder_rms_rad": _rms(rudders),
        "sail_rms_rad": _rms(sails),
        "waypoint_switch_count": float(
            sum(1 for a, b in zip(waypoint_indices[:-1], waypoint_indices[1:]) if b != a)
        ),
        "stuck_time_s": _safe_float(online_stats.get("stuck_time_s")),
        "roll_source_valid_ratio": sum(roll_source_valid) / len(roll_source_valid) if roll_source_valid else 0.0,
        "roll_violation_total_s": _safe_float(online_stats.get("roll_violation_total_s")),
        "roll_violation_peak_continuous_s": _safe_float(
            online_stats.get("roll_violation_peak_continuous_s")
        ),
        "max_abs_roll_deg_after_grace": _safe_float(
            online_stats.get("max_abs_roll_deg_after_grace"),
            max(rolls) if rolls else 0.0,
        ),
        "mean_realtime_factor": _safe_float(
            online_stats.get("mean_realtime_factor"),
            sample_mean_realtime_factor,
        ),
        "final_realtime_factor": _safe_float(
            online_stats.get("final_realtime_factor"),
            sample_final_realtime_factor,
        ),
        "min_realtime_factor": _safe_float(
            online_stats.get("min_realtime_factor"),
            sample_min_realtime_factor,
        ),
    }
    return metrics


def compute_constraints(
    metrics: Mapping[str, float],
    termination_cfg: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, bool]:
    success_radius_m = _safe_float(termination_cfg.get("success_radius_m"), 5.0)
    max_roll_deg = _safe_float(termination_cfg.get("max_roll_deg"), 45.0)
    roll_violation_window_s = _safe_float(termination_cfg.get("roll_violation_window_s"), 0.0)
    progress_ratio = metrics.get("progress_ratio", 0.0)
    final_distance_to_wp_m = metrics.get("final_distance_to_wp_m", 1e9)

    mission_complete = bool(outcome.get("mission_complete"))
    if not outcome.get("mission_complete_explicit"):
        mission_complete = (
            progress_ratio >= 0.999
            and final_distance_to_wp_m <= success_radius_m
        )
    elif mission_complete and progress_ratio < 0.95 and final_distance_to_wp_m > (3.0 * success_radius_m):
        mission_complete = False

    excessive_roll = bool(outcome.get("excessive_roll"))
    if not excessive_roll and roll_violation_window_s > 0.0:
        excessive_roll = metrics.get("roll_violation_peak_continuous_s", 0.0) >= roll_violation_window_s
    if not excessive_roll and roll_violation_window_s <= 0.0:
        excessive_roll = metrics.get("max_abs_roll_deg", 0.0) > max_roll_deg

    return {
        "constraint_mission_complete": mission_complete,
        "constraint_timeout": bool(outcome.get("timeout")),
        "constraint_stuck": bool(outcome.get("stuck")),
        "constraint_no_progress": bool(outcome.get("no_progress")),
        "constraint_excessive_roll": excessive_roll,
    }


def compute_objective(
    metrics: Mapping[str, float],
    constraints: Mapping[str, bool],
    termination_cfg: Mapping[str, Any],
) -> dict[str, float]:
    timeout_s = _safe_float(termination_cfg.get("timeout_s"), 240.0)

    violated = (
        not constraints["constraint_mission_complete"]
        or constraints["constraint_timeout"]
        or constraints["constraint_stuck"]
        or constraints["constraint_no_progress"]
        or constraints["constraint_excessive_roll"]
    )
    penalty = 5.0 if violated else 0.0

    time_term = clip(metrics.get("mission_time_s", 0.0) / max(timeout_s, 1e-6), 0.0, 1.0)
    xte_term = clip(metrics.get("xte_rms_m", 0.0) / 10.0, 0.0, 1.0)
    progress_term = 1.0 - clip(metrics.get("progress_ratio", 0.0), 0.0, 1.0)
    control_raw = metrics.get("rudder_total_variation_rad", 0.0) + (
        0.5 * metrics.get("sail_total_variation_rad", 0.0)
    )
    control_term = clip(control_raw / 40.0, 0.0, 1.0)
    roll_reference_deg = metrics.get(
        "max_abs_roll_deg_after_grace",
        metrics.get("max_abs_roll_deg", 0.0),
    )
    roll_excess = max(0.0, roll_reference_deg - 30.0)
    roll_term = clip(roll_excess / 15.0, 0.0, 1.0)

    scenario_cost = (
        penalty
        + 0.35 * time_term
        + 0.25 * xte_term
        + 0.20 * progress_term
        + 0.10 * control_term
        + 0.10 * roll_term
    )

    return {
        "penalty": penalty,
        "time_term": time_term,
        "xte_term": xte_term,
        "progress_term": progress_term,
        "control_term": control_term,
        "roll_term": roll_term,
        "scenario_cost": scenario_cost,
    }


def build_summary_row(
    *,
    param_names: Iterable[str],
    metadata: Mapping[str, Any],
    params: Mapping[str, float],
    metrics: Mapping[str, float],
    constraints: Mapping[str, bool],
    objective: Mapping[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in BASE_SUMMARY_COLUMNS:
        row[key] = metadata.get(key)

    for name in param_names:
        row[f"param__{name}"] = params.get(name)

    for key in METRIC_COLUMNS:
        row[key] = metrics.get(key)
    for key in CONSTRAINT_COLUMNS:
        row[key] = constraints.get(key)
    for key in OBJECTIVE_COLUMNS:
        row[key] = objective.get(key)
    return row


def append_summary_row(csv_path: str | Path, row: Mapping[str, Any], columns: Iterable[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns)
    existing_rows: list[dict[str, Any]] = []
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not needs_header:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames or []
            if existing_header != fieldnames:
                existing_rows = [dict(item) for item in reader]
                with csv_path.open("w", newline="", encoding="utf-8") as rewrite_handle:
                    rewrite_writer = csv.DictWriter(rewrite_handle, fieldnames=fieldnames)
                    rewrite_writer.writeheader()
                    for existing_row in existing_rows:
                        rewrite_writer.writerow({key: existing_row.get(key) for key in fieldnames})
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})
