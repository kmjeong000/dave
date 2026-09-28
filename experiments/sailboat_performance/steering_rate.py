"""Pure metrics for direct GUIDED steering-rate characterization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any


def symmetric_turn_rate_sequence(
    rate_deg_s: float,
    active_hold_s: float,
    final_zero_hold_s: float,
) -> tuple[tuple[float, float], ...]:
    """Return a zero-net-heading +rate/-rate/+rate diagnostic sequence."""

    rate = float(rate_deg_s)
    active = float(active_hold_s)
    zero = float(final_zero_hold_s)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_deg_s must be finite and positive")
    if not math.isfinite(active) or active <= 0.0:
        raise ValueError("active_hold_s must be finite and positive")
    if not math.isfinite(zero) or zero <= 0.0:
        raise ValueError("final_zero_hold_s must be finite and positive")
    return ((rate, active), (-rate, 2.0 * active), (rate, active), (0.0, zero))


def _rms(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return math.sqrt(fmean(value * value for value in data)) if data else None


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return fmean(data) if data else None


def _settling_time(
    rows: Sequence[dict[str, Any]],
    *,
    threshold_deg_s: float,
    hold_s: float,
) -> float | None:
    for index, row in enumerate(rows):
        start = float(row["step_elapsed_s"])
        window = [
            candidate
            for candidate in rows[index:]
            if float(candidate["step_elapsed_s"]) <= start + hold_s + 1e-9
        ]
        if not window or float(window[-1]["step_elapsed_s"]) < start + hold_s - 0.1:
            continue
        if all(
            abs(float(candidate["mavlink_yaw_rate_deg_s"])) <= threshold_deg_s
            for candidate in window
        ):
            return start
    return None


def summarize_turn_rate_steps(
    records: Sequence[dict[str, Any]],
    *,
    rudder_limit_deg: float = 45.0,
    saturation_margin_deg: float = 1.0,
    steady_tail_s: float = 1.0,
    zero_settle_threshold_deg_s: float = 2.0,
    zero_settle_hold_s: float = 1.0,
) -> dict[str, Any]:
    """Summarize direct rate tracking without imposing a tuning pass gate."""

    valid = [
        row
        for row in records
        if bool(row.get("telemetry_valid"))
        and row.get("mavlink_yaw_rate_deg_s") is not None
        and row.get("target_turn_rate_deg_s") is not None
    ]
    step_indices = sorted({int(row["step_index"]) for row in valid})
    steps: list[dict[str, Any]] = []
    positive_peaks: list[float] = []
    negative_peaks: list[float] = []

    for step_index in step_indices:
        rows = [row for row in valid if int(row["step_index"]) == step_index]
        if not rows:
            continue
        target = float(rows[0]["target_turn_rate_deg_s"])
        actual = [float(row["mavlink_yaw_rate_deg_s"]) for row in rows]
        gazebo = [
            float(row["gazebo_angular_yaw_rate_deg_s"])
            for row in rows
            if row.get("gazebo_angular_yaw_rate_deg_s") is not None
        ]
        errors = [target - value for value in actual]
        end_time = max(float(row["step_elapsed_s"]) for row in rows)
        tail = [
            row
            for row in rows
            if float(row["step_elapsed_s"]) >= end_time - steady_tail_s
        ]
        final_commands_deg = [
            math.degrees(float(row["final_rudder_rad"]))
            for row in rows
            if row.get("final_rudder_rad") is not None
        ]
        tracking_errors_deg = [
            math.degrees(
                float(row["actual_rudder_angle_rad"])
                - float(row["final_rudder_rad"])
            )
            for row in rows
            if row.get("actual_rudder_angle_rad") is not None
            and row.get("final_rudder_rad") is not None
        ]
        sign = 1.0 if target > 0.0 else -1.0 if target < 0.0 else 0.0
        signed_response = [sign * value for value in actual] if sign else []
        peak = max(signed_response) if signed_response else max(abs(value) for value in actual)
        if target > 0.0:
            positive_peaks.append(peak)
        elif target < 0.0:
            negative_peaks.append(peak)
        rise_time = None
        if sign:
            threshold = 0.9 * abs(target)
            rise_time = next(
                (
                    float(row["step_elapsed_s"])
                    for row in rows
                    if sign * float(row["mavlink_yaw_rate_deg_s"]) >= threshold
                ),
                None,
            )
        saturated = [
            abs(value) >= rudder_limit_deg - saturation_margin_deg
            for value in final_commands_deg
        ]
        step = {
            "step_index": step_index,
            "target_turn_rate_deg_s": target,
            "sample_count": len(rows),
            "duration_s": end_time,
            "rise_time_90_s": rise_time,
            "peak_response_deg_s": peak,
            "overshoot_deg_s": max(0.0, peak - abs(target)) if sign else None,
            "rms_rate_error_deg_s": _rms(errors),
            "tail_mean_rate_error_deg_s": _mean(
                target - float(row["mavlink_yaw_rate_deg_s"]) for row in tail
            ),
            "target_sign_fraction": (
                fmean(1.0 if target * value > 0.0 else 0.0 for value in actual)
                if sign
                else None
            ),
            "rudder_command_peak_abs_deg": (
                max(abs(value) for value in final_commands_deg)
                if final_commands_deg
                else None
            ),
            "rudder_saturation_fraction": fmean(saturated) if saturated else None,
            "rudder_tracking_error_rms_deg": _rms(tracking_errors_deg),
            "mean_speed_mps": _mean(
                float(row["speed_over_ground_mps"])
                for row in rows
                if row.get("speed_over_ground_mps") is not None
            ),
            "minimum_speed_mps": min(
                (
                    float(row["speed_over_ground_mps"])
                    for row in rows
                    if row.get("speed_over_ground_mps") is not None
                ),
                default=None,
            ),
            "mavlink_gazebo_rate_error_rms_deg_s": (
                _rms(
                    float(row["mavlink_yaw_rate_deg_s"])
                    - float(row["gazebo_angular_yaw_rate_deg_s"])
                    for row in rows
                    if row.get("gazebo_angular_yaw_rate_deg_s") is not None
                )
                if gazebo
                else None
            ),
            "zero_rate_settling_time_s": (
                _settling_time(
                    rows,
                    threshold_deg_s=zero_settle_threshold_deg_s,
                    hold_s=zero_settle_hold_s,
                )
                if not sign
                else None
            ),
        }
        steps.append(step)

    positive = _mean(positive_peaks)
    negative = _mean(negative_peaks)
    denominator = (
        0.5 * (positive + negative)
        if positive is not None
        and negative is not None
        and positive > 0.0
        and negative > 0.0
        else None
    )
    return {
        "valid_sample_count": len(valid),
        "steps": steps,
        "positive_peak_mean_deg_s": positive,
        "negative_peak_mean_deg_s": negative,
        "reversal_peak_asymmetry_fraction": (
            abs(positive - negative) / denominator
            if denominator is not None and denominator > 0.0
            else None
        ),
    }
