"""Pure metrics for the unloaded fixed-rudder tracking diagnostic.

The live runner deliberately bypasses the ArduPilot steering output through
the existing command-adapter diagnostic override.  This module contains only
configuration validation and post-processing so it can be tested without ROS,
Gazebo, or SITL.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RudderTrackingConfig:
    sample_hz: float = 20.0
    setpoint_hold_s: float = 8.0
    steady_window_s: float = 2.0
    position_tolerance_deg: float = 1.0
    maximum_steady_error_deg: float = 1.0
    maximum_overshoot_deg: float = 2.0
    maximum_unloaded_speed_mps: float = 0.25
    pre_measurement_settle_required_s: float = 2.0
    pre_measurement_settle_timeout_s: float = 30.0
    joint_velocity_limit_rad_s: float = 3.0
    velocity_limit_threshold_fraction: float = 0.99
    maximum_velocity_limit_fraction: float = 0.05
    maximum_source_age_s: float = 0.75

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RudderTrackingConfig":
        config = cls(
            sample_hz=float(raw.get("sample_hz", cls.sample_hz)),
            setpoint_hold_s=float(
                raw.get("setpoint_hold_s", cls.setpoint_hold_s)
            ),
            steady_window_s=float(raw.get("steady_window_s", cls.steady_window_s)),
            position_tolerance_deg=float(
                raw.get("position_tolerance_deg", cls.position_tolerance_deg)
            ),
            maximum_steady_error_deg=float(
                raw.get(
                    "maximum_steady_error_deg", cls.maximum_steady_error_deg
                )
            ),
            maximum_overshoot_deg=float(
                raw.get("maximum_overshoot_deg", cls.maximum_overshoot_deg)
            ),
            maximum_unloaded_speed_mps=float(
                raw.get(
                    "maximum_unloaded_speed_mps",
                    cls.maximum_unloaded_speed_mps,
                )
            ),
            pre_measurement_settle_required_s=float(
                raw.get(
                    "pre_measurement_settle_required_s",
                    cls.pre_measurement_settle_required_s,
                )
            ),
            pre_measurement_settle_timeout_s=float(
                raw.get(
                    "pre_measurement_settle_timeout_s",
                    cls.pre_measurement_settle_timeout_s,
                )
            ),
            joint_velocity_limit_rad_s=float(
                raw.get(
                    "joint_velocity_limit_rad_s",
                    cls.joint_velocity_limit_rad_s,
                )
            ),
            velocity_limit_threshold_fraction=float(
                raw.get(
                    "velocity_limit_threshold_fraction",
                    cls.velocity_limit_threshold_fraction,
                )
            ),
            maximum_velocity_limit_fraction=float(
                raw.get(
                    "maximum_velocity_limit_fraction",
                    cls.maximum_velocity_limit_fraction,
                )
            ),
            maximum_source_age_s=float(
                raw.get("maximum_source_age_s", cls.maximum_source_age_s)
            ),
        )
        positive = (
            "sample_hz",
            "setpoint_hold_s",
            "steady_window_s",
            "position_tolerance_deg",
            "maximum_steady_error_deg",
            "maximum_overshoot_deg",
            "maximum_unloaded_speed_mps",
            "pre_measurement_settle_required_s",
            "pre_measurement_settle_timeout_s",
            "joint_velocity_limit_rad_s",
            "maximum_source_age_s",
        )
        for name in positive:
            if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"diagnostic.{name} must be finite and positive")
        if config.steady_window_s > config.setpoint_hold_s:
            raise ValueError(
                "diagnostic.steady_window_s must not exceed setpoint_hold_s"
            )
        for name in (
            "velocity_limit_threshold_fraction",
            "maximum_velocity_limit_fraction",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"diagnostic.{name} must be within [0, 1]")
        return config


@dataclass
class PreMeasurementSettlingState:
    """Track a continuous low-speed interval before rudder measurement.

    Stability duration is measured in simulation time.  The live runner owns
    the independent wall-clock timeout, so a paused or stalled simulation
    cannot wait forever.
    """

    required_speed_mps: float
    required_consecutive_s: float
    first_sim_time_s: float | None = None
    last_sim_time_s: float | None = None
    below_threshold_since_sim_time_s: float | None = None
    reached_sim_time_s: float | None = None
    initial_speed_mps: float | None = None
    final_speed_mps: float | None = None
    max_speed_mps: float | None = None
    sample_count: int = 0
    final_x_east_m: float | None = None
    final_y_north_m: float | None = None
    final_z_up_m: float | None = None
    final_roll_frd_deg: float | None = None
    final_pitch_frd_deg: float | None = None
    final_heading_deg: float | None = None
    neutral_rudder_owned_at_ready: bool = False

    def __post_init__(self) -> None:
        for name in ("required_speed_mps", "required_consecutive_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def reached(self) -> bool:
        return self.reached_sim_time_s is not None

    @property
    def consecutive_s(self) -> float:
        if (
            self.last_sim_time_s is None
            or self.below_threshold_since_sim_time_s is None
        ):
            return 0.0
        return max(
            0.0,
            self.last_sim_time_s - self.below_threshold_since_sim_time_s,
        )

    def observe(
        self,
        *,
        sim_time_s: float,
        speed_mps: float | None,
        telemetry_valid: bool,
        neutral_rudder_owned: bool,
        x_east_m: float | None = None,
        y_north_m: float | None = None,
        z_up_m: float | None = None,
        roll_frd_deg: float | None = None,
        pitch_frd_deg: float | None = None,
        heading_deg: float | None = None,
    ) -> bool:
        """Observe one new simulation sample and return readiness."""

        sim_time = float(sim_time_s)
        speed = float(speed_mps) if speed_mps is not None else None
        if not math.isfinite(sim_time):
            raise ValueError("settling samples require finite simulation time")
        if speed is not None and (not math.isfinite(speed) or speed < 0.0):
            raise ValueError("settling speed must be finite and non-negative")
        if self.last_sim_time_s is not None and sim_time < self.last_sim_time_s:
            raise ValueError("settling simulation time must be non-decreasing")
        if self.last_sim_time_s == sim_time:
            return self.reached

        if self.first_sim_time_s is None:
            self.first_sim_time_s = sim_time
        if self.initial_speed_mps is None and speed is not None:
            self.initial_speed_mps = speed
        self.last_sim_time_s = sim_time
        if speed is not None:
            self.final_speed_mps = speed
            self.max_speed_mps = (
                speed
                if self.max_speed_mps is None
                else max(self.max_speed_mps, speed)
            )
        self.sample_count += 1
        self.final_x_east_m = x_east_m
        self.final_y_north_m = y_north_m
        self.final_z_up_m = z_up_m
        self.final_roll_frd_deg = roll_frd_deg
        self.final_pitch_frd_deg = pitch_frd_deg
        self.final_heading_deg = heading_deg

        qualifies = (
            bool(telemetry_valid)
            and bool(neutral_rudder_owned)
            and speed is not None
            and speed <= self.required_speed_mps
        )
        if not qualifies:
            self.below_threshold_since_sim_time_s = None
            return self.reached
        if self.below_threshold_since_sim_time_s is None:
            self.below_threshold_since_sim_time_s = sim_time
        if self.consecutive_s + 1.0e-9 >= self.required_consecutive_s:
            self.reached_sim_time_s = sim_time
            self.neutral_rudder_owned_at_ready = True
        return self.reached

    def to_summary(self, *, timeout_s: float, wall_duration_s: float) -> dict[str, Any]:
        duration_s = (
            self.last_sim_time_s - self.first_sim_time_s
            if self.last_sim_time_s is not None
            and self.first_sim_time_s is not None
            else 0.0
        )
        return {
            "reached": self.reached,
            "duration_s": duration_s,
            "required_speed_mps": self.required_speed_mps,
            "required_consecutive_s": self.required_consecutive_s,
            "timeout_s": float(timeout_s),
            "initial_speed_mps": self.initial_speed_mps,
            "final_speed_mps": self.final_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "consecutive_s": self.consecutive_s,
            "sample_count": self.sample_count,
            "criterion_clock": "simulation_time",
            "timeout_clock": "wall_time",
            "wall_duration_s": float(wall_duration_s),
            "neutral_rudder_owned_at_ready": self.neutral_rudder_owned_at_ready,
            "final_x_east_m": self.final_x_east_m,
            "final_y_north_m": self.final_y_north_m,
            "final_z_up_m": self.final_z_up_m,
            "final_roll_frd_deg": self.final_roll_frd_deg,
            "final_pitch_frd_deg": self.final_pitch_frd_deg,
            "final_heading_deg": self.final_heading_deg,
        }


def pre_measurement_settle_timed_out(
    *, wall_elapsed_s: float, timeout_s: float, reached: bool
) -> bool:
    """Apply the independent wall-clock safety timeout."""

    elapsed = float(wall_elapsed_s)
    timeout = float(timeout_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("wall_elapsed_s must be finite and non-negative")
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    return not bool(reached) and elapsed >= timeout


def validate_setpoints_deg(raw: Sequence[Any]) -> tuple[float, ...]:
    """Validate the small-angle diagnostic sequence without changing it."""

    if not raw:
        raise ValueError("diagnostic.setpoints_deg must not be empty")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("diagnostic.setpoints_deg must contain finite values")
    if any(abs(value) > 10.0 for value in values):
        raise ValueError("unloaded diagnostic setpoints must remain within +/-10 deg")
    return values


def _statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "max_abs": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": fmean(numeric),
        "std": pstdev(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "max_abs": max(abs(value) for value in numeric),
    }


def _first_rise_time_s(
    records: Sequence[Mapping[str, Any]],
    *,
    start_angle_rad: float,
    target_rad: float,
) -> float | None:
    change = target_rad - start_angle_rad
    if abs(change) <= math.radians(0.05):
        return 0.0
    direction = 1.0 if change > 0.0 else -1.0
    threshold = start_angle_rad + 0.9 * change
    for record in records:
        actual = float(record["actual_rudder_angle_rad"])
        if direction * (actual - threshold) >= 0.0:
            return float(record["elapsed_step_s"])
    return None


def _settling_time_s(
    records: Sequence[Mapping[str, Any]],
    *,
    target_rad: float,
    tolerance_rad: float,
) -> float | None:
    for index, record in enumerate(records):
        if all(
            abs(float(item["actual_rudder_angle_rad"]) - target_rad)
            <= tolerance_rad
            for item in records[index:]
        ):
            return float(record["elapsed_step_s"])
    return None


def summarize_tracking_step(
    records: Sequence[Mapping[str, Any]],
    *,
    start_angle_rad: float,
    target_rad: float,
    config: RudderTrackingConfig,
) -> dict[str, Any]:
    """Summarize one fixed target and apply the diagnostic-only gates."""

    valid = [record for record in records if bool(record.get("telemetry_valid"))]
    if not valid:
        return {
            "target_rudder_rad": target_rad,
            "target_rudder_deg": math.degrees(target_rad),
            "sample_count": len(records),
            "valid_sample_count": 0,
            "passed": False,
            "failed_gates": ["telemetry"],
        }

    end_s = max(float(record["elapsed_step_s"]) for record in valid)
    steady = [
        record
        for record in valid
        if float(record["elapsed_step_s"]) >= end_s - config.steady_window_s
    ]
    actual = [float(record["actual_rudder_angle_rad"]) for record in valid]
    final_command = [float(record["final_rudder_rad"]) for record in valid]
    velocity = [float(record["actual_rudder_velocity_rad_s"]) for record in valid]
    speed = [float(record["speed_over_ground_mps"]) for record in valid]
    physical_error = [value - target_rad for value in actual]
    steady_error = [
        float(record["actual_rudder_angle_rad"]) - target_rad
        for record in steady
    ]
    command_error = [value - target_rad for value in final_command]
    steady_command_error = [
        float(record["final_rudder_rad"]) - target_rad for record in steady
    ]
    direction = 1.0 if target_rad >= start_angle_rad else -1.0
    overshoot_rad = max(
        0.0,
        max(direction * (value - target_rad) for value in actual),
    )
    velocity_threshold = (
        config.joint_velocity_limit_rad_s
        * config.velocity_limit_threshold_fraction
    )
    velocity_limit_fraction = sum(
        abs(value) >= velocity_threshold for value in velocity
    ) / len(velocity)
    tolerance_rad = math.radians(config.position_tolerance_deg)
    steady_mean_abs_deg = math.degrees(abs(fmean(steady_error)))
    steady_max_abs_deg = math.degrees(max(abs(value) for value in steady_error))
    steady_rmse_deg = math.degrees(
        math.sqrt(fmean(value * value for value in steady_error))
    )
    maximum_speed = max(speed)
    rise_time_s = _first_rise_time_s(
        valid,
        start_angle_rad=start_angle_rad,
        target_rad=target_rad,
    )
    settling_time_s = _settling_time_s(
        valid,
        target_rad=target_rad,
        tolerance_rad=tolerance_rad,
    )

    gates = {
        "telemetry": len(valid) == len(records),
        # Parameter writes and the adapter's next base-command callback are
        # asynchronous.  Only the final steady window must contain the target;
        # an old command in the first sample is an expected transient.
        "single_owner_command_delivered": max(
            abs(value) for value in steady_command_error
        ) <= math.radians(0.1),
        "unloaded_speed": maximum_speed <= config.maximum_unloaded_speed_mps,
        "steady_tracking_error": (
            steady_max_abs_deg <= config.maximum_steady_error_deg
        ),
        "overshoot": (
            math.degrees(overshoot_rad) <= config.maximum_overshoot_deg
        ),
        "rise_reached": rise_time_s is not None,
        # Require a complete steady window after entering the error band; a
        # final isolated in-band sample is not a valid settled response.
        "settled": (
            settling_time_s is not None
            and settling_time_s <= end_s - config.steady_window_s
        ),
        "velocity_limit_fraction": (
            velocity_limit_fraction <= config.maximum_velocity_limit_fraction
        ),
    }
    return {
        "target_rudder_rad": target_rad,
        "target_rudder_deg": math.degrees(target_rad),
        "start_actual_rudder_rad": start_angle_rad,
        "start_actual_rudder_deg": math.degrees(start_angle_rad),
        "sample_count": len(records),
        "valid_sample_count": len(valid),
        "actual_rudder_angle_rad": _statistics(actual),
        "final_command_error_rad": _statistics(command_error),
        "steady_final_command_error_rad": _statistics(steady_command_error),
        "tracking_error_rad": _statistics(physical_error),
        "steady_tracking_error_rad": _statistics(steady_error),
        "steady_tracking_error_mean_abs_deg": steady_mean_abs_deg,
        "steady_tracking_error_max_abs_deg": steady_max_abs_deg,
        "steady_tracking_error_rmse_deg": steady_rmse_deg,
        "rise_time_s": rise_time_s,
        "settling_time_s": settling_time_s,
        "maximum_overshoot_rad": overshoot_rad,
        "maximum_overshoot_deg": math.degrees(overshoot_rad),
        "actual_rudder_velocity_rad_s": _statistics(velocity),
        "velocity_limit_threshold_rad_s": velocity_threshold,
        "velocity_limit_fraction": velocity_limit_fraction,
        "speed_over_ground_mps": _statistics(speed),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "passed": all(gates.values()),
    }


def summarize_tracking_run(
    records: Sequence[Mapping[str, Any]],
    *,
    starts_by_step: Mapping[int, float],
    targets_by_step: Mapping[int, float],
    config: RudderTrackingConfig,
) -> dict[str, Any]:
    steps = []
    for step_index in sorted(targets_by_step):
        step_records = [
            record
            for record in records
            if int(record["step_index"]) == step_index
        ]
        steps.append(
            summarize_tracking_step(
                step_records,
                start_angle_rad=float(starts_by_step[step_index]),
                target_rad=float(targets_by_step[step_index]),
                config=config,
            )
        )
    return {
        "step_count": len(steps),
        "steps": steps,
        "passed": bool(steps) and all(step["passed"] for step in steps),
    }
