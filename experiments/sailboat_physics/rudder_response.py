"""Pure parsing and checks for rudder command / hydrodynamic response.

``FoilLiftDragSystem`` reports the actual rudder angle, applied water force,
force lever arm and the resulting moment about the hull yaw axis.  This module
aligns those rate-limited records with BO ``raw/samples.csv`` commands so a
servo-sign hypothesis can be tested without inferring it from vehicle motion
alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import bisect
import math
import re
from statistics import fmean
from typing import Mapping, Sequence


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUDDER_WRENCH_LINE = re.compile(
    rf"\[FoilLiftDragSystem\].*?link=rudder_link"
    rf".*?simTimeS=({_NUMBER})"
    rf".*?Vfluid=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?speed=({_NUMBER})"
    rf".*?alpha_deg=({_NUMBER})"
    rf".*?cl=({_NUMBER})"
    rf".*?cd=({_NUMBER})"
    rf".*?forceWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?jointAngleDeg=({_NUMBER})"
    rf".*?commandDeg=({_NUMBER})"
    rf".*?cpWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?leverFromBaseWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?momentAboutBaseWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?yawAxisWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?yawMomentNm=({_NUMBER})"
)


@dataclass(frozen=True)
class RudderWrenchSample:
    sim_time_s: float
    water_relative_x_mps: float
    water_relative_y_mps: float
    water_relative_z_mps: float
    speed_mps: float
    alpha_deg: float
    cl: float
    cd: float
    force_world_x_n: float
    force_world_y_n: float
    force_world_z_n: float
    joint_angle_deg: float
    command_deg: float
    cp_world_x_m: float
    cp_world_y_m: float
    cp_world_z_m: float
    lever_world_x_m: float
    lever_world_y_m: float
    lever_world_z_m: float
    moment_world_x_nm: float
    moment_world_y_nm: float
    moment_world_z_nm: float
    yaw_axis_world_x: float
    yaw_axis_world_y: float
    yaw_axis_world_z: float
    yaw_moment_nm: float


@dataclass(frozen=True)
class RudderCommandSample:
    sim_time_s: float
    command_deg: float
    yaw_rate_deg_s: float


@dataclass(frozen=True)
class AlignedRudderSample:
    trial_id: str
    sim_time_s: float
    alignment_delta_s: float
    command_deg: float
    actual_angle_deg: float
    speed_mps: float
    force_world_x_n: float
    force_world_y_n: float
    force_world_z_n: float
    lever_world_x_m: float
    lever_world_y_m: float
    lever_world_z_m: float
    moment_world_x_nm: float
    moment_world_y_nm: float
    moment_world_z_nm: float
    yaw_axis_world_x: float
    yaw_axis_world_y: float
    yaw_axis_world_z: float
    yaw_moment_nm: float
    yaw_rate_deg_s: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


def parse_rudder_wrench(text: str) -> list[RudderWrenchSample]:
    """Parse unique, time-ordered rudder wrench records from launch logs."""

    text = _ANSI_ESCAPE.sub("", text)
    records: dict[float, RudderWrenchSample] = {}
    for match in _RUDDER_WRENCH_LINE.finditer(text):
        values = [float(match.group(index)) for index in range(1, 27)]
        record = RudderWrenchSample(*values)
        records[record.sim_time_s] = record
    return [records[key] for key in sorted(records)]


def _wrap_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def build_rudder_command_samples(
    rows: Sequence[Mapping[str, str]],
) -> list[RudderCommandSample]:
    """Convert BO sample rows into commands with a central-difference yaw rate."""

    points = sorted(
        (
            float(row["sim_time_s"]),
            math.degrees(float(row["rudder_cmd_rad"])),
            float(row["yaw_rad"]),
        )
        for row in rows
    )
    samples: list[RudderCommandSample] = []
    for index, (sim_time_s, command_deg, _yaw_rad) in enumerate(points):
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        elapsed_s = after[0] - before[0]
        yaw_rate_deg_s = 0.0
        if elapsed_s > 1e-9:
            yaw_rate_deg_s = math.degrees(
                _wrap_pi(after[2] - before[2]) / elapsed_s
            )
        samples.append(
            RudderCommandSample(
                sim_time_s=sim_time_s,
                command_deg=command_deg,
                yaw_rate_deg_s=yaw_rate_deg_s,
            )
        )
    return samples


def align_rudder_samples(
    trial_id: str,
    wrench_samples: Sequence[RudderWrenchSample],
    command_samples: Sequence[RudderCommandSample],
    *,
    maximum_time_delta_s: float = 0.75,
) -> list[AlignedRudderSample]:
    """Nearest-neighbour align plugin samples with the BO command time base."""

    if not command_samples:
        return []
    commands = sorted(command_samples, key=lambda sample: sample.sim_time_s)
    command_times = [sample.sim_time_s for sample in commands]
    aligned: list[AlignedRudderSample] = []
    for wrench in sorted(wrench_samples, key=lambda sample: sample.sim_time_s):
        insertion = bisect.bisect_left(command_times, wrench.sim_time_s)
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(commands)
        ]
        command = min(
            (commands[index] for index in candidates),
            key=lambda sample: abs(sample.sim_time_s - wrench.sim_time_s),
        )
        delta_s = abs(command.sim_time_s - wrench.sim_time_s)
        if delta_s > maximum_time_delta_s:
            continue
        aligned.append(
            AlignedRudderSample(
                trial_id=trial_id,
                sim_time_s=wrench.sim_time_s,
                alignment_delta_s=delta_s,
                # This is the final command observed on the same Gazebo topic
                # consumed by JointPositionController.  The raw BO command is
                # used only to obtain the nearest vehicle yaw-rate sample.
                command_deg=wrench.command_deg,
                actual_angle_deg=wrench.joint_angle_deg,
                speed_mps=wrench.speed_mps,
                force_world_x_n=wrench.force_world_x_n,
                force_world_y_n=wrench.force_world_y_n,
                force_world_z_n=wrench.force_world_z_n,
                lever_world_x_m=wrench.lever_world_x_m,
                lever_world_y_m=wrench.lever_world_y_m,
                lever_world_z_m=wrench.lever_world_z_m,
                moment_world_x_nm=wrench.moment_world_x_nm,
                moment_world_y_nm=wrench.moment_world_y_nm,
                moment_world_z_nm=wrench.moment_world_z_nm,
                yaw_axis_world_x=wrench.yaw_axis_world_x,
                yaw_axis_world_y=wrench.yaw_axis_world_y,
                yaw_axis_world_z=wrench.yaw_axis_world_z,
                yaw_moment_nm=wrench.yaw_moment_nm,
                yaw_rate_deg_s=command.yaw_rate_deg_s,
            )
        )
    return aligned


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def summarize_rudder_response(
    samples: Sequence[AlignedRudderSample],
    *,
    command_deadband_deg: float = 5.0,
    actual_angle_deadband_deg: float = 2.0,
    minimum_speed_mps: float = 0.25,
    minimum_force_n: float = 0.1,
    minimum_samples_per_side: int = 3,
    minimum_sign_agreement_ratio: float = 0.8,
    yaw_moment_deadband_nm: float = 0.5,
    wrench_math_tolerance_nm: float = 0.05,
) -> dict[str, object]:
    """Check the complete command -> angle -> yaw-moment sign chain."""

    moment_errors: list[float] = []
    projection_errors: list[float] = []
    active: list[AlignedRudderSample] = []
    for sample in samples:
        force = (
            sample.force_world_x_n,
            sample.force_world_y_n,
            sample.force_world_z_n,
        )
        lever = (
            sample.lever_world_x_m,
            sample.lever_world_y_m,
            sample.lever_world_z_m,
        )
        logged_moment = (
            sample.moment_world_x_nm,
            sample.moment_world_y_nm,
            sample.moment_world_z_nm,
        )
        yaw_axis = (
            sample.yaw_axis_world_x,
            sample.yaw_axis_world_y,
            sample.yaw_axis_world_z,
        )
        recomputed_moment = _cross(lever, force)
        moment_errors.append(
            _norm(
                tuple(
                    actual - expected
                    for actual, expected in zip(logged_moment, recomputed_moment)
                )
            )
        )
        projection_errors.append(
            abs(sample.yaw_moment_nm - _dot(logged_moment, yaw_axis))
        )
        if (
            abs(sample.command_deg) > command_deadband_deg
            and abs(sample.actual_angle_deg) > actual_angle_deadband_deg
            and sample.speed_mps >= minimum_speed_mps
            and _norm(force) >= minimum_force_n
        ):
            active.append(sample)

    negative_command = [sample for sample in active if sample.command_deg < 0.0]
    positive_command = [sample for sample in active if sample.command_deg > 0.0]
    both_command_sides = (
        len(negative_command) >= minimum_samples_per_side
        and len(positive_command) >= minimum_samples_per_side
    )
    sign_agreement_ratio = (
        sum(
            sample.command_deg * sample.actual_angle_deg > 0.0
            for sample in active
        )
        / len(active)
        if active
        else 0.0
    )
    negative_mean_angle = (
        fmean(sample.actual_angle_deg for sample in negative_command)
        if negative_command
        else None
    )
    positive_mean_angle = (
        fmean(sample.actual_angle_deg for sample in positive_command)
        if positive_command
        else None
    )
    actual_tracks_command = bool(
        both_command_sides
        and negative_mean_angle is not None
        and positive_mean_angle is not None
        and negative_mean_angle < -actual_angle_deadband_deg
        and positive_mean_angle > actual_angle_deadband_deg
        and sign_agreement_ratio >= minimum_sign_agreement_ratio
    )

    negative_rudder = [sample for sample in active if sample.actual_angle_deg < 0.0]
    positive_rudder = [sample for sample in active if sample.actual_angle_deg > 0.0]
    negative_mean_yaw_moment = (
        fmean(sample.yaw_moment_nm for sample in negative_rudder)
        if negative_rudder
        else None
    )
    positive_mean_yaw_moment = (
        fmean(sample.yaw_moment_nm for sample in positive_rudder)
        if positive_rudder
        else None
    )
    yaw_moment_reverses = bool(
        len(negative_rudder) >= minimum_samples_per_side
        and len(positive_rudder) >= minimum_samples_per_side
        and negative_mean_yaw_moment is not None
        and positive_mean_yaw_moment is not None
        and abs(negative_mean_yaw_moment) >= yaw_moment_deadband_nm
        and abs(positive_mean_yaw_moment) >= yaw_moment_deadband_nm
        and negative_mean_yaw_moment * positive_mean_yaw_moment < 0.0
    )

    maximum_moment_error = max(moment_errors, default=0.0)
    maximum_projection_error = max(projection_errors, default=0.0)
    wrench_math_consistent = (
        maximum_moment_error <= wrench_math_tolerance_nm
        and maximum_projection_error <= wrench_math_tolerance_nm
    )
    gates = {
        "minimum_samples_each_command_side": both_command_sides,
        "actual_rudder_tracks_command_sign": actual_tracks_command,
        "wrench_math_consistent": wrench_math_consistent,
        "yaw_moment_reverses_with_rudder_side": yaw_moment_reverses,
    }
    return {
        "aligned_sample_count": len(samples),
        "active_sample_count": len(active),
        "negative_command_sample_count": len(negative_command),
        "positive_command_sample_count": len(positive_command),
        "command_actual_sign_agreement_ratio": sign_agreement_ratio,
        "negative_command_mean_actual_angle_deg": negative_mean_angle,
        "positive_command_mean_actual_angle_deg": positive_mean_angle,
        "negative_rudder_mean_yaw_moment_nm": negative_mean_yaw_moment,
        "positive_rudder_mean_yaw_moment_nm": positive_mean_yaw_moment,
        "negative_command_mean_yaw_rate_deg_s": (
            fmean(sample.yaw_rate_deg_s for sample in negative_command)
            if negative_command
            else None
        ),
        "positive_command_mean_yaw_rate_deg_s": (
            fmean(sample.yaw_rate_deg_s for sample in positive_command)
            if positive_command
            else None
        ),
        "command_actual_angle_correlation": _correlation(
            [sample.command_deg for sample in active],
            [sample.actual_angle_deg for sample in active],
        ),
        "rudder_angle_yaw_moment_correlation": _correlation(
            [sample.actual_angle_deg for sample in active],
            [sample.yaw_moment_nm for sample in active],
        ),
        "yaw_moment_yaw_rate_correlation": _correlation(
            [sample.yaw_moment_nm for sample in active],
            [sample.yaw_rate_deg_s for sample in active],
        ),
        "maximum_alignment_delta_s": max(
            (sample.alignment_delta_s for sample in samples),
            default=None,
        ),
        "maximum_moment_cross_product_error_nm": maximum_moment_error,
        "maximum_yaw_projection_error_nm": maximum_projection_error,
        "gates": gates,
        "passed": all(gates.values()),
    }
