"""Pure parsing and checks for sail-wrench tacking diagnostics.

The simulator emits one rate-limited ``SailLiftDragSystem`` record containing
the actual boom angle, applied world force, force application lever arm, and
the resulting moment projected onto the hull roll axis.  This module keeps the
parsing and sign checks independent of Gazebo so they can be unit-tested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from statistics import fmean
from typing import Sequence


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SAIL_WRENCH_LINE = re.compile(
    rf"\[SailLiftDragSystem\].*?simTimeS=({_NUMBER})"
    rf".*?forceScale=({_NUMBER})"
    rf".*?boomAngleDeg=({_NUMBER})"
    rf".*?rawForceWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?appliedForceWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?cpWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?leverFromBaseWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?momentAboutBaseWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?rollAxisWorld=({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
    rf".*?rollMomentNm=({_NUMBER})"
)


@dataclass(frozen=True)
class SailWrenchSample:
    sim_time_s: float
    force_scale: float
    boom_angle_deg: float
    raw_force_x_n: float
    raw_force_y_n: float
    raw_force_z_n: float
    applied_force_x_n: float
    applied_force_y_n: float
    applied_force_z_n: float
    cp_world_x_m: float
    cp_world_y_m: float
    cp_world_z_m: float
    lever_world_x_m: float
    lever_world_y_m: float
    lever_world_z_m: float
    moment_world_x_nm: float
    moment_world_y_nm: float
    moment_world_z_nm: float
    roll_axis_world_x: float
    roll_axis_world_y: float
    roll_axis_world_z: float
    roll_moment_nm: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def parse_sail_wrench(text: str) -> list[SailWrenchSample]:
    """Parse unique, time-ordered sail-wrench records from launch output."""

    text = _ANSI_ESCAPE.sub("", text)
    records: dict[float, SailWrenchSample] = {}
    for match in _SAIL_WRENCH_LINE.finditer(text):
        values = [float(match.group(index)) for index in range(1, 23)]
        record = SailWrenchSample(*values)
        records[record.sim_time_s] = record
    return [records[key] for key in sorted(records)]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(left * right for left, right in zip(a, b))


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


def summarize_sail_wrench(
    samples: Sequence[SailWrenchSample],
    *,
    boom_deadband_deg: float = 5.0,
    minimum_force_n: float = 0.1,
    minimum_samples_per_side: int = 3,
    roll_moment_deadband_nm: float = 0.5,
    wrench_math_tolerance_nm: float = 0.05,
) -> dict[str, object]:
    """Check whether applied sail roll moment reverses across boom sides."""

    active: list[SailWrenchSample] = []
    moment_errors: list[float] = []
    projection_errors: list[float] = []

    for sample in samples:
        applied_force = (
            sample.applied_force_x_n,
            sample.applied_force_y_n,
            sample.applied_force_z_n,
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
        roll_axis = (
            sample.roll_axis_world_x,
            sample.roll_axis_world_y,
            sample.roll_axis_world_z,
        )
        recomputed_moment = _cross(lever, applied_force)
        moment_errors.append(
            _norm(
                tuple(
                    actual - expected
                    for actual, expected in zip(logged_moment, recomputed_moment)
                )
            )
        )
        projection_errors.append(
            abs(sample.roll_moment_nm - _dot(logged_moment, roll_axis))
        )

        if (
            sample.force_scale > 1e-9
            and _norm(applied_force) >= minimum_force_n
            and abs(sample.boom_angle_deg) > boom_deadband_deg
        ):
            active.append(sample)

    negative = [sample for sample in active if sample.boom_angle_deg < 0.0]
    positive = [sample for sample in active if sample.boom_angle_deg > 0.0]
    negative_mean = fmean(sample.roll_moment_nm for sample in negative) if negative else None
    positive_mean = fmean(sample.roll_moment_nm for sample in positive) if positive else None
    both_sides_observed = (
        len(negative) >= minimum_samples_per_side
        and len(positive) >= minimum_samples_per_side
    )
    roll_moment_reverses = bool(
        both_sides_observed
        and negative_mean is not None
        and positive_mean is not None
        and abs(negative_mean) >= roll_moment_deadband_nm
        and abs(positive_mean) >= roll_moment_deadband_nm
        and negative_mean * positive_mean < 0.0
    )
    maximum_wrench_error = max(moment_errors, default=0.0)
    maximum_projection_error = max(projection_errors, default=0.0)
    wrench_math_consistent = (
        maximum_wrench_error <= wrench_math_tolerance_nm
        and maximum_projection_error <= wrench_math_tolerance_nm
    )
    boom_roll_moment_correlation = _correlation(
        [sample.boom_angle_deg for sample in active],
        [sample.roll_moment_nm for sample in active],
    )

    gates = {
        "minimum_samples_each_boom_side": both_sides_observed,
        "wrench_math_consistent": wrench_math_consistent,
        "roll_moment_reverses_with_boom_side": roll_moment_reverses,
    }
    return {
        "sample_count": len(samples),
        "active_force_sample_count": len(active),
        "negative_boom_sample_count": len(negative),
        "positive_boom_sample_count": len(positive),
        "negative_boom_mean_roll_moment_nm": negative_mean,
        "positive_boom_mean_roll_moment_nm": positive_mean,
        "boom_roll_moment_correlation": boom_roll_moment_correlation,
        "maximum_moment_cross_product_error_nm": maximum_wrench_error,
        "maximum_roll_projection_error_nm": maximum_projection_error,
        "gates": gates,
        "passed": all(gates.values()),
    }
