"""Pure frame and sailing-performance calculations.

Gazebo world vectors use ENU (x=east, y=north, z=up).  Navigation headings
are compass/NED bearings (0=north, positive clockwise).  World wind vectors
point *to* where the air moves, while TWA uses the meteorological wind-from
direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Vector2:
    east: float
    north: float

    @property
    def magnitude(self) -> float:
        return math.hypot(self.east, self.north)


def normalize_degrees_360(value: float) -> float:
    normalized = float(value) % 360.0
    return 0.0 if math.isclose(normalized, 360.0, abs_tol=1e-12) else normalized


def normalize_degrees_signed(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-12):
        return 180.0
    return normalized


def angle_error_deg(actual_deg: float, target_deg: float) -> float:
    """Return the shortest signed angle ``actual - target``."""

    return normalize_degrees_signed(float(actual_deg) - float(target_deg))


def heading_rate_deg_s(
    previous_heading_deg: float,
    current_heading_deg: float,
    elapsed_s: float,
) -> float:
    """Return wrapped compass-heading rate, positive clockwise, in deg/s."""

    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("elapsed_s must be finite and positive")
    return angle_error_deg(current_heading_deg, previous_heading_deg) / elapsed


def bearing_unit_enu(bearing_deg: float) -> Vector2:
    """Return an ENU unit vector for a compass/NED bearing."""

    bearing_rad = math.radians(float(bearing_deg))
    return Vector2(east=math.sin(bearing_rad), north=math.cos(bearing_rad))


def wind_world_enu_from_condition(
    tws_mps: float,
    target_heading_deg: float,
    target_signed_twa_deg: float,
) -> Vector2:
    """Return the Gazebo wind-to vector for a fixed-heading polar cell."""

    speed = float(tws_mps)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("tws_mps must be finite and positive")
    signed_twa = float(target_signed_twa_deg)
    if not math.isfinite(signed_twa) or not -180.0 <= signed_twa <= 180.0:
        raise ValueError("target_signed_twa_deg must be within [-180, 180]")

    wind_from_deg = normalize_degrees_360(float(target_heading_deg) + signed_twa)
    wind_to = bearing_unit_enu(wind_from_deg + 180.0)
    return Vector2(east=speed * wind_to.east, north=speed * wind_to.north)


def wind_from_bearing_deg(wind_world_enu: Vector2) -> float | None:
    if wind_world_enu.magnitude <= 1e-12:
        return None
    wind_to_deg = normalize_degrees_360(
        math.degrees(math.atan2(wind_world_enu.east, wind_world_enu.north))
    )
    return normalize_degrees_360(wind_to_deg + 180.0)


def signed_twa_deg(wind_from_deg: float, boat_heading_deg: float) -> float:
    """Return signed true-wind angle: wind-from bearing minus heading."""

    return normalize_degrees_signed(float(wind_from_deg) - float(boat_heading_deg))


def course_over_ground_deg(
    velocity_east_mps: float,
    velocity_north_mps: float,
    *,
    minimum_speed_mps: float = 1e-6,
) -> float | None:
    if math.hypot(float(velocity_east_mps), float(velocity_north_mps)) < float(
        minimum_speed_mps
    ):
        return None
    return normalize_degrees_360(
        math.degrees(math.atan2(float(velocity_east_mps), float(velocity_north_mps)))
    )


def vmg_toward_bearing_mps(
    velocity_east_mps: float,
    velocity_north_mps: float,
    bearing_deg: float,
) -> float:
    target = bearing_unit_enu(bearing_deg)
    return (
        float(velocity_east_mps) * target.east
        + float(velocity_north_mps) * target.north
    )


def model_velocity_to_world_enu(
    model_vx_mps: float,
    model_vy_mps: float,
    model_vz_mps: float,
    quaternion_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate Gazebo OdometryPublisher body/model twist into world ENU.

    gz-sim8's 3-D OdometryPublisher rotates world displacement back into the
    model body frame before publishing ``twist.linear``.  This sailboat's
    model +Y is forward and +X is starboard; applying the model quaternion
    returns the unambiguous world ENU velocity needed for VMG.
    """

    if len(quaternion_xyzw) != 4:
        raise ValueError("quaternion_xyzw must contain four values")
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Efficient quaternion-vector rotation: v' = v + 2w(q x v) + 2q x(q x v)
    vx = float(model_vx_mps)
    vy = float(model_vy_mps)
    vz = float(model_vz_mps)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def circular_mean_deg(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one angle is required")
    sine = fmean(math.sin(math.radians(float(value))) for value in values)
    cosine = fmean(math.cos(math.radians(float(value))) for value in values)
    if math.hypot(sine, cosine) <= 1e-12:
        raise ValueError("circular mean is undefined for the supplied angles")
    return normalize_degrees_360(math.degrees(math.atan2(sine, cosine)))


def circular_std_deg(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one angle is required")
    sine = fmean(math.sin(math.radians(float(value))) for value in values)
    cosine = fmean(math.cos(math.radians(float(value))) for value in values)
    resultant = min(1.0, max(1e-15, math.hypot(sine, cosine)))
    return math.degrees(math.sqrt(-2.0 * math.log(resultant)))


def linear_slope(values: Sequence[tuple[float, float]]) -> float:
    """Return least-squares dy/dt for finite ``(time, value)`` pairs."""

    finite = [
        (float(t), float(value))
        for t, value in values
        if math.isfinite(float(t)) and math.isfinite(float(value))
    ]
    if len(finite) < 2:
        raise ValueError("at least two finite samples are required")
    mean_t = fmean(t for t, _ in finite)
    mean_value = fmean(value for _, value in finite)
    denominator = sum((t - mean_t) ** 2 for t, _ in finite)
    if denominator <= 1e-12:
        raise ValueError("sample times must span a non-zero interval")
    return sum(
        (t - mean_t) * (value - mean_value) for t, value in finite
    ) / denominator


def scalar_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": fmean(finite),
        "std": pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }
