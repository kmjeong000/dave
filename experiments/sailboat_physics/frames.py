"""Pure coordinate transforms used by the sailboat frame diagnostic.

Gazebo uses an ENU world and this sailboat model points along model ``+Y``.
The model axes map to a conventional aircraft/body FRD frame as follows::

    forward = model +Y
    right   = model +X
    down    = model -Z

Keeping this math independent of ROS, Gazebo and pymavlink makes the frame
contract directly unit-testable before any SDF transform is changed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Iterable, Sequence


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

# This matrix is both:
#   * ENU vector -> NED vector, and
#   * model XYZ vector -> body FRD vector for the +Y-forward hull.
# It is symmetric and self-inverse.
_ENU_TO_NED_AND_MODEL_TO_FRD: Matrix3 = (
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
)


@dataclass(frozen=True)
class AttitudeDeg:
    """Roll, pitch and compass heading in a body FRD / world NED frame."""

    roll_deg: float
    pitch_deg: float
    heading_deg: float


@dataclass(frozen=True)
class FrameSample:
    """One time-aligned Gazebo and MAVLink attitude observation."""

    elapsed_wall_s: float
    gazebo_roll_deg: float
    gazebo_pitch_deg: float
    gazebo_heading_deg: float
    mav_roll_deg: float
    mav_pitch_deg: float
    mav_heading_deg: float
    gazebo_qx: float | None = None
    gazebo_qy: float | None = None
    gazebo_qz: float | None = None
    gazebo_qw: float | None = None
    mav_roll_rad: float | None = None
    mav_pitch_rad: float | None = None
    mav_yaw_rad: float | None = None


def normalize_degrees_360(value: float) -> float:
    normalized = float(value) % 360.0
    if math.isclose(normalized, 0.0, abs_tol=1e-12) or math.isclose(
        normalized,
        360.0,
        abs_tol=1e-12,
    ):
        return 0.0
    return normalized


def normalize_degrees_signed(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-12):
        return 180.0
    return normalized


def attitude_error_deg(actual_deg: float, expected_deg: float) -> float:
    """Return the shortest signed angular error ``actual - expected``."""

    return normalize_degrees_signed(float(actual_deg) - float(expected_deg))


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _quaternion_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> Matrix3:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _rpy_rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Matrix3:
    """Return the standard 3-2-1 ``Rz(yaw) Ry(pitch) Rx(roll)`` matrix."""

    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _rotation_matrix_to_rpy(matrix: Matrix3) -> tuple[float, float, float]:
    sin_pitch = max(-1.0, min(1.0, -matrix[2][0]))
    pitch = math.asin(sin_pitch)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-9:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        # The diagnostics stay far from gimbal lock. This deterministic branch
        # still keeps conversion well-defined for completeness.
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return roll, pitch, yaw


def _rotation_matrix_to_quaternion(matrix: Matrix3) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
    return x, y, z, w


def gazebo_quaternion_to_frd_attitude_deg(
    x: float,
    y: float,
    z: float,
    w: float,
) -> AttitudeDeg:
    """Convert a Gazebo ENU/model-XYZ quaternion to NED/body-FRD attitude."""

    gazebo_enu_from_model = _quaternion_to_rotation_matrix(x, y, z, w)
    basis = _ENU_TO_NED_AND_MODEL_TO_FRD
    ned_from_frd = _matmul(_matmul(basis, gazebo_enu_from_model), basis)
    roll, pitch, yaw = _rotation_matrix_to_rpy(ned_from_frd)
    return AttitudeDeg(
        roll_deg=math.degrees(roll),
        pitch_deg=math.degrees(pitch),
        heading_deg=normalize_degrees_360(math.degrees(yaw)),
    )


def frd_attitude_to_gazebo_quaternion(
    roll_deg: float,
    pitch_deg: float,
    heading_deg: float,
) -> tuple[float, float, float, float]:
    """Return Gazebo model quaternion for a requested body-FRD attitude."""

    ned_from_frd = _rpy_rotation_matrix(
        math.radians(float(roll_deg)),
        math.radians(float(pitch_deg)),
        math.radians(float(heading_deg)),
    )
    basis = _ENU_TO_NED_AND_MODEL_TO_FRD
    gazebo_enu_from_model = _matmul(_matmul(basis, ned_from_frd), basis)
    return _rotation_matrix_to_quaternion(gazebo_enu_from_model)


def frd_attitude_to_gazebo_euler_deg(
    roll_deg: float,
    pitch_deg: float,
    heading_deg: float,
) -> tuple[float, float, float]:
    """Return Gazebo model roll/pitch/yaw for a requested FRD attitude."""

    quaternion = frd_attitude_to_gazebo_quaternion(roll_deg, pitch_deg, heading_deg)
    matrix = _quaternion_to_rotation_matrix(*quaternion)
    roll, pitch, yaw = _rotation_matrix_to_rpy(matrix)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _circular_mean_deg(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one angle is required")
    sin_mean = fmean(math.sin(math.radians(value)) for value in values)
    cos_mean = fmean(math.cos(math.radians(value)) for value in values)
    if math.hypot(sin_mean, cos_mean) <= 1e-12:
        raise ValueError("circular mean is undefined for the supplied angles")
    return normalize_degrees_360(math.degrees(math.atan2(sin_mean, cos_mean)))


def _rmse(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise ValueError("at least one value is required")
    return math.sqrt(fmean(value * value for value in collected))


def summarize_frame_case(
    samples: Sequence[FrameSample],
    *,
    expected: AttitudeDeg,
    heading_tolerance_deg: float = 3.0,
    attitude_tolerance_deg: float = 3.0,
    minimum_samples: int = 10,
) -> dict[str, object]:
    """Summarize one held-pose case and evaluate frame-alignment gates."""

    if not samples:
        raise ValueError("frame case has no samples")

    gazebo_heading = _circular_mean_deg([sample.gazebo_heading_deg for sample in samples])
    mav_heading = _circular_mean_deg([sample.mav_heading_deg for sample in samples])
    gazebo_roll = fmean(sample.gazebo_roll_deg for sample in samples)
    gazebo_pitch = fmean(sample.gazebo_pitch_deg for sample in samples)
    mav_roll = fmean(sample.mav_roll_deg for sample in samples)
    mav_pitch = fmean(sample.mav_pitch_deg for sample in samples)

    gazebo_heading_error = attitude_error_deg(gazebo_heading, expected.heading_deg)
    mav_heading_error = attitude_error_deg(mav_heading, expected.heading_deg)
    mav_gazebo_heading_error = attitude_error_deg(mav_heading, gazebo_heading)
    gazebo_roll_error = gazebo_roll - expected.roll_deg
    mav_roll_error = mav_roll - expected.roll_deg
    mav_gazebo_roll_error = mav_roll - gazebo_roll
    gazebo_pitch_error = gazebo_pitch - expected.pitch_deg
    mav_pitch_error = mav_pitch - expected.pitch_deg
    mav_gazebo_pitch_error = mav_pitch - gazebo_pitch

    heading_rmse = _rmse(
        attitude_error_deg(sample.mav_heading_deg, sample.gazebo_heading_deg)
        for sample in samples
    )
    roll_rmse = _rmse(
        sample.mav_roll_deg - sample.gazebo_roll_deg for sample in samples
    )
    pitch_rmse = _rmse(
        sample.mav_pitch_deg - sample.gazebo_pitch_deg for sample in samples
    )

    gates = {
        "minimum_samples": len(samples) >= int(minimum_samples),
        "gazebo_heading_expected": abs(gazebo_heading_error) <= heading_tolerance_deg,
        "mav_heading_expected": abs(mav_heading_error) <= heading_tolerance_deg,
        "mav_gazebo_heading": abs(mav_gazebo_heading_error) <= heading_tolerance_deg,
        "gazebo_roll_expected": abs(gazebo_roll_error) <= attitude_tolerance_deg,
        "mav_roll_expected": abs(mav_roll_error) <= attitude_tolerance_deg,
        "mav_gazebo_roll": abs(mav_gazebo_roll_error) <= attitude_tolerance_deg,
        "gazebo_pitch_expected": abs(gazebo_pitch_error) <= attitude_tolerance_deg,
        "mav_pitch_expected": abs(mav_pitch_error) <= attitude_tolerance_deg,
        "mav_gazebo_pitch": abs(mav_gazebo_pitch_error) <= attitude_tolerance_deg,
    }
    return {
        "sample_count": len(samples),
        "expected": asdict(expected),
        "mean": {
            "gazebo_roll_deg": gazebo_roll,
            "gazebo_pitch_deg": gazebo_pitch,
            "gazebo_heading_deg": gazebo_heading,
            "mav_roll_deg": mav_roll,
            "mav_pitch_deg": mav_pitch,
            "mav_heading_deg": mav_heading,
        },
        "error": {
            "gazebo_heading_expected_deg": gazebo_heading_error,
            "mav_heading_expected_deg": mav_heading_error,
            "mav_gazebo_heading_deg": mav_gazebo_heading_error,
            "gazebo_roll_expected_deg": gazebo_roll_error,
            "mav_roll_expected_deg": mav_roll_error,
            "mav_gazebo_roll_deg": mav_gazebo_roll_error,
            "gazebo_pitch_expected_deg": gazebo_pitch_error,
            "mav_pitch_expected_deg": mav_pitch_error,
            "mav_gazebo_pitch_deg": mav_gazebo_pitch_error,
        },
        "rmse": {
            "mav_gazebo_heading_deg": heading_rmse,
            "mav_gazebo_roll_deg": roll_rmse,
            "mav_gazebo_pitch_deg": pitch_rmse,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
