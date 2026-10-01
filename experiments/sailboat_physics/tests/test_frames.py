from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.sailboat_physics.frames import (
    AttitudeDeg,
    FrameSample,
    attitude_error_deg,
    frd_attitude_to_gazebo_euler_deg,
    frd_attitude_to_gazebo_quaternion,
    gazebo_quaternion_to_frd_attitude_deg,
    summarize_frame_case,
)
from experiments.sailboat_physics.tests.sdf_test_utils import (
    parse_sdf_with_extensions,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_FILE = REPO_ROOT / "models/dave_robot_models/description/sailboat/model.sdf"


def _pose_values(text: str | None) -> list[float]:
    assert text is not None
    values = [float(value) for value in text.split()]
    assert len(values) == 6
    return values


def _matrix_multiply(left, right):
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _matrix_transpose(matrix):
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _matrix_vector_multiply(matrix, vector):
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _rotation_matrix_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    roll_matrix = (
        (1.0, 0.0, 0.0),
        (0.0, cr, -sr),
        (0.0, sr, cr),
    )
    pitch_matrix = (
        (cp, 0.0, sp),
        (0.0, 1.0, 0.0),
        (-sp, 0.0, cp),
    )
    yaw_matrix = (
        (cy, -sy, 0.0),
        (sy, cy, 0.0),
        (0.0, 0.0, 1.0),
    )
    return _matrix_multiply(
        _matrix_multiply(yaw_matrix, pitch_matrix),
        roll_matrix,
    )


def test_imu_sensor_pose_composes_to_ardupilot_aircraft_body_axes():
    root = parse_sdf_with_extensions(MODEL_FILE)
    sensor = root.find(".//sensor[@name='imu_sensor']")
    plugin = root.find(".//plugin[@name='ArduPilotPlugin']")

    assert sensor is not None
    assert plugin is not None

    sensor_pose = _pose_values(sensor.findtext("pose"))
    model_to_aircraft_pose = _pose_values(
        plugin.findtext("modelXYZToAirplaneXForwardZDown")
    )
    assert sensor_pose[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    # SDF pose rotation maps sensor-frame basis vectors into the model frame.
    # Gazebo reports IMU vectors in the sensor-local frame, so model-frame
    # vectors use the inverse (transpose) rotation before ArduPilotPlugin
    # forwards them unchanged as aircraft-body vectors.
    sensor_to_model = _rotation_matrix_from_rpy(*sensor_pose[3:])
    model_to_sensor = _matrix_transpose(sensor_to_model)
    model_to_aircraft = _rotation_matrix_from_rpy(*model_to_aircraft_pose[3:])

    axis_tolerance = 1e-3
    model_basis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for basis_vector in model_basis:
        reported_sensor_vector = _matrix_vector_multiply(
            model_to_sensor, basis_vector
        )
        expected_aircraft_vector = _matrix_vector_multiply(
            model_to_aircraft, basis_vector
        )
        assert reported_sensor_vector == pytest.approx(
            expected_aircraft_vector, abs=axis_tolerance
        )

    sensor_basis_expected_in_model = (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    for sensor_basis, expected_model_axis in zip(
        model_basis, sensor_basis_expected_in_model
    ):
        actual_model_axis = _matrix_vector_multiply(sensor_to_model, sensor_basis)
        assert actual_model_axis == pytest.approx(
            expected_model_axis, abs=axis_tolerance
        )


def quaternion_from_gazebo_rpy_deg(roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


@pytest.mark.parametrize(
    ("gazebo_yaw_deg", "expected_heading_deg"),
    [(0.0, 0.0), (-90.0, 90.0), (180.0, 180.0), (90.0, 270.0)],
)
def test_y_forward_hull_cardinal_heading_mapping(gazebo_yaw_deg, expected_heading_deg):
    attitude = gazebo_quaternion_to_frd_attitude_deg(
        *quaternion_from_gazebo_rpy_deg(0.0, 0.0, gazebo_yaw_deg)
    )

    assert attitude.heading_deg == pytest.approx(expected_heading_deg)
    assert attitude.roll_deg == pytest.approx(0.0)
    assert attitude.pitch_deg == pytest.approx(0.0)


def test_gazebo_model_pitch_is_physical_frd_roll_for_y_forward_hull():
    attitude = gazebo_quaternion_to_frd_attitude_deg(
        *quaternion_from_gazebo_rpy_deg(0.0, 12.0, 0.0)
    )

    assert attitude.roll_deg == pytest.approx(12.0)
    assert attitude.pitch_deg == pytest.approx(0.0)


def test_gazebo_model_roll_is_physical_frd_pitch_for_y_forward_hull():
    attitude = gazebo_quaternion_to_frd_attitude_deg(
        *quaternion_from_gazebo_rpy_deg(-7.0, 0.0, 0.0)
    )

    assert attitude.roll_deg == pytest.approx(0.0)
    assert attitude.pitch_deg == pytest.approx(-7.0)


@pytest.mark.parametrize(
    "expected",
    [
        AttitudeDeg(0.0, 0.0, 0.0),
        AttitudeDeg(0.0, 0.0, 90.0),
        AttitudeDeg(10.0, 0.0, 0.0),
        AttitudeDeg(-10.0, 5.0, 270.0),
    ],
)
def test_frd_to_gazebo_round_trip(expected):
    quaternion = frd_attitude_to_gazebo_quaternion(
        expected.roll_deg,
        expected.pitch_deg,
        expected.heading_deg,
    )

    actual = gazebo_quaternion_to_frd_attitude_deg(*quaternion)

    assert actual.roll_deg == pytest.approx(expected.roll_deg)
    assert actual.pitch_deg == pytest.approx(expected.pitch_deg)
    assert attitude_error_deg(actual.heading_deg, expected.heading_deg) == pytest.approx(0.0)


def test_heading_90_maps_to_negative_90_gazebo_yaw():
    roll_deg, pitch_deg, yaw_deg = frd_attitude_to_gazebo_euler_deg(0.0, 0.0, 90.0)

    assert roll_deg == pytest.approx(0.0)
    assert pitch_deg == pytest.approx(0.0)
    assert yaw_deg == pytest.approx(-90.0)


def make_sample(**overrides):
    values = {
        "elapsed_wall_s": 0.0,
        "gazebo_roll_deg": 10.0,
        "gazebo_pitch_deg": 0.0,
        "gazebo_heading_deg": 359.0,
        "mav_roll_deg": 11.0,
        "mav_pitch_deg": 1.0,
        "mav_heading_deg": 1.0,
    }
    values.update(overrides)
    return FrameSample(**values)


def test_summary_uses_circular_heading_mean_and_component_gates():
    samples = [
        make_sample(gazebo_heading_deg=359.0, mav_heading_deg=1.0),
        make_sample(gazebo_heading_deg=1.0, mav_heading_deg=359.0),
    ] * 5

    summary = summarize_frame_case(
        samples,
        expected=AttitudeDeg(roll_deg=10.0, pitch_deg=0.0, heading_deg=0.0),
    )

    assert summary["sample_count"] == 10
    assert summary["mean"]["gazebo_heading_deg"] == pytest.approx(0.0)
    assert summary["mean"]["mav_heading_deg"] == pytest.approx(0.0)
    assert summary["passed"] is True


def test_summary_reports_fixed_heading_offset_as_failure():
    samples = [make_sample(mav_heading_deg=90.0)] * 10

    summary = summarize_frame_case(
        samples,
        expected=AttitudeDeg(roll_deg=10.0, pitch_deg=0.0, heading_deg=0.0),
    )

    assert summary["error"]["mav_gazebo_heading_deg"] == pytest.approx(91.0)
    assert summary["gates"]["mav_gazebo_heading"] is False
    assert summary["passed"] is False
