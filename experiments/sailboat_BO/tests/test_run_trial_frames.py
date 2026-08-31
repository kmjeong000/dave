from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.sailboat_BO.run_trial import RosTelemetryCollector
from experiments.sailboat_physics.frames import (
    frd_attitude_to_gazebo_euler_deg,
    frd_attitude_to_gazebo_quaternion,
)


def make_odometry(*, quaternion, x_m=0.0, y_m=0.0, vx_mps=0.0, vy_mps=0.0):
    x, y, z, w = quaternion
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x_m, y=y_m),
                orientation=SimpleNamespace(x=x, y=y, z=z, w=w),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=vx_mps, y=vy_mps),
            )
        ),
    )


@pytest.mark.parametrize(
    "frd_roll_deg,frd_pitch_deg,heading_deg",
    [
        (12.0, 0.0, 0.0),
        (-18.0, 4.0, 125.0),
        (23.0, -7.0, 275.0),
    ],
)
def test_odometry_roll_uses_physical_frd_frame(
    frd_roll_deg,
    frd_pitch_deg,
    heading_deg,
):
    quaternion = frd_attitude_to_gazebo_quaternion(
        frd_roll_deg,
        frd_pitch_deg,
        heading_deg,
    )
    collector = RosTelemetryCollector("sailboat", use_odom=True, use_imu=False)

    collector._on_odometry(make_odometry(quaternion=quaternion))

    snapshot = collector.snapshot()
    expected_raw_roll, expected_raw_pitch, _ = frd_attitude_to_gazebo_euler_deg(
        frd_roll_deg,
        frd_pitch_deg,
        heading_deg,
    )
    assert snapshot.odom_roll_deg == pytest.approx(frd_roll_deg)
    assert snapshot.odom_raw_roll_deg == pytest.approx(expected_raw_roll)
    assert snapshot.odom_raw_pitch_deg == pytest.approx(expected_raw_pitch)


def test_y_forward_hull_roll_is_not_raw_gazebo_x_roll():
    quaternion = frd_attitude_to_gazebo_quaternion(15.0, 0.0, 0.0)
    collector = RosTelemetryCollector("sailboat", use_odom=True, use_imu=False)

    collector._on_odometry(make_odometry(quaternion=quaternion))

    snapshot = collector.snapshot()
    assert snapshot.odom_roll_deg == pytest.approx(15.0)
    assert snapshot.odom_raw_roll_deg == pytest.approx(0.0)
    assert snapshot.odom_raw_pitch_deg == pytest.approx(15.0)
