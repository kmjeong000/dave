"""Physical-frame diagnostics for the DAVE sailboat."""

from .frames import (
    AttitudeDeg,
    FrameSample,
    attitude_error_deg,
    frd_attitude_to_gazebo_euler_deg,
    frd_attitude_to_gazebo_quaternion,
    gazebo_quaternion_to_frd_attitude_deg,
    summarize_frame_case,
)

__all__ = [
    "AttitudeDeg",
    "FrameSample",
    "attitude_error_deg",
    "frd_attitude_to_gazebo_euler_deg",
    "frd_attitude_to_gazebo_quaternion",
    "gazebo_quaternion_to_frd_attitude_deg",
    "summarize_frame_case",
]
