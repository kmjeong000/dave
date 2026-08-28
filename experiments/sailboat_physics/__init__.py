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
from .wind import (
    SailPluginWindSample,
    Vector3,
    WindSample,
    summarize_wind_case,
    wind_from_compass_deg,
    wind_to_compass_deg,
    world_enu_to_model_sensor,
)

__all__ = [
    "AttitudeDeg",
    "FrameSample",
    "attitude_error_deg",
    "frd_attitude_to_gazebo_euler_deg",
    "frd_attitude_to_gazebo_quaternion",
    "gazebo_quaternion_to_frd_attitude_deg",
    "summarize_frame_case",
    "SailPluginWindSample",
    "Vector3",
    "WindSample",
    "summarize_wind_case",
    "wind_from_compass_deg",
    "wind_to_compass_deg",
    "world_enu_to_model_sensor",
]
