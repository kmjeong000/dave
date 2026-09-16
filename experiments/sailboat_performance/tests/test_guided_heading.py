from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.sailboat_performance.guided_heading import (
    GuidedHeadingTarget,
    guided_velocity_yaw_type_mask,
    local_ned_velocity,
    parse_ardupilot_parameter_file,
    validate_wind_only_sdf,
    validate_wind_only_servo_functions,
)


class FakeMav:
    def __init__(self):
        self.calls = []

    def set_position_target_local_ned_send(self, *args):
        self.calls.append(args)


MAVLINK = SimpleNamespace(
    MAV_FRAME_LOCAL_NED=1,
    POSITION_TARGET_TYPEMASK_X_IGNORE=1,
    POSITION_TARGET_TYPEMASK_Y_IGNORE=2,
    POSITION_TARGET_TYPEMASK_Z_IGNORE=4,
    POSITION_TARGET_TYPEMASK_VZ_IGNORE=32,
    POSITION_TARGET_TYPEMASK_AX_IGNORE=64,
    POSITION_TARGET_TYPEMASK_AY_IGNORE=128,
    POSITION_TARGET_TYPEMASK_AZ_IGNORE=256,
    POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE=2048,
)


def test_fixed_heading_target_uses_velocity_xy_and_yaw_only():
    master = SimpleNamespace(mav=FakeMav(), target_system=1, target_component=1)
    target = GuidedHeadingTarget(master, MAVLINK, 90.0, 1.0, refresh_hz=2.0)
    assert guided_velocity_yaw_type_mask(MAVLINK) == 2535
    assert local_ned_velocity(90.0, 1.0) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert target.send_if_due(10.0)
    assert not target.send_if_due(10.49)
    assert target.send_if_due(10.50)
    first = master.mav.calls[0]
    assert first[4] == 2535
    assert first[8:11] == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert first[14] == pytest.approx(1.57079632679)


def test_wind_only_contract_matches_repository_configuration():
    repo = Path(__file__).resolve().parents[3]
    values = parse_ardupilot_parameter_file(
        repo / "models/dave_robot_models/config/sailboat/ardurover.parm"
    )
    validate_wind_only_servo_functions(values)
    validate_wind_only_sdf(
        repo / "models/dave_robot_models/description/sailboat/model.sdf"
    )


def test_motor_function_is_rejected():
    with pytest.raises(ValueError, match="motor"):
        validate_wind_only_servo_functions(
            {
                "SERVO1_FUNCTION": 26,
                "SERVO2_FUNCTION": 89,
                "SERVO3_FUNCTION": 70,
            }
        )


def test_extra_ardupilot_control_mapping_is_rejected(tmp_path):
    sdf = tmp_path / "model.sdf"
    sdf.write_text(
        """<sdf><model><plugin><control><jointName>rudder_joint</jointName></control>
        <control><jointName>sail_joint</jointName></control>
        <control><jointName>propeller_joint</jointName></control></plugin></model></sdf>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_wind_only_sdf(sdf)
