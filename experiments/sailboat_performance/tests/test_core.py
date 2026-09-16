import math
from pathlib import Path

import pytest

from experiments.sailboat_BO.common import load_yaml
from experiments.sailboat_performance.core import (
    Vector2,
    angle_error_deg,
    course_over_ground_deg,
    heading_rate_deg_s,
    model_velocity_to_world_enu,
    signed_twa_deg,
    vmg_toward_bearing_mps,
    wind_from_bearing_deg,
    wind_world_enu_from_condition,
)
from experiments.sailboat_performance.run_polar_trial import (
    validate_performance_config,
)


def test_wrap_heading_and_signed_twa_contract():
    assert angle_error_deg(1.0, 359.0) == pytest.approx(2.0)
    assert signed_twa_deg(90.0, 0.0) == pytest.approx(90.0)
    assert signed_twa_deg(270.0, 0.0) == pytest.approx(-90.0)
    assert signed_twa_deg(0.0, 180.0) == pytest.approx(180.0)


@pytest.mark.parametrize(
    ("measured_heading_deg", "target_heading_deg", "expected_error_deg"),
    [
        (10.0, 0.0, -10.0),
        (-10.0, 0.0, 10.0),
        (359.0, 0.0, 1.0),
        (0.0, 359.0, -1.0),
        (1.0, 0.0, -1.0),
    ],
)
def test_heading_error_uses_target_minus_measured_contract(
    measured_heading_deg, target_heading_deg, expected_error_deg
):
    assert angle_error_deg(target_heading_deg, measured_heading_deg) == pytest.approx(
        expected_error_deg
    )


def test_heading_rate_uses_wrapped_compass_heading_difference():
    assert heading_rate_deg_s(359.0, 1.0, 0.5) == pytest.approx(4.0)
    assert heading_rate_deg_s(1.0, 359.0, 0.5) == pytest.approx(-4.0)
    with pytest.raises(ValueError):
        heading_rate_deg_s(0.0, 1.0, 0.0)


def test_polar_condition_converts_wind_from_to_world_enu():
    wind = wind_world_enu_from_condition(8.0, 0.0, 90.0)
    assert wind.east == pytest.approx(-8.0)
    assert wind.north == pytest.approx(0.0, abs=1e-12)
    assert wind_from_bearing_deg(wind) == pytest.approx(90.0)


def test_vmg_and_course_use_world_enu_with_compass_bearings():
    assert vmg_toward_bearing_mps(1.0, 2.0, 0.0) == pytest.approx(2.0)
    assert vmg_toward_bearing_mps(1.0, 2.0, 90.0) == pytest.approx(1.0)
    assert course_over_ground_deg(1.0, 0.0) == pytest.approx(90.0)
    assert course_over_ground_deg(0.0, 0.0) is None


def test_body_twist_rotates_to_world_enu():
    identity = (0.0, 0.0, 0.0, 1.0)
    assert model_velocity_to_world_enu(0.0, 2.0, 0.0, identity) == pytest.approx(
        (0.0, 2.0, 0.0)
    )
    yaw_90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    assert model_velocity_to_world_enu(0.0, 2.0, 0.0, yaw_90) == pytest.approx(
        (-2.0, 0.0, 0.0), abs=1e-12
    )


def valid_config():
    return {
        "study": {},
        "condition": {
            "tws_mps": 8.0,
            "target_heading_deg": 0.0,
            "signed_twa_deg": 90.0,
        },
        "world": {},
        "spawn": {},
        "performance": {
            "desired_speed_mps": 1.0,
            "target_refresh_hz": 2.0,
            "sample_hz": 10.0,
            "max_sim_duration_s": 180.0,
            "measurement_duration_s": 30.0,
            "minimum_measurement_samples": 285,
            "steady_state": {},
        },
    }


def test_repository_polar_smoke_config_is_valid():
    repo = Path(__file__).resolve().parents[3]
    config = load_yaml(
        repo / "experiments/sailboat_performance/config/polar_smoke.yaml"
    )
    assert validate_performance_config(config) is config


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("condition", "tws_mps"), 0.0),
        (("condition", "signed_twa_deg"), 181.0),
        (("performance", "sample_hz"), 0.0),
        (("performance", "minimum_measurement_samples"), 1),
    ],
)
def test_invalid_performance_config_is_rejected(path, value):
    config = valid_config()
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_performance_config(config)
