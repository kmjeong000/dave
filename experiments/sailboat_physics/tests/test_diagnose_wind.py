from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.sailboat_physics.diagnose_wind import (
    build_cases,
    make_case_config,
    parse_gz_vector3,
    parse_sail_plugin_wind,
    read_anemometer_once,
    select_anemometer_topic,
    wait_for_wind_settle,
)
from experiments.sailboat_physics.wind import SailPluginWindSample, Vector3


def base_config():
    return {
        "study": {
            "gui": True,
            "headless": False,
            "paused": False,
            "use_mavproxy": True,
            "use_mavros": True,
            "sitl_start_delay_s": 10.0,
        }
    }


def base_scenario():
    return {
        "id": "train_crosswind_straight",
        "split": "train",
        "world": {
            "template": "waves_wind",
            "internal_world_name": "waves",
            "wind_world_xyz_mps": [0.0, 8.0, 0.0],
        },
        "spawn": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.2, "heading_deg": 0.0},
        "mission": {"frame": "gazebo_xy_m", "waypoints": [[60.0, 0.0]]},
    }


def test_build_cases_covers_cardinal_rotated_and_dynamic_contracts():
    cases = build_cases(8.0)

    assert len(cases) == 6
    assert [case.name for case in cases[:4]] == [
        "north_to_heading_000",
        "east_to_heading_000",
        "south_to_heading_000",
        "west_to_heading_000",
    ]
    assert cases[4].heading_deg == 90.0
    assert cases[4].dynamic is False
    assert cases[5].dynamic is True


def test_static_case_config_removes_world_variation_and_keeps_force_unarmed():
    case = build_cases(8.0)[1]

    config, scenario = make_case_config(
        base_config(),
        base_scenario(),
        case,
        sitl_start_delay_s=0.0,
    )

    assert scenario["split"] == "diagnostic"
    assert scenario["world"]["wind_world_xyz_mps"] == [8.0, 0.0, 0.0]
    assert scenario["world"]["wind_mag_noise_stddev"] == 0.0
    assert scenario["world"]["wind_dir_noise_stddev"] == 0.0
    assert scenario["world"]["wind_mag_sin_amp_percent"] == 0.0
    assert scenario["world"]["wind_dir_sin_amp_deg"] == 0.0
    assert config["study"]["gui"] is False
    assert config["study"]["use_mavproxy"] is False
    assert config["study"]["use_mavros"] is False


def test_dynamic_case_config_introduces_only_magnitude_variation():
    case = build_cases(8.0)[-1]

    _, scenario = make_case_config(
        base_config(),
        base_scenario(),
        case,
        sitl_start_delay_s=10.0,
    )

    assert scenario["world"]["wind_mag_sin_amp_percent"] == 0.5
    assert scenario["world"]["wind_dir_sin_amp_deg"] == 0.0
    assert scenario["world"]["wind_mag_noise_stddev"] == 0.0
    assert scenario["world"]["wind_dir_noise_stddev"] == 0.0


def test_parse_gz_vector3_extracts_timestamp_and_vector():
    text = """
header {
  stamp {
    sec: 12
    nsec: 250000000
  }
}
x: -1.5
y: 8
z: 0.25
"""

    sim_time_s, vector = parse_gz_vector3(text)

    assert sim_time_s == pytest.approx(12.25)
    assert vector.x == pytest.approx(-1.5)
    assert vector.y == pytest.approx(8.0)
    assert vector.z == pytest.approx(0.25)


def test_select_anemometer_topic_prefers_sailboat_scoped_topic():
    topic = select_anemometer_topic(
        [
            "/model/other/link/anemometer/anemometer",
            "/world/waves/model/sailboat/link/anemometer_link/sensor/anemometer/anemometer",
        ],
        "sailboat",
    )

    assert "sailboat" in topic
    assert topic.endswith("/anemometer")


def test_read_anemometer_once_uses_one_message_gz_command():
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="x: 0\ny: 8\nz: 0\n",
        stderr="",
    )
    with patch(
        "experiments.sailboat_physics.diagnose_wind.subprocess.run",
        return_value=completed,
    ) as run:
        _, vector = read_anemometer_once("/anemometer", 3.0)

    assert run.call_args.args[0] == [
        "gz",
        "topic",
        "-e",
        "-t",
        "/anemometer",
        "-n",
        "1",
    ]
    assert vector.y == 8.0


def sail_plugin_sample(
    sim_time_s: float,
    wind_y_mps: float,
    *,
    source: str = "world_component",
) -> SailPluginWindSample:
    return SailPluginWindSample(
        sim_time_s=sim_time_s,
        source=source,
        force_scale=0.0,
        wind_x_mps=0.0,
        wind_y_mps=wind_y_mps,
        wind_z_mps=0.0,
        apparent_x_mps=0.0,
        apparent_y_mps=wind_y_mps,
        apparent_z_mps=0.0,
        aerodynamic_speed_mps=wind_y_mps,
        alpha_deg=0.0,
        lift_coefficient=0.0,
        drag_coefficient=0.0,
    )


def test_wait_for_wind_settle_requires_continuous_live_component_samples():
    with patch(
        "experiments.sailboat_physics.diagnose_wind.read_sail_plugin_wind",
        side_effect=[
            [sail_plugin_sample(26.0, 7.4)],
            [sail_plugin_sample(28.0, 7.6)],
            [sail_plugin_sample(29.0, 7.7)],
            [sail_plugin_sample(30.1, 7.8)],
        ],
    ) as read, patch("experiments.sailboat_physics.diagnose_wind.time.sleep"):
        settled_at = wait_for_wind_settle(
            [Path("/tmp/launch.stderr.log")],
            expected_world=Vector3(0.0, 8.0, 0.0),
            tolerance_mps=0.5,
            hold_sim_time_s=2.0,
            timeout_s=5.0,
            poll_interval_s=0.01,
        )

    assert settled_at == pytest.approx(30.1)
    assert read.call_count == 4


def test_wait_for_wind_settle_resets_streak_after_out_of_tolerance_sample():
    with patch(
        "experiments.sailboat_physics.diagnose_wind.read_sail_plugin_wind",
        side_effect=[
            [sail_plugin_sample(28.0, 7.6)],
            [sail_plugin_sample(29.0, 7.4)],
            [sail_plugin_sample(30.0, 7.6)],
            [sail_plugin_sample(32.1, 7.7)],
        ],
    ), patch("experiments.sailboat_physics.diagnose_wind.time.sleep"):
        settled_at = wait_for_wind_settle(
            [Path("/tmp/launch.stderr.log")],
            expected_world=Vector3(0.0, 8.0, 0.0),
            tolerance_mps=0.5,
            hold_sim_time_s=2.0,
            timeout_s=5.0,
            poll_interval_s=0.01,
        )

    assert settled_at == pytest.approx(32.1)


def test_wait_for_wind_settle_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="tolerance"):
        wait_for_wind_settle(
            [],
            expected_world=Vector3(0.0, 8.0, 0.0),
            tolerance_mps=-0.1,
            hold_sim_time_s=2.0,
            timeout_s=5.0,
            poll_interval_s=0.01,
        )


def test_parse_sail_plugin_wind_accepts_new_gated_and_active_lines():
    text = """
[gazebo-1] \x1b[1;31m[SailLiftDragSystem] link=\x1b[0msail_link simTimeS=\x1b[1;31m12.5\x1b[0m windSource=world_component forceScale=0 windWorld=0 8 0 Vapp=0 7.75 0 speed=7.75 alpha_deg=30 cl=0.8 cd=0.2
[SailLiftDragSystem] link=sail_link simTimeS=13.5 windSource=world_component forceScale=0.1 forceLimitScale=1 rawForceN=4 windWorld=1.0e-1 7.5 -0.2 Vapp=0.1 7.25 -0.2 speed=7.2 alpha_deg=25 cl=0.7 cd=0.15
"""

    records = parse_sail_plugin_wind(text)

    assert len(records) == 2
    assert records[0].source == "world_component"
    assert records[0].force_scale == 0.0
    assert records[0].wind_y_mps == 8.0
    assert records[0].apparent_y_mps == pytest.approx(7.75)
    assert records[0].aerodynamic_speed_mps == pytest.approx(7.75)
    assert records[0].alpha_deg == pytest.approx(30.0)
    assert records[0].lift_coefficient == pytest.approx(0.8)
    assert records[0].drag_coefficient == pytest.approx(0.2)
    assert records[1].force_scale == pytest.approx(0.1)
    assert records[1].wind_x_mps == pytest.approx(0.1)
    assert records[1].wind_z_mps == pytest.approx(-0.2)
    assert records[1].apparent_x_mps == pytest.approx(0.1)
    assert records[1].apparent_z_mps == pytest.approx(-0.2)
