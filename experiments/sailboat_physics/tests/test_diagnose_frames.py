from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from experiments.sailboat_physics.diagnose_frames import (
    build_cases,
    make_case_config,
    safe_label,
    set_model_pose,
)


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
        "spawn": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.2, "yaw_deg": 0.0},
        "mission": {"frame": "gazebo_xy_m", "waypoints": [[60.0, 0.0]]},
    }


def test_build_cases_includes_cardinal_roll_and_pitch_checks():
    cases = build_cases(
        headings=[0.0, 90.0, 180.0, 270.0],
        roll_tests=[-10.0, 10.0],
        pitch_tests=[-10.0, 10.0],
    )

    assert len(cases) == 8
    assert [case.expected.heading_deg for case in cases[:4]] == [0.0, 90.0, 180.0, 270.0]
    assert [case.expected.roll_deg for case in cases[4:6]] == [-10.0, 10.0]
    assert [case.expected.pitch_deg for case in cases[6:]] == [-10.0, 10.0]


def test_case_config_disables_wind_and_maps_frd_roll_to_gazebo_pitch():
    case = build_cases(headings=[], roll_tests=[10.0], pitch_tests=[])[0]

    config, scenario = make_case_config(
        base_config(),
        base_scenario(),
        case,
        sitl_start_delay_s=0.0,
    )

    assert scenario["split"] == "diagnostic"
    assert scenario["world"]["wind_world_xyz_mps"] == [0.0, 0.0, 0.0]
    assert scenario["spawn"]["heading_deg"] == 0.0
    assert scenario["spawn"]["gazebo_roll_deg"] == pytest.approx(0.0)
    assert scenario["spawn"]["gazebo_pitch_deg"] == pytest.approx(10.0)
    assert config["study"]["gui"] is False
    assert config["study"]["headless"] is True
    assert config["study"]["use_mavproxy"] is False
    assert config["study"]["use_mavros"] is False
    assert config["study"]["sitl_start_delay_s"] == 0.0


def test_set_model_pose_uses_world_service_and_complete_quaternion():
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="data: true\n",
        stderr="",
    )
    with patch(
        "experiments.sailboat_physics.diagnose_frames.subprocess.run",
        return_value=completed,
    ) as run:
        set_model_pose(
            world_name="waves",
            model_name="sailboat",
            position_xyz_m=(1.0, 2.0, 0.2),
            quaternion_xyzw=(0.1, 0.2, 0.3, 0.9),
            timeout_ms=3000,
        )

    command = run.call_args.args[0]
    assert command[:4] == ["gz", "service", "-s", "/world/waves/set_pose"]
    request = command[command.index("--req") + 1]
    assert 'name: "sailboat"' in request
    assert "x: 1" in request
    assert "y: 2" in request
    assert "z: 0.2" in request
    assert "w: 0.9" in request


def test_set_model_pose_rejects_false_response():
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="data: false\n",
        stderr="",
    )
    with patch(
        "experiments.sailboat_physics.diagnose_frames.subprocess.run",
        return_value=completed,
    ):
        with pytest.raises(RuntimeError, match="set_pose failed"):
            set_model_pose(
                world_name="waves",
                model_name="sailboat",
                position_xyz_m=(0.0, 0.0, 0.2),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                timeout_ms=3000,
            )


def test_safe_label_removes_path_separators_and_spaces():
    assert safe_label("pre fix/one") == "pre_fix_one"
