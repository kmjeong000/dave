from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.sailboat_BO.common import load_yaml
from experiments.sailboat_performance.rudder_loaded import (
    ControlledRudderLoadConfig,
    parse_diagnostic_flow_activation_sim_time,
    render_rudder_loaded_model,
    summarize_loaded_rudder_wrench,
    summarize_loaded_tracking,
)
from experiments.sailboat_performance.run_rudder_tracking import (
    append_local_robot_description_file,
    make_yaw_origin_record,
    make_scenario,
    publish_rudder_test_flow_enabled,
    summarize_yaw_origin_records,
    validate_calm_environment,
    validate_config,
)
from experiments.sailboat_physics.rudder_response import RudderWrenchSample


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "experiments/sailboat_performance/config/rudder_loaded_1mps.yaml"
CONFIG_2MPS = REPO / "experiments/sailboat_performance/config/rudder_loaded_2mps.yaml"
MODEL = REPO / "models/dave_robot_models/description/sailboat/model.sdf"


def make_wrench_sample(sim_time_s: float = 10.0) -> RudderWrenchSample:
    return RudderWrenchSample(
        sim_time_s=sim_time_s,
        water_relative_x_mps=0.0,
        water_relative_y_mps=-1.0,
        water_relative_z_mps=0.0,
        speed_mps=1.0,
        alpha_deg=5.0,
        cl=0.07,
        cd=0.004,
        force_world_x_n=-10.0,
        force_world_y_n=-0.6,
        force_world_z_n=0.0,
        joint_angle_deg=5.0,
        command_deg=5.0,
        cp_world_x_m=0.0,
        cp_world_y_m=-1.5,
        cp_world_z_m=0.0,
        lever_world_x_m=0.0,
        lever_world_y_m=-1.5,
        lever_world_z_m=0.0,
        moment_world_x_nm=0.0,
        moment_world_y_nm=0.0,
        moment_world_z_nm=-15.0,
        yaw_axis_world_x=0.0,
        yaw_axis_world_y=0.0,
        yaw_axis_world_z=1.0,
        yaw_moment_nm=-15.0,
    )


def make_heading_samples(
    samples: list[RudderWrenchSample],
    headings_deg: list[float] | None = None,
) -> list[dict[str, float]]:
    headings = headings_deg or [0.0] * len(samples)
    return [
        {"sim_time_s": sample.sim_time_s, "heading_deg": heading}
        for sample, heading in zip(samples, headings, strict=True)
    ]


def test_repository_loaded_config_is_one_plus_five_degree_case():
    raw = load_yaml(CONFIG)
    tracking = validate_config(raw)
    load = ControlledRudderLoadConfig.from_mapping(raw["load"])

    assert raw["load"]["type"] == (
        "rudder_foil_runtime_gated_world_water_velocity"
    )
    assert raw["diagnostic"]["setpoints_deg"] == [5.0]
    assert tracking.setpoint_hold_s == pytest.approx(8.0)
    assert load.water_velocity_world_enu_mps == (0.0, -1.0, 0.0)
    assert load.diagnostic_flow_enable_topic == (
        "/model/sailboat/rudder_test_flow/enable"
    )
    assert load.target_speed_mps == pytest.approx(1.0)
    assert load.minimum_abs_yaw_moment_nm == pytest.approx(0.1)
    assert load.expected_yaw_moment_sign == -1
    assert load.maximum_sign_heading_deviation_deg == pytest.approx(15.0)
    assert load.wrench_sample_period_s == pytest.approx(0.1)
    assert make_scenario(raw)["id"] == "rudder_loaded_tracking_1mps"
    # The calm contract checks the production SDF before the trial-only copy.
    assert validate_calm_environment(REPO, raw)["foil_water_velocity_mps"] == [
        0.0,
        0.0,
        0.0,
    ]


def test_repository_two_mps_config_changes_only_the_load_condition(tmp_path):
    one_mps = load_yaml(CONFIG)
    two_mps = load_yaml(CONFIG_2MPS)
    tracking = validate_config(two_mps)
    load = ControlledRudderLoadConfig.from_mapping(two_mps["load"])

    assert two_mps["diagnostic"] == one_mps["diagnostic"]
    assert two_mps["world"] == one_mps["world"]
    assert two_mps["spawn"] == one_mps["spawn"]
    assert tracking.setpoint_hold_s == pytest.approx(8.0)
    assert load.water_velocity_world_enu_mps == (0.0, -2.0, 0.0)
    assert load.diagnostic_flow_enable_topic == (
        "/model/sailboat/rudder_test_flow/enable"
    )
    assert load.target_speed_mps == pytest.approx(2.0)
    assert load.minimum_abs_yaw_moment_nm == pytest.approx(0.1)
    assert load.expected_yaw_moment_sign == -1
    assert load.maximum_sign_heading_deviation_deg == pytest.approx(15.0)
    assert load.wrench_sample_period_s == pytest.approx(0.1)
    assert make_scenario(two_mps)["id"] == "rudder_loaded_tracking_2mps"
    assert validate_calm_environment(REPO, two_mps)["foil_water_velocity_mps"] == [
        0.0,
        0.0,
        0.0,
    ]

    output = tmp_path / "rudder_loaded_2mps_model.sdf"
    render_rudder_loaded_model(
        MODEL,
        output,
        load.water_velocity_world_enu_mps,
        load.wrench_sample_period_s,
    )
    rendered = output.read_text(encoding="utf-8")
    rudder_start = rendered.index("<daggerboard_link>rudder_link</daggerboard_link>")
    rudder_block = rendered[
        rudder_start : rendered.index("</plugin>", rudder_start)
    ]
    assert "<water_velocity>0 0 0</water_velocity>" in rudder_block
    assert (
        "<diagnostic_water_velocity>0 -2 0</diagnostic_water_velocity>"
        in rudder_block
    )
    assert (
        "<diagnostic_flow_enabled>false</diagnostic_flow_enabled>"
        in rudder_block
    )
    assert 'filename="libRudderDiagnosticLoadSystem.so"' in rendered
    assert "<debug_period>0.1</debug_period>" in rudder_block


def test_loaded_model_render_changes_only_rudder_foil(tmp_path):
    before = MODEL.read_text(encoding="utf-8")
    output = tmp_path / "rudder_loaded_model.sdf"

    contract = render_rudder_loaded_model(
        MODEL, output, (0.0, -1.0, 0.0), 0.1
    )
    rendered = output.read_text(encoding="utf-8")

    assert MODEL.read_text(encoding="utf-8") == before
    daggerboard_start = rendered.index(
        "<daggerboard_link>daggerboard_link</daggerboard_link>"
    )
    assert "<water_velocity>0 0 0</water_velocity>" in rendered[
        daggerboard_start : daggerboard_start + 300
    ]
    assert "<daggerboard_link>rudder_link</daggerboard_link>" in rendered
    rudder_start = rendered.index("<daggerboard_link>rudder_link</daggerboard_link>")
    rudder_block = rendered[
        rudder_start : rendered.index("</plugin>", rudder_start)
    ]
    assert "<water_velocity>0 0 0</water_velocity>" in rudder_block
    assert (
        "<diagnostic_water_velocity>0 -1 0</diagnostic_water_velocity>"
        in rudder_block
    )
    assert (
        "<diagnostic_flow_enable_topic>"
        "/model/sailboat/rudder_test_flow/enable"
        "</diagnostic_flow_enable_topic>"
    ) in rudder_block
    assert (
        "<diagnostic_flow_enabled>false</diagnostic_flow_enabled>"
        in rudder_block
    )
    assert contract["rudder_base_water_velocity_world_enu_mps"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert contract["rudder_diagnostic_water_velocity_world_enu_mps"] == [
        0.0,
        -1.0,
        0.0,
    ]
    assert not contract["diagnostic_flow_initially_enabled"]
    assert contract["diagnostic_plugin"] == (
        "ioes::sim::RudderDiagnosticLoadSystem"
    )
    assert "<debug_period>0.1</debug_period>" in rudder_block
    assert contract["rudder_wrench_sample_period_s"] == pytest.approx(0.1)
    assert not contract["production_model_modified"]


def test_flow_activation_parser_reports_first_real_transition():
    text = """
[RudderDiagnosticLoadSystem] flow command received: requested_enabled=true
[RudderDiagnosticLoadSystem] diagnostic flow transition: link=rudder_link, false -> true, simTimeS=31.25, active_water_velocity=0 -1 0
[RudderDiagnosticLoadSystem] diagnostic flow transition: link=rudder_link, false -> true, simTimeS=42.5, active_water_velocity=0 -1 0
"""

    assert parse_diagnostic_flow_activation_sim_time(text) == pytest.approx(
        31.25
    )
    assert parse_diagnostic_flow_activation_sim_time("no transition") is None


def test_flow_publish_uses_boolean_topic_and_retries(monkeypatch):
    calls = []

    def fake_execute(execution, command):
        calls.append((execution, command))
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(
        "experiments.sailboat_performance.run_rudder_tracking.execute_shell",
        fake_execute,
    )
    execution = object()
    publish_rudder_test_flow_enabled(
        execution,
        "/model/sailboat/rudder_test_flow/enable",
        True,
        repeats=2,
        interval_s=0.0,
    )

    assert len(calls) == 2
    assert all(item[0] is execution for item in calls)
    assert all("gz.msgs.Boolean" in item[1] for item in calls)
    assert all("data: true" in item[1] for item in calls)


def test_yaw_origin_record_preserves_timing_and_motion_telemetry():
    snapshot = SimpleNamespace(
        sim_time_s=12.5,
        heading_deg=354.0,
        derived_yaw_rate_deg_s=-2.5,
        gazebo_angular_yaw_rate_deg_s=-2.4,
        x_east_m=1.2,
        y_north_m=-0.7,
        speed_over_ground_mps=0.15,
        actual_rudder_angle_rad=0.1,
        odom_age_s=0.02,
        rudder_age_s=0.03,
    )

    record = make_yaw_origin_record(
        snapshot,
        phase="pre_measurement_settling",
        startup_hold_requested_held=False,
        diagnostic_flow_requested_enabled=False,
    )

    assert record["sim_time_s"] == pytest.approx(12.5)
    assert record["heading_deg"] == pytest.approx(354.0)
    assert record["derived_yaw_rate_deg_s"] == pytest.approx(-2.5)
    assert record["gazebo_angular_yaw_rate_deg_s"] == pytest.approx(-2.4)
    assert record["x_east_m"] == pytest.approx(1.2)
    assert record["y_north_m"] == pytest.approx(-0.7)
    assert record["speed_over_ground_mps"] == pytest.approx(0.15)
    assert record["actual_rudder_angle_deg"] == pytest.approx(5.72957795)
    assert not record["startup_hold_requested_held"]
    assert not record["diagnostic_flow_requested_enabled"]


def test_yaw_origin_summary_reports_phase_coverage_without_guessing_onset():
    summary = summarize_yaw_origin_records(
        [
            {"sim_time_s": 1.0, "phase": "arming_hold"},
            {"sim_time_s": 1.1, "phase": "arming_hold"},
            {"sim_time_s": 1.2, "phase": "pre_measurement_settling"},
        ]
    )

    assert summary["sample_count"] == 3
    assert summary["phase_sample_counts"] == {
        "arming_hold": 2,
        "pre_measurement_settling": 1,
    }
    assert summary["first_sim_time_s"] == pytest.approx(1.0)
    assert summary["last_sim_time_s"] == pytest.approx(1.2)
    assert not summary["automatic_yaw_onset_classification"]


def test_loaded_wrench_summary_requires_real_flow_force_and_expected_sign():
    config = ControlledRudderLoadConfig.from_mapping(load_yaml(CONFIG)["load"])
    samples = [make_wrench_sample(10.0 + index) for index in range(6)]

    summary = summarize_loaded_rudder_wrench(
        samples,
        heading_samples=make_heading_samples(samples),
        sign_reference_heading_deg=0.0,
        sign_reference_sim_time_s=9.9,
        target_rudder_deg=5.0,
        config=config,
    )

    assert summary["passed"]
    assert summary["relative_water_speed_mps"]["mean"] == pytest.approx(1.0)
    assert summary["force_magnitude_n"]["mean"] > 10.0
    assert summary["yaw_moment_nm"]["mean"] == pytest.approx(-15.0)
    assert summary["sign_evaluation_sample_count"] == 6
    assert summary["sign_evaluation_heading_deviation_deg"]["max"] == pytest.approx(
        0.0
    )
    assert summary["maximum_wrench_error_nm"] == pytest.approx(0.0)


def test_loaded_wrench_summary_rejects_missing_load_and_reversed_moment():
    config = ControlledRudderLoadConfig.from_mapping(load_yaml(CONFIG)["load"])
    samples = [
        RudderWrenchSample(
            **{
                **make_wrench_sample(10.0 + index).__dict__,
                "force_world_x_n": 0.0,
                "force_world_y_n": 0.0,
                "moment_world_z_nm": 0.0,
                "yaw_moment_nm": 0.0,
            }
        )
        for index in range(6)
    ]

    summary = summarize_loaded_rudder_wrench(
        samples,
        heading_samples=make_heading_samples(samples),
        sign_reference_heading_deg=0.0,
        sign_reference_sim_time_s=9.9,
        target_rudder_deg=5.0,
        config=config,
    )

    assert not summary["gates"]["nonzero_hydrodynamic_force"]
    assert not summary["gates"]["nonzero_yaw_moment"]
    assert not summary["gates"]["expected_yaw_moment_sign"]
    assert not summary["passed"]


def test_sign_gate_uses_only_initial_heading_envelope():
    config = ControlledRudderLoadConfig.from_mapping(load_yaml(CONFIG)["load"])
    samples = [make_wrench_sample(10.0 + index) for index in range(8)]
    for index in range(5, 8):
        samples[index] = RudderWrenchSample(
            **{
                **samples[index].__dict__,
                "force_world_x_n": 10.0,
                "moment_world_z_nm": 15.0,
                "yaw_moment_nm": 15.0,
            }
        )
    headings = make_heading_samples(
        samples,
        [359.0, 1.0, 5.0, 10.0, 14.0, 310.0, 305.0, 300.0],
    )

    summary = summarize_loaded_rudder_wrench(
        samples,
        heading_samples=headings,
        sign_reference_heading_deg=359.0,
        sign_reference_sim_time_s=9.9,
        target_rudder_deg=5.0,
        config=config,
    )

    assert summary["gates"]["minimum_sign_evaluation_samples"]
    assert summary["gates"]["expected_yaw_moment_sign"]
    assert summary["expected_yaw_moment_sign_fraction"] == pytest.approx(1.0)
    assert summary["sign_reference_heading_deg"] == pytest.approx(359.0)
    assert summary["sign_reference_capture_sim_time_s"] == pytest.approx(9.9)
    assert summary["sign_evaluation_sample_count"] == 5
    assert summary["sign_evaluation_heading_deviation_deg"]["min"] == pytest.approx(
        0.0
    )
    assert summary["sign_evaluation_heading_deviation_deg"]["max"] == pytest.approx(
        15.0
    )
    assert summary["sign_evaluation_all_matched_heading_deviation_deg"][
        "max"
    ] == pytest.approx(59.0)
    assert summary["passed"]


def test_sign_gate_fails_when_initial_heading_samples_are_insufficient():
    config = ControlledRudderLoadConfig.from_mapping(load_yaml(CONFIG)["load"])
    samples = [make_wrench_sample(10.0 + index) for index in range(7)]
    headings = make_heading_samples(
        samples,
        [0.0, 5.0, 10.0, 15.0, 50.0, 55.0, 60.0],
    )

    summary = summarize_loaded_rudder_wrench(
        samples,
        heading_samples=headings,
        sign_reference_heading_deg=0.0,
        sign_reference_sim_time_s=9.9,
        target_rudder_deg=5.0,
        config=config,
    )

    assert summary["sign_evaluation_sample_count"] == 4
    assert not summary["gates"]["minimum_sign_evaluation_samples"]
    assert not summary["gates"]["expected_yaw_moment_sign"]
    assert not summary["passed"]


def test_sign_gate_does_not_use_first_wrench_heading_as_reference():
    config = ControlledRudderLoadConfig.from_mapping(load_yaml(CONFIG)["load"])
    samples = [make_wrench_sample(10.0 + index) for index in range(6)]
    headings = make_heading_samples(
        samples,
        [307.0, 309.0, 311.0, 313.0, 315.0, 316.0],
    )

    summary = summarize_loaded_rudder_wrench(
        samples,
        heading_samples=headings,
        sign_reference_heading_deg=0.0,
        sign_reference_sim_time_s=9.9,
        target_rudder_deg=5.0,
        config=config,
    )

    assert summary["sign_reference_heading_deg"] == pytest.approx(0.0)
    assert summary["sign_evaluation_heading_valid_sample_count"] == 0
    assert summary["sign_evaluation_sample_count"] == 0
    assert not summary["gates"]["minimum_sign_evaluation_samples"]
    assert not summary["gates"]["expected_yaw_moment_sign"]
    assert not summary["passed"]


def test_loaded_tracking_reuses_actuator_gates_but_not_unloaded_speed():
    gates = {
        "telemetry": True,
        "single_owner_command_delivered": True,
        "unloaded_speed": False,
        "steady_tracking_error": True,
        "overshoot": True,
        "rise_reached": True,
        "settled": True,
        "velocity_limit_fraction": True,
    }
    summary = summarize_loaded_tracking(
        {"steps": [{"target_rudder_deg": 5.0, "gates": gates}]}
    )

    assert summary["excluded_unloaded_only_gate"] == "unloaded_speed"
    assert summary["passed"]


@dataclass(frozen=True)
class FakeLaunchPlan:
    backend: str
    launch_cmd: list[str]
    process_cmd: list[str]
    display_cmd: str
    cleanup_token: str | None = None


def test_trial_model_argument_is_appended_only_to_local_launch():
    original = FakeLaunchPlan(
        backend="local",
        launch_cmd=["ros2", "launch", "demo.py"],
        process_cmd=["bash", "-lc", "ros2 launch demo.py"],
        display_cmd="ros2 launch demo.py",
    )
    updated = append_local_robot_description_file(
        original, Path("/tmp/trial/rudder_loaded_model.sdf")
    )

    assert original.launch_cmd == ["ros2", "launch", "demo.py"]
    assert updated.launch_cmd[-1] == (
        "robot_description_file:=/tmp/trial/rudder_loaded_model.sdf"
    )
    assert updated.process_cmd[-1].endswith(
        "robot_description_file:=/tmp/trial/rudder_loaded_model.sdf"
    )


def test_launch_defaults_preserve_installed_production_model():
    outer = (REPO / "examples/dave_demos/launch/dave_robot.launch.py").read_text(
        encoding="utf-8"
    )
    upload = (
        REPO / "models/dave_robot_models/launch/upload_robot.launch.py"
    ).read_text(encoding="utf-8")

    assert '"robot_description_file": robot_description_file' in outer
    assert '"robot_description_file"' in upload
    assert 'FindPackageShare("dave_robot_models")' in upload
    assert '"description"' in upload
    assert '"model.sdf"' in upload
    assert '"-file",\n            robot_description_file' in upload
