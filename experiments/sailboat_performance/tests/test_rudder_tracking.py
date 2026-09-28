import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest

from experiments.sailboat_BO.common import load_yaml
from experiments.sailboat_performance.rudder_tracking import (
    PreMeasurementSettlingState,
    RudderTrackingConfig,
    pre_measurement_settle_timed_out,
    summarize_tracking_run,
    summarize_tracking_step,
    validate_setpoints_deg,
)
from experiments.sailboat_performance.run_rudder_tracking import (
    PreMeasurementSettleTimeout,
    make_record,
    make_scenario,
    pre_measurement_settle_timeout_outcome,
    serialize_ros_parameter_value,
    set_rudder_override,
    validate_calm_environment,
    validate_config,
)
from experiments.sailboat_performance.telemetry import RosPerformanceSnapshot


def make_records(
    *,
    target_deg=5.0,
    start_deg=0.0,
    response=None,
    final_command=None,
    velocity=None,
    speed=0.01,
    valid=True,
):
    target = math.radians(target_deg)
    start = math.radians(start_deg)
    records = []
    for index in range(161):
        elapsed = index * 0.05
        progress = min(1.0, elapsed / 0.5)
        actual = start + progress * (target - start)
        if response is not None:
            actual = math.radians(float(response(elapsed)))
        command = target
        if final_command is not None:
            command = math.radians(float(final_command(elapsed)))
        elif index == 0:
            command = start
        actual_velocity = (
            float(velocity(elapsed))
            if velocity is not None
            else (target - start) / 0.5 if elapsed < 0.5 else 0.0
        )
        records.append(
            {
                "step_index": 0,
                "elapsed_step_s": elapsed,
                "actual_rudder_angle_rad": actual,
                "actual_rudder_velocity_rad_s": actual_velocity,
                "final_rudder_rad": command,
                "speed_over_ground_mps": speed,
                "telemetry_valid": valid,
            }
        )
    return records


def test_repository_unloaded_config_and_setpoint_contract():
    repo = Path(__file__).resolve().parents[3]
    raw = load_yaml(
        repo / "experiments/sailboat_performance/config/rudder_unloaded.yaml"
    )
    config = validate_config(raw)

    assert validate_setpoints_deg(raw["diagnostic"]["setpoints_deg"]) == (
        0.0,
        5.0,
        -5.0,
        10.0,
        -10.0,
        0.0,
    )
    assert config.setpoint_hold_s == pytest.approx(8.0)
    assert config.steady_window_s == pytest.approx(2.0)
    assert config.joint_velocity_limit_rad_s == pytest.approx(3.0)
    assert config.pre_measurement_settle_required_s == pytest.approx(2.0)
    assert config.pre_measurement_settle_timeout_s == pytest.approx(30.0)
    assert raw["world"]["template"] == "rudder_calm"
    assert make_scenario(raw)["world"]["wind_world_xyz_mps"] == [0.0, 0.0, 0.0]


def test_repository_calm_environment_contract():
    repo = Path(__file__).resolve().parents[3]
    raw = load_yaml(
        repo / "experiments/sailboat_performance/config/rudder_unloaded.yaml"
    )

    contract = validate_calm_environment(repo, raw)

    assert contract["world_template"] == "rudder_calm"
    assert contract["wave_model_uri"] == "model://waves_low"
    assert contract["wind_world_enu_mps"] == [0.0, 0.0, 0.0]
    assert not contract["wave_forcing_enabled"]
    assert contract["ocean_current_enu_mps"] == [0.0, 0.0, 0.0]
    assert contract["foil_water_velocity_mps"] == [0.0, 0.0, 0.0]


def test_calm_environment_rejects_nonzero_rendered_wind_forcing():
    repo = Path(__file__).resolve().parents[3]
    raw = load_yaml(
        repo / "experiments/sailboat_performance/config/rudder_unloaded.yaml"
    )
    raw["world"]["wind_mag_noise_stddev"] = 0.01

    with pytest.raises(
        ValueError, match="zero configured wind magnitude noise"
    ):
        validate_calm_environment(repo, raw)


def test_calm_world_is_isolated_from_production_wave_world():
    repo = Path(__file__).resolve().parents[3]
    worlds = repo / "models/dave_worlds/worlds"

    calm_root = ET.parse(worlds / "rudder_calm.world").getroot()
    production_root = ET.parse(worlds / "waves_wind.world").getroot()

    assert calm_root.findtext(".//include/uri") == "model://waves_low"
    assert production_root.findtext(".//include/uri") == "model://waves"

    wave_model = ET.parse(
        repo / "models/dave_world_models/waves/model.sdf"
    ).getroot()
    assert any(float(node.text) > 0.0 for node in wave_model.iter("wind_speed"))
    assert any(float(node.text) > 0.0 for node in wave_model.iter("steepness"))


def test_rendered_calm_world_keeps_all_environment_forcing_zero(tmp_path):
    from experiments.sailboat_BO.run_trial import render_trial_world

    repo = Path(__file__).resolve().parents[3]
    raw = load_yaml(
        repo / "experiments/sailboat_performance/config/rudder_unloaded.yaml"
    )
    scenario = make_scenario(raw)
    output = tmp_path / "generated_calm.world"
    context = SimpleNamespace(
        scenario_cfg=scenario,
        world_template_dir=repo / "models/dave_worlds/worlds",
        workspace_root=tmp_path / "workspace",
        files=SimpleNamespace(world_file=output),
    )

    assert render_trial_world(context) == "generated_calm"
    root = ET.parse(output).getroot()
    assert root.findtext(".//wind/linear_velocity") == "0.0 0.0 0.0"
    assert root.findtext(".//include/uri") == "model://waves_low"
    assert float(
        root.findtext(
            ".//plugin[@filename='gz-sim-wind-effects-system']"
            "/horizontal/magnitude/noise/stddev"
        )
    ) == 0.0
    mirrored = (
        tmp_path
        / "workspace/install/dave_worlds/share/dave_worlds/worlds"
        / "generated_calm.world"
    )
    assert mirrored.exists()


def test_config_rejects_large_setpoints_and_invalid_windows():
    with pytest.raises(ValueError, match=r"within \+/-10 deg"):
        validate_setpoints_deg([0.0, 10.1])
    with pytest.raises(ValueError, match="must not exceed"):
        RudderTrackingConfig.from_mapping(
            {"setpoint_hold_s": 1.0, "steady_window_s": 2.0}
        )
    with pytest.raises(ValueError, match="pre_measurement_settle_timeout_s"):
        RudderTrackingConfig.from_mapping(
            {"pre_measurement_settle_timeout_s": 0.0}
        )


def observe_settling_sequence(state, samples, *, neutral_rudder_owned=True):
    ready = []
    for sim_time_s, speed_mps in samples:
        ready.append(
            state.observe(
                sim_time_s=sim_time_s,
                speed_mps=speed_mps,
                telemetry_valid=True,
                neutral_rudder_owned=neutral_rudder_owned,
            )
        )
    return ready


def test_pre_measurement_settling_requires_two_continuous_sim_seconds():
    state = PreMeasurementSettlingState(0.25, 2.0)
    ready = observe_settling_sequence(
        state,
        [
            (0.0, 0.7),
            (0.5, 0.5),
            (1.0, 0.3),
            (1.5, 0.24),
            (2.0, 0.20),
            (2.5, 0.18),
            (3.0, 0.18),
            (3.5, 0.18),
        ],
    )

    assert ready[:-1] == [False] * 7
    assert ready[-1]
    assert state.consecutive_s == pytest.approx(2.0)
    assert state.to_summary(timeout_s=30.0, wall_duration_s=4.0)[
        "reached"
    ]


def test_pre_measurement_settling_resets_after_threshold_crossing():
    state = PreMeasurementSettlingState(0.25, 2.0)
    ready = observe_settling_sequence(
        state,
        [
            (0.0, 0.24),
            (1.0, 0.27),
            (1.5, 0.20),
            (3.4, 0.20),
            (3.5, 0.20),
        ],
    )

    assert ready == [False, False, False, False, True]
    assert state.below_threshold_since_sim_time_s == pytest.approx(1.5)


def test_pre_measurement_settling_timeout_is_independent_wall_clock_gate():
    state = PreMeasurementSettlingState(0.25, 2.0)
    observe_settling_sequence(
        state,
        [(0.0, 0.7), (1.0, 0.5), (2.0, 0.3)],
    )

    assert not state.reached
    assert not pre_measurement_settle_timed_out(
        wall_elapsed_s=29.9, timeout_s=30.0, reached=state.reached
    )
    assert pre_measurement_settle_timed_out(
        wall_elapsed_s=30.0, timeout_s=30.0, reached=state.reached
    )
    outcome = pre_measurement_settle_timeout_outcome(
        PreMeasurementSettleTimeout("not settled")
    )
    assert outcome["status"] == "failure"
    assert outcome["failure_reason"] == "pre_measurement_settle_timeout"


def test_pre_measurement_settling_samples_do_not_enter_step_zero_metrics():
    state = PreMeasurementSettlingState(0.25, 2.0)
    observe_settling_sequence(
        state,
        [(0.0, 0.7), (1.0, 0.2), (2.0, 0.2), (3.0, 0.2)],
    )
    measurement_records = make_records(target_deg=0.0, speed=0.01)
    summary = summarize_tracking_run(
        measurement_records,
        starts_by_step={0: 0.0},
        targets_by_step={0: 0.0},
        config=RudderTrackingConfig(),
    )

    assert state.max_speed_mps == pytest.approx(0.7)
    assert summary["steps"][0]["speed_over_ground_mps"]["max"] == pytest.approx(
        0.01
    )


def test_pre_measurement_settling_requires_neutral_diagnostic_ownership():
    state = PreMeasurementSettlingState(0.25, 2.0)
    observe_settling_sequence(
        state,
        [(0.0, 0.1), (2.0, 0.1)],
        neutral_rudder_owned=False,
    )
    assert not state.reached
    assert state.consecutive_s == pytest.approx(0.0)

    ready = observe_settling_sequence(
        state,
        [(2.1, 0.1), (4.1, 0.1)],
        neutral_rudder_owned=True,
    )
    assert ready == [False, True]
    assert state.neutral_rudder_owned_at_ready


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0.0"),
        (5.0, "5.0"),
        (-5.0, "-5.0"),
        (0.08726646259971647, "0.08726646259971647"),
        (True, "true"),
        (False, "false"),
    ],
)
def test_ros_parameter_serialization_preserves_double_and_bool_types(
    value, expected
):
    assert serialize_ros_parameter_value(value) == expected


def test_rudder_override_builds_double_and_bool_ros_parameter_commands():
    completed = SimpleNamespace(
        returncode=0,
        stdout="Set parameter successful\n",
        stderr="",
    )
    with patch(
        "experiments.sailboat_performance.run_rudder_tracking.execute_shell",
        return_value=completed,
    ) as execute:
        set_rudder_override(
            object(),
            "/sailboat_command_adapter",
            command_rad=0.0,
            enabled=True,
        )

    commands = [call.args[1] for call in execute.call_args_list]
    assert commands == [
        (
            "ros2 param set /sailboat_command_adapter "
            "diagnostic_rudder_command_rad 0.0"
        ),
        (
            "ros2 param set /sailboat_command_adapter "
            "diagnostic_rudder_override_enabled true"
        ),
    ]


def test_good_response_passes_and_ignores_initial_command_delivery_transient():
    config = RudderTrackingConfig()
    summary = summarize_tracking_step(
        make_records(),
        start_angle_rad=0.0,
        target_rad=math.radians(5.0),
        config=config,
    )

    assert summary["passed"]
    assert summary["rise_time_s"] == pytest.approx(0.45)
    assert summary["settling_time_s"] == pytest.approx(0.4)
    assert summary["steady_tracking_error_max_abs_deg"] == pytest.approx(0.0)
    assert summary["gates"]["single_owner_command_delivered"]


def test_steady_oscillation_cannot_cancel_into_a_false_pass():
    summary = summarize_tracking_step(
        make_records(
            response=lambda elapsed: (
                5.0
                if elapsed < 6.0
                else 5.0 + 2.0 * math.sin(2.0 * math.pi * elapsed)
            )
        ),
        start_angle_rad=0.0,
        target_rad=math.radians(5.0),
        config=RudderTrackingConfig(),
    )

    assert summary["steady_tracking_error_mean_abs_deg"] < 0.1
    assert summary["steady_tracking_error_max_abs_deg"] > 1.9
    assert not summary["gates"]["steady_tracking_error"]
    assert not summary["passed"]


def test_velocity_limit_and_unloaded_speed_are_independent_failures():
    summary = summarize_tracking_step(
        make_records(velocity=lambda elapsed: 3.0, speed=0.3),
        start_angle_rad=0.0,
        target_rad=math.radians(5.0),
        config=RudderTrackingConfig(),
    )

    assert not summary["gates"]["velocity_limit_fraction"]
    assert not summary["gates"]["unloaded_speed"]
    assert summary["velocity_limit_fraction"] == pytest.approx(1.0)


def test_invalid_telemetry_returns_explicit_failure():
    summary = summarize_tracking_step(
        make_records(valid=False),
        start_angle_rad=0.0,
        target_rad=math.radians(5.0),
        config=RudderTrackingConfig(),
    )
    assert summary["failed_gates"] == ["telemetry"]
    assert not summary["passed"]


def test_run_summary_preserves_each_target():
    records = make_records()
    for record in records:
        record["step_index"] = 3
    summary = summarize_tracking_run(
        records,
        starts_by_step={3: 0.0},
        targets_by_step={3: math.radians(5.0)},
        config=RudderTrackingConfig(),
    )
    assert summary["step_count"] == 1
    assert summary["steps"][0]["target_rudder_deg"] == pytest.approx(5.0)
    assert summary["passed"]


def test_live_record_schema_keeps_target_command_actual_and_source_ages():
    snapshot = RosPerformanceSnapshot(
        sim_time_s=12.5,
        x_east_m=0.01,
        y_north_m=-0.02,
        z_up_m=0.2,
        speed_over_ground_mps=0.03,
        roll_frd_deg=0.5,
        pitch_frd_deg=-0.2,
        heading_deg=359.0,
        actual_rudder_angle_rad=0.08,
        actual_rudder_velocity_rad_s=0.2,
        base_rudder_rad=-0.1,
        final_rudder_rad=0.087,
        odom_age_s=0.1,
        rudder_age_s=0.1,
        actuator_age_s=0.1,
    )
    record = make_record(
        trial_id="test",
        step_index=1,
        target_rad=math.radians(5.0),
        start_sim_time_s=12.0,
        snapshot=snapshot,
        maximum_source_age_s=0.75,
    )

    assert record["telemetry_valid"]
    assert record["elapsed_step_s"] == pytest.approx(0.5)
    assert record["tracking_error_rad"] == pytest.approx(0.08 - 0.087)
    assert record["target_tracking_error_rad"] == pytest.approx(
        0.08 - math.radians(5.0)
    )


def test_rudder_command_has_one_base_to_adapter_to_controller_path():
    repo = Path(__file__).resolve().parents[3]
    # The SDF uses Gazebo extension tags without declaring an XML namespace;
    # inject a parser-only namespace instead of rewriting the production SDF.
    sdf_path = repo / "models/dave_robot_models/description/sailboat/model.sdf"
    source = sdf_path.read_text(encoding="utf-8").replace(
        '<sdf version="1.7">',
        '<sdf version="1.7" xmlns:gz="urn:gazebo-extension">',
        1,
    )
    root = ET.fromstring(source)
    plugins = list(root.iter("plugin"))

    ardupilot = next(item for item in plugins if item.get("name") == "ArduPilotPlugin")
    rudder_channel = next(
        item for item in ardupilot.findall("control") if item.get("channel") == "0"
    )
    controller = next(
        item
        for item in plugins
        if item.get("name") == "gz::sim::systems::JointPositionController"
        and item.findtext("joint_name") == "rudder_joint"
    )

    assert rudder_channel.findtext("type") == "COMMAND"
    assert rudder_channel.findtext("cmd_topic") == (
        "/model/sailboat/joint/rudder_joint/base_cmd_pos"
    )
    assert controller.findtext("topic") == (
        "/model/sailboat/joint/rudder_joint/cmd_pos"
    )
    assert float(controller.findtext("p_gain")) == pytest.approx(400.0)
    assert float(controller.findtext("i_gain")) == pytest.approx(0.0)
    assert float(controller.findtext("d_gain")) == pytest.approx(7.5)
    assert float(controller.findtext("cmd_max")) == pytest.approx(50.0)
    assert float(controller.findtext("cmd_min")) == pytest.approx(-50.0)

    rudder_joint = root.find(".//joint[@name='rudder_joint']")
    assert rudder_joint is not None
    assert float(rudder_joint.findtext("axis/limit/lower")) == pytest.approx(
        -math.pi / 4, abs=1e-5
    )
    assert float(rudder_joint.findtext("axis/limit/upper")) == pytest.approx(
        math.pi / 4, abs=1e-5
    )
    assert float(rudder_joint.findtext("axis/limit/effort")) == pytest.approx(50.0)
    assert float(rudder_joint.findtext("axis/limit/velocity")) == pytest.approx(3.0)

    adapter = (
        repo / "models/dave_robot_models/scripts/sailboat_command_adapter.py"
    ).read_text(encoding="utf-8")
    assert 'f"{joint_prefix}/{joint}/base_cmd_pos"' in adapter
    assert 'f"{joint_prefix}/rudder_joint/cmd_pos"' in adapter
    assert "diagnostic_rudder_override_enabled" in adapter
