import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.sailboat_performance.telemetry as telemetry_module
from experiments.sailboat_physics.frames import (
    frd_attitude_to_gazebo_quaternion,
)
from experiments.sailboat_performance.telemetry import (
    MavlinkPerformanceState,
    PerformanceRosCollector,
    RosPerformanceSnapshot,
    extract_joint_state,
    message_stamp_s,
    telemetry_ages_are_fresh,
)
from experiments.sailboat_performance.run_polar_trial import (
    diagnostic_comparison_fields,
    guided_target_record_fields,
    summarize,
    summarize_diagnostic_window,
)


def test_extract_actual_boom_by_name_not_order():
    value = extract_joint_state(
        ["rudder_joint", "sailboat::sail_joint"],
        [0.1, -0.4],
        [0.2, -0.3],
        "sail_joint",
    )
    assert value == pytest.approx((-0.4, -0.3))


def test_extract_joint_rejects_missing_or_ambiguous_names():
    assert extract_joint_state([], [], [], "sail_joint") is None
    assert (
        extract_joint_state(
            ["sail_joint", "model::sail_joint"], [0.1, 0.2], [], "sail_joint"
        )
        is None
    )


def test_collector_extracts_rudder_and_boom_from_same_joint_message():
    collector = PerformanceRosCollector("sailboat", "waves")
    collector._on_joint(
        SimpleNamespace(
            name=["sailboat::sail_joint", "rudder_joint"],
            position=[-0.6, 0.12],
            velocity=[-0.03, 0.45],
        )
    )

    assert collector._snapshot.actual_boom_angle_rad == pytest.approx(-0.6)
    assert collector._snapshot.actual_boom_velocity_rad_s == pytest.approx(-0.03)
    assert collector._snapshot.actual_rudder_angle_rad == pytest.approx(0.12)
    assert collector._snapshot.actual_rudder_velocity_rad_s == pytest.approx(0.45)
    assert collector._received_wall["rudder"] == pytest.approx(
        collector._received_wall["boom"]
    )


def test_ros_stamp_uses_nanoseconds():
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=250_000_000))
    )
    assert message_stamp_s(msg) == pytest.approx(12.25)


class FakeMessage:
    def __init__(self, kind, **values):
        self.kind = kind
        vars(self).update(values)

    def get_type(self):
        return self.kind


def test_mavlink_wind_age_and_heading_semantics():
    state = MavlinkPerformanceState()
    mavlink = SimpleNamespace(MAV_MODE_FLAG_SAFETY_ARMED=128)
    state.update(
        FakeMessage("WIND", direction=450.0, speed=8.0, speed_z=0.2),
        mavlink,
        10.0,
    )
    assert state.wind_from_deg == pytest.approx(90.0)
    assert state.wind_speed_mps == pytest.approx(8.0)
    assert state.wind_age_s(10.4) == pytest.approx(0.4)
    state.update(
        FakeMessage("HEARTBEAT", custom_mode=15, base_mode=128), mavlink, 11.0
    )
    assert state.heartbeat_custom_mode == 15
    assert state.armed


def test_mavlink_attitude_uses_ned_heading_and_frd_yaw_rate():
    state = MavlinkPerformanceState()
    mavlink = SimpleNamespace(MAV_MODE_FLAG_SAFETY_ARMED=128)

    state.update(
        FakeMessage("ATTITUDE", yaw=-math.pi / 2.0, yawspeed=0.5),
        mavlink,
        20.0,
    )

    assert state.yaw_deg == pytest.approx(270.0)
    assert state.yaw_rate_deg_s == pytest.approx(math.degrees(0.5))
    assert state.attitude_age_s(20.25) == pytest.approx(0.25)


def test_launch_bridges_actual_gazebo_joint_state_to_ros():
    repo = Path(__file__).resolve().parents[3]
    source = (
        repo / "models/dave_robot_models/config/sailboat/robot_config.py"
    ).read_text(encoding="utf-8")
    assert (
        "/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model" in source
    )


def test_runner_requests_validated_ten_hz_attitude_stream():
    from experiments.sailboat_BO.run_trial import MESSAGE_INTERVAL_HZ

    repo = Path(__file__).resolve().parents[3]
    source = (
        repo / "experiments/sailboat_performance/run_polar_trial.py"
    ).read_text(encoding="utf-8")
    assert MESSAGE_INTERVAL_HZ["ATTITUDE"] == pytest.approx(10.0)
    assert "request_message_intervals(master)" in source


def test_guided_target_fields_log_command_and_wrapped_heading_error():
    target = SimpleNamespace(
        heading_deg=0.0,
        velocity_north_mps=1.0,
        velocity_east_mps=0.0,
    )
    fields = guided_target_record_fields(target, 359.0)
    assert fields == pytest.approx(
        {
            "commanded_heading_deg": 0.0,
            "commanded_velocity_north_mps": 1.0,
            "commanded_velocity_east_mps": 0.0,
            "heading_error_deg": 1.0,
        }
    )


def make_odom(
    *, sim_time_s: float, heading_deg: float, angular_z_rad_s: float = 0.0
):
    qx, qy, qz, qw = frd_attitude_to_gazebo_quaternion(
        0.0,
        0.0,
        heading_deg,
    )
    seconds = int(sim_time_s)
    nanoseconds = round((sim_time_s - seconds) * 1e9)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds),
            frame_id="world",
        ),
        child_frame_id="sailboat/base_link",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=angular_z_rad_s),
            )
        ),
    )


def test_collector_derives_wrapped_yaw_rate_from_odom_sim_time():
    collector = PerformanceRosCollector("sailboat", "waves")

    collector._on_odom(make_odom(sim_time_s=10.0, heading_deg=359.0))
    assert collector._snapshot.derived_yaw_rate_deg_s is None

    collector._on_odom(
        make_odom(
            sim_time_s=10.5,
            heading_deg=1.0,
            angular_z_rad_s=-math.radians(3.0),
        )
    )
    assert collector._snapshot.heading_deg == pytest.approx(1.0)
    assert collector._snapshot.derived_yaw_rate_deg_s == pytest.approx(4.0)
    assert collector._snapshot.gazebo_angular_yaw_rate_deg_s == pytest.approx(3.0)


def test_snapshot_reference_is_taken_after_consistent_source_copy(monkeypatch):
    collector = PerformanceRosCollector("sailboat", "waves")
    collector._snapshot = RosPerformanceSnapshot(sim_time_s=10.0)
    source_time = 10.1
    old_caller_reference = 10.0
    collector._received_wall = {
        "odom": source_time,
        "rudder": source_time,
        "boom": source_time,
        "base_rudder_rad": source_time,
        "final_rudder_rad": source_time,
        "base_sheet_allowance_rad": source_time,
        "final_sheet_allowance_rad": source_time,
    }
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: 10.2)

    # This is the old race ordering: a caller reference captured before a
    # callback receipt timestamp would have produced a false negative age.
    assert old_caller_reference - source_time < 0.0
    snapshot = collector.snapshot()

    assert snapshot.odom_age_s == pytest.approx(0.1)
    assert snapshot.rudder_age_s == pytest.approx(0.1)
    assert snapshot.boom_age_s == pytest.approx(0.1)
    assert snapshot.actuator_age_s == pytest.approx(0.1)
    assert telemetry_ages_are_fresh(
        (
            snapshot.odom_age_s,
            snapshot.rudder_age_s,
            snapshot.boom_age_s,
            snapshot.actuator_age_s,
        ),
        0.75,
    )


def test_stale_detector_keeps_true_stale_and_negative_age_invalid():
    assert telemetry_ages_are_fresh((0.0, 0.25, 0.75), 0.75)
    assert not telemetry_ages_are_fresh((0.0, 0.751), 0.75)
    assert not telemetry_ages_are_fresh((-0.001, 0.1), 0.75)
    assert not telemetry_ages_are_fresh((None, 0.1), 0.75)


def test_diagnostic_comparisons_use_independent_sources_and_wrap_heading():
    fields = diagnostic_comparison_fields(
        RosPerformanceSnapshot(
            heading_deg=1.0,
            gazebo_angular_yaw_rate_deg_s=4.0,
            final_rudder_rad=0.1,
            actual_rudder_angle_rad=0.12,
        ),
        MavlinkPerformanceState(yaw_deg=359.0, yaw_rate_deg_s=3.0),
    )

    assert fields == pytest.approx(
        {
            "rudder_tracking_error_rad": 0.02,
            "mavlink_gazebo_heading_error_deg": -2.0,
            "mavlink_gazebo_yaw_rate_error_deg_s": -1.0,
        }
    )


def test_last_window_diagnostics_expose_control_response_and_boom_limit():
    records = []
    for index in range(4):
        records.append(
            {
                "sim_time_s": float(index),
                "commanded_heading_deg": 0.0,
                "commanded_velocity_north_mps": 1.0,
                "commanded_velocity_east_mps": 0.0,
                "heading_deg": [358.0, 359.0, 1.0, 2.0][index],
                "heading_error_deg": [-2.0, -1.0, 1.0, 2.0][index],
                "derived_yaw_rate_deg_s": [-3.0, -1.0, 1.0, 3.0][index],
                "gazebo_angular_yaw_rate_deg_s": [-2.5, -0.5, 0.5, 2.5][index],
                "mavlink_yaw_deg": [357.0, 358.0, 0.0, 1.0][index],
                "mavlink_yaw_rate_deg_s": [-2.0, 0.0, 1.0, 2.0][index],
                "mavlink_gazebo_heading_error_deg": -1.0,
                "mavlink_gazebo_yaw_rate_error_deg_s": 0.5,
                "final_rudder_rad": [-0.2, -0.1, 0.1, 0.2][index],
                "actual_rudder_angle_rad": [-0.18, -0.08, 0.12, 0.22][index],
                "actual_rudder_velocity_rad_s": [-0.4, -0.2, 0.2, 0.4][index],
                "rudder_tracking_error_rad": 0.02,
                "final_sheet_allowance_rad": [0.7854, 0.6, 0.3, 0.3][index],
                "actual_boom_angle_rad": [-0.7854, -0.7854, -0.2, 0.2][index],
                "actual_boom_velocity_rad_s": [-0.2, -0.1, 0.1, 0.2][index],
                "measured_signed_twa_deg": 90.0 + index,
                "roll_frd_deg": float(index),
                "speed_over_ground_mps": 2.0 + index,
            }
        )

    summary = summarize_diagnostic_window(
        records,
        window_s=2.0,
        boom_limit_rad=0.7854,
    )

    assert summary["sample_count"] == 3
    assert summary["heading_error_deg"]["rmse"] == pytest.approx(math.sqrt(2.0))
    assert summary["derived_yaw_rate_deg_s"]["max_abs"] == pytest.approx(3.0)
    assert summary["gazebo_angular_yaw_rate_deg_s"]["max_abs"] == pytest.approx(
        2.5
    )
    assert summary["mavlink_yaw_rate_deg_s"]["max_abs"] == pytest.approx(2.0)
    assert summary["mavlink_gazebo_heading_error_deg"]["mean"] == pytest.approx(
        -1.0
    )
    assert summary["mavlink_gazebo_heading_error_deg"]["max_abs"] == pytest.approx(
        1.0
    )
    assert summary["final_rudder_rad"]["sign_change_count"] == 1
    assert summary["actual_rudder_angle_rad"]["mean"] == pytest.approx(
        (-0.08 + 0.12 + 0.22) / 3.0
    )
    assert summary["actual_rudder_velocity_rad_s"]["max_abs"] == pytest.approx(
        0.4
    )
    assert summary["rudder_tracking_error_rad"]["mean"] == pytest.approx(0.02)
    assert summary["rudder_tracking_error_rad"]["max_abs"] == pytest.approx(0.02)
    assert summary["actual_boom_angle_rad"]["near_limit_fraction"] == pytest.approx(
        1.0 / 3.0
    )
    assert summary["actual_boom_angle_rad"][
        "outside_sheet_allowance_fraction"
    ] == pytest.approx(1.0 / 3.0)
    assert summary["actual_boom_velocity_rad_s"]["max_abs"] == pytest.approx(0.2)
    assert summary["final_sheet_allowance_rad"]["mean"] == pytest.approx(0.4)
    assert summary["measured_signed_twa_deg"]["mean"] == pytest.approx(92.0)


def test_summary_keeps_raw_quality_separate_from_measurement_quality():
    records = [
        {
            "sim_time_s": 1.0,
            "telemetry_valid": True,
            "guided_mode_active": True,
            "tack_detected": False,
            "odom_frame_id": "world",
            "odom_child_frame_id": "sailboat/base_link",
        },
        {
            "sim_time_s": 2.0,
            "telemetry_valid": False,
            "guided_mode_active": True,
            "tack_detected": False,
            "odom_frame_id": "world",
            "odom_child_frame_id": "sailboat/base_link",
        },
    ]

    summary = summarize(
        records,
        [],
        context=SimpleNamespace(
            trial_id="test_trial",
            started_at_utc="2026-09-09T00:00:00+00:00",
            params={},
        ),
        config_path=Path("polar_smoke.yaml"),
        params_path=Path("params.json"),
        config={
            "condition": {},
            "performance": {"steady_state": {"window_s": 2.0}},
        },
        commanded_wind=SimpleNamespace(east=0.0, north=8.0),
        outcome={"status": "failure", "failure_reason": "not_steady"},
        git_commit="deadbeef",
        boom_limit_rad=math.radians(45.0),
    )

    quality = summary["quality"]
    assert quality["raw_sample_count"] == 2
    assert quality["raw_telemetry_valid_sample_count"] == 1
    assert quality["raw_telemetry_valid_ratio"] == pytest.approx(0.5)
    assert quality["raw_guided_mode_active_ratio"] == pytest.approx(1.0)
    assert quality["measurement_sample_count"] == 0
    assert quality["valid_measurement_sample_count"] == 0
    assert quality["valid_measurement_ratio"] == pytest.approx(0.0)
    assert quality["guided_mode_active_ratio"] == pytest.approx(0.0)
