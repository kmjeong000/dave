from __future__ import annotations

import math
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from experiments.sailboat_BO import run_trial


class FakeMessage:
    def __init__(self, message_type: str, **fields):
        self._message_type = message_type
        for name, value in fields.items():
            setattr(self, name, value)

    def get_type(self) -> str:
        return self._message_type


class FakeMavlink:
    MAV_MISSION_ACCEPTED = 0
    MAV_MISSION_OPERATION_CANCELLED = 15
    MAV_MISSION_NO_SPACE = 4
    MAV_MISSION_ERROR = 1
    MAV_MISSION_TYPE_MISSION = 0
    MAV_FRAME_GLOBAL = 0
    MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
    MAV_FRAME_GLOBAL_INT = 5
    MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6
    MAV_CMD_NAV_WAYPOINT = 16
    MAV_CMD_DO_SET_MISSION_CURRENT = 224
    MAV_CMD_SET_MESSAGE_INTERVAL = 511
    enums = {}


class FakeMavSender:
    def __init__(self):
        self.calls: dict[str, list[tuple]] = {}

    def _record(self, name: str, args: tuple) -> None:
        self.calls.setdefault(name, []).append(args)

    def mission_request_list_send(self, *args) -> None:
        self._record("mission_request_list_send", args)

    def mission_clear_all_send(self, *args) -> None:
        self._record("mission_clear_all_send", args)

    def mission_count_send(self, *args) -> None:
        self._record("mission_count_send", args)

    def mission_item_send(self, *args) -> None:
        self._record("mission_item_send", args)

    def mission_item_int_send(self, *args) -> None:
        self._record("mission_item_int_send", args)

    def mission_request_int_send(self, *args) -> None:
        self._record("mission_request_int_send", args)

    def mission_ack_send(self, *args) -> None:
        self._record("mission_ack_send", args)

    def mission_set_current_send(self, *args) -> None:
        self._record("mission_set_current_send", args)

    def command_long_send(self, *args) -> None:
        self._record("command_long_send", args)


class FakeMaster:
    def __init__(self, messages: list[FakeMessage] | None = None):
        self.target_system = 1
        self.target_component = 1
        self.mavlink = FakeMavlink
        self.mav = FakeMavSender()
        self.messages = list(messages or [])

    def recv_match(self, **_kwargs):
        if not self.messages:
            return None
        return self.messages.pop(0)


def make_context(*, home_offset: bool = True):
    return SimpleNamespace(
        study_cfg={
            "home_llh": [44.65870, -124.06556, 0.0],
            "mission_seq_home_offset": home_offset,
        },
        scenario_cfg={"mission": {"frame": "gazebo_xy_m"}},
        termination_cfg={"success_radius_m": 5.0},
    )


def downloaded_item(
    seq: int,
    lat_deg: float,
    lon_deg: float,
) -> FakeMessage:
    return FakeMessage(
        "MISSION_ITEM_INT",
        seq=seq,
        command=FakeMavlink.MAV_CMD_NAV_WAYPOINT,
        x=int(round(lat_deg * 1e7)),
        y=int(round(lon_deg * 1e7)),
    )


class MissionUploadTests(unittest.TestCase):
    def test_wind_message_interval_is_requested(self):
        master = FakeMaster()

        run_trial.request_message_intervals(master)

        interval_calls = master.mav.calls["command_long_send"]
        wind_call = next(call for call in interval_calls if call[4] == 168.0)
        self.assertEqual(wind_call[2], FakeMavlink.MAV_CMD_SET_MESSAGE_INTERVAL)
        self.assertEqual(wind_call[5], 200000.0)

    def test_servo_and_mavlink_wind_messages_preserve_raw_diagnostics(self):
        state = run_trial.TelemetryState()
        servo_params = {
            "SERVO1_MIN": 1000.0,
            "SERVO1_MAX": 2000.0,
            "SERVO2_MIN": 1000.0,
            "SERVO2_MAX": 2000.0,
            "SAIL_ANGLE_MIN": 0.0,
            "SAIL_ANGLE_MAX": 45.0,
        }

        run_trial.update_state_from_message(
            state,
            FakeMessage("SERVO_OUTPUT_RAW", servo1_raw=1400, servo2_raw=1000),
            44.65870,
            -124.06556,
            servo_params,
            "gazebo_xy_m",
        )
        run_trial.update_state_from_message(
            state,
            FakeMessage("WIND", direction=180.0, speed=8.25, speed_z=-0.2),
            44.65870,
            -124.06556,
            servo_params,
            "gazebo_xy_m",
        )

        self.assertEqual(state.servo1_raw_pwm, 1400.0)
        self.assertEqual(state.servo2_raw_pwm, 1000.0)
        self.assertEqual(state.servo_output_update_count, 1)
        self.assertAlmostEqual(state.sail_cmd_rad, 0.0)
        self.assertEqual(state.mavlink_wind_direction_deg, 180.0)
        self.assertEqual(state.mavlink_wind_speed_mps, 8.25)
        self.assertEqual(state.mavlink_wind_speed_z_mps, -0.2)
        self.assertEqual(state.mavlink_wind_update_count, 1)

    def test_sample_row_contains_servo_and_mavlink_wind_diagnostics(self):
        context = SimpleNamespace(
            study_cfg={"mission_seq_home_offset": True},
            scenario_cfg={
                "spawn": {"x_m": 0.0, "y_m": 0.0},
                "world": {"wind_world_xyz_mps": [0.0, 8.0, 0.0]},
            },
            termination_cfg={
                "position_source": "mavlink",
                "roll_source": "mavlink",
            },
        )
        state = run_trial.TelemetryState(
            sim_time_s=10.0,
            x_m=1.0,
            y_m=2.0,
            yaw_rad=0.0,
            surge_speed_mps=1.5,
            roll_deg=2.0,
            servo1_raw_pwm=1400.0,
            servo2_raw_pwm=1000.0,
            servo_output_update_count=7,
            rudder_cmd_rad=-0.1,
            sail_cmd_rad=0.0,
            mavlink_wind_direction_deg=180.0,
            mavlink_wind_speed_mps=8.25,
            mavlink_wind_speed_z_mps=-0.2,
            mavlink_wind_update_count=4,
            mission_seq=1,
        )

        sample = run_trial.build_sample_row(
            context=context,
            state=state,
            mission_waypoints=[run_trial.MissionWaypoint(seq=0, x_m=0.0, y_m=60.0)],
            ros_snapshot=None,
            raw_waypoint_index=0,
            effective_target_index=0,
            waypoint_completed_count=0,
            waypoint_capture_count=0,
            waypoint_capture_active=False,
        )

        self.assertIsNotNone(sample)
        self.assertTrue(sample["servo_output_valid"])
        self.assertEqual(sample["servo2_raw_pwm"], 1000.0)
        self.assertEqual(sample["servo_output_update_count"], 7)
        self.assertTrue(sample["mavlink_wind_valid"])
        self.assertEqual(sample["mavlink_wind_update_count"], 4)
        self.assertEqual(sample["mavlink_wind_direction_deg"], 180.0)
        self.assertAlmostEqual(sample["mavlink_wind_from_direction_rad"], -math.pi)
        self.assertEqual(sample["mavlink_wind_speed_mps"], 8.25)
        self.assertEqual(sample["mavlink_wind_speed_z_mps"], -0.2)

    def test_sheet_minimum_diagnostic_requires_valid_near_minimum_pwm(self):
        self.assertFalse(
            run_trial.servo_pwm_at_minimum(
                servo_output_valid=False,
                servo_raw_pwm=1000.0,
                servo_min_pwm=1000.0,
            )
        )
        self.assertTrue(
            run_trial.servo_pwm_at_minimum(
                servo_output_valid=True,
                servo_raw_pwm=1001.0,
                servo_min_pwm=1000.0,
            )
        )
        self.assertFalse(
            run_trial.servo_pwm_at_minimum(
                servo_output_valid=True,
                servo_raw_pwm=1002.0,
                servo_min_pwm=1000.0,
            )
        )

    def test_scenarios_persist_servo_and_mavlink_wind_csv_fields(self):
        repo_root = Path(__file__).resolve().parents[3]
        required_fields = {
            "servo_output_valid",
            "servo_output_update_count",
            "servo1_raw_pwm",
            "servo2_raw_pwm",
            "servo2_at_min",
            "servo2_at_min_continuous_s",
            "servo2_entered_min_this_sample",
            "servo2_recovered_this_sample",
            "mavlink_wind_valid",
            "mavlink_wind_update_count",
            "mavlink_wind_direction_deg",
            "mavlink_wind_from_direction_rad",
            "mavlink_wind_speed_mps",
            "mavlink_wind_speed_z_mps",
        }

        for scenario_name in ("scenario.yaml", "scenario_generalization.yaml"):
            config = run_trial.load_yaml(
                repo_root / "experiments" / "sailboat_BO" / scenario_name
            )
            csv_fields = set(config["logging"]["csv_fields"])
            self.assertTrue(
                required_fields.issubset(csv_fields),
                f"{scenario_name} is missing {sorted(required_fields - csv_fields)}",
            )

    def test_gazebo_xy_m_uses_standard_enu_world_axes(self):
        self.assertEqual(
            run_trial.mission_xy_to_enu("gazebo_xy_m", 12.5, -34.0),
            (12.5, -34.0),
        )
        self.assertEqual(
            run_trial.enu_to_mission_xy("gazebo_xy_m", 12.5, -34.0),
            (12.5, -34.0),
        )

    def test_upwind_tack_waypoints_remain_on_gazebo_north_south_axis(self):
        repo_root = Path(__file__).resolve().parents[3]
        scenario = run_trial.load_yaml(
            repo_root / "experiments" / "sailboat_BO" / "scenario.yaml"
        )
        upwind = next(
            item
            for item in scenario["scenarios"]
            if item["id"] == "train_upwind_tack"
        )
        frame = upwind["mission"]["frame"]
        waypoints = upwind["mission"]["waypoints"]
        self.assertEqual(frame, "gazebo_xy_m")
        self.assertEqual(waypoints, [[0, 60], [0, -60]])

        home_lat, home_lon = 44.65870, -124.06556

        north_lat, north_lon = run_trial.mission_xy_to_geodetic(
            frame,
            home_lat,
            home_lon,
            *waypoints[0],
        )
        south_lat, south_lon = run_trial.mission_xy_to_geodetic(
            frame,
            home_lat,
            home_lon,
            *waypoints[1],
        )
        north_east_m, north_north_m = run_trial.geodetic_to_enu(
            home_lat,
            home_lon,
            north_lat,
            north_lon,
        )
        south_east_m, south_north_m = run_trial.geodetic_to_enu(
            home_lat,
            home_lon,
            south_lat,
            south_lon,
        )

        self.assertAlmostEqual(north_east_m, 0.0, places=6)
        self.assertAlmostEqual(north_north_m, 60.0, places=6)
        self.assertAlmostEqual(south_east_m, 0.0, places=6)
        self.assertAlmostEqual(south_north_m, -60.0, places=6)

    def test_mavlink_position_round_trip_preserves_gazebo_xy(self):
        home_lat, home_lon = 44.65870, -124.06556
        expected_x_m, expected_y_m = 0.0, 60.0
        lat_deg, lon_deg = run_trial.mission_xy_to_geodetic(
            "gazebo_xy_m",
            home_lat,
            home_lon,
            expected_x_m,
            expected_y_m,
        )
        state = run_trial.TelemetryState()
        message = FakeMessage(
            "GLOBAL_POSITION_INT",
            time_boot_ms=1000,
            lat=int(round(lat_deg * 1e7)),
            lon=int(round(lon_deg * 1e7)),
            hdg=0,
            vx=0,
            vy=0,
        )

        run_trial.update_state_from_message(
            state,
            message,
            home_lat,
            home_lon,
            {},
            "gazebo_xy_m",
        )

        self.assertAlmostEqual(state.x_m, expected_x_m, delta=0.1)
        self.assertAlmostEqual(state.y_m, expected_y_m, delta=0.1)

    def test_model_sdf_uses_official_gazebo_enu_to_ned_transform(self):
        repo_root = Path(__file__).resolve().parents[3]
        model_sdf = (
            repo_root
            / "models"
            / "dave_robot_models"
            / "description"
            / "sailboat"
            / "model.sdf"
        )
        model_text = model_sdf.read_text(encoding="utf-8")
        plugin_match = re.search(
            r"<plugin\b[^>]*\bname=['\"]ArduPilotPlugin['\"][^>]*>.*?</plugin>",
            model_text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(plugin_match, "ArduPilotPlugin block is missing")
        plugin = ET.fromstring(plugin_match.group(0))
        transform = tuple(
            float(value)
            for value in plugin.findtext("gazeboXYZToNED", "").split()
        )

        self.assertEqual(len(transform), 6)
        self.assertEqual(transform[:3], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(transform[3], math.pi, places=3)
        self.assertAlmostEqual(transform[4], 0.0, places=6)
        self.assertAlmostEqual(transform[5], math.pi / 2.0, places=3)

        rudder_control = next(
            control
            for control in plugin.findall("control")
            if control.get("channel") == "0"
        )
        self.assertEqual(rudder_control.findtext("jointName"), "rudder_joint")
        self.assertAlmostEqual(
            float(rudder_control.findtext("offset", "nan")),
            -0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(rudder_control.findtext("multiplier", "nan")),
            1.5708,
            places=6,
        )

    def test_trial_param_rendering_does_not_add_rudder_servo_reversal(self):
        repo_root = Path(__file__).resolve().parents[3]
        base_param_file = (
            repo_root
            / "models"
            / "dave_robot_models"
            / "config"
            / "sailboat"
            / "ardurover.parm"
        )

        with TemporaryDirectory() as temporary_directory:
            trial_param_file = Path(temporary_directory) / "trial.parm"
            run_trial.write_trial_param_file(
                base_param_file,
                {"SAIL_NO_GO_ANGLE": 57.0},
                trial_param_file,
            )
            trial_params = run_trial.read_param_map(trial_param_file)

        self.assertNotIn("SERVO1_REVERSED", trial_params)
        self.assertEqual(trial_params["SAIL_NO_GO_ANGLE"], 57.0)

    def test_upload_includes_home_verifies_readback_and_starts_at_seq_one(self):
        context = make_context()
        waypoints = [
            run_trial.MissionWaypoint(seq=0, x_m=60.0, y_m=0.0),
            run_trial.MissionWaypoint(seq=1, x_m=-60.0, y_m=0.0),
        ]
        home_lat, home_lon, _ = context.study_cfg["home_llh"]
        first_lat, first_lon = run_trial.mission_xy_to_geodetic(
            "gazebo_xy_m", home_lat, home_lon, 60.0, 0.0
        )
        second_lat, second_lon = run_trial.mission_xy_to_geodetic(
            "gazebo_xy_m", home_lat, home_lon, -60.0, 0.0
        )
        adjusted_home_lat = home_lat + math.degrees(2.0 / run_trial.EARTH_RADIUS_M)
        messages = [
            FakeMessage("MISSION_COUNT", count=0),
            FakeMessage("MISSION_ACK", type=FakeMavlink.MAV_MISSION_ACCEPTED),
            FakeMessage("MISSION_REQUEST_INT", seq=0),
            FakeMessage("MISSION_REQUEST_INT", seq=1),
            FakeMessage("MISSION_REQUEST_INT", seq=2),
            FakeMessage("MISSION_ACK", type=FakeMavlink.MAV_MISSION_ACCEPTED),
            FakeMessage("MISSION_COUNT", count=3),
            downloaded_item(0, adjusted_home_lat, home_lon),
            downloaded_item(1, first_lat, first_lon),
            downloaded_item(2, second_lat, second_lon),
        ]
        master = FakeMaster(messages)

        run_trial.upload_mission(master, context, waypoints)

        count_calls = master.mav.calls["mission_count_send"]
        self.assertEqual(count_calls[-1][2], 3)
        sent_items = master.mav.calls["mission_item_int_send"]
        self.assertEqual([call[2] for call in sent_items], [0, 1, 2])
        self.assertEqual(sent_items[0][3], FakeMavlink.MAV_FRAME_GLOBAL_INT)
        self.assertEqual(
            [call[2] for call in master.mav.calls["mission_request_int_send"]],
            [0, 1, 2],
        )
        self.assertEqual(master.mav.calls["mission_set_current_send"][-1][2], 1)

    def test_verification_rejects_wrong_mission_count(self):
        context = make_context()
        upload_items = run_trial.build_mission_upload_items(
            [run_trial.MissionWaypoint(seq=0, x_m=10.0, y_m=0.0)]
        )
        master = FakeMaster([FakeMessage("MISSION_COUNT", count=1)])

        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            run_trial.verify_uploaded_mission(master, context, upload_items)

    def test_verification_allows_sub_meter_waypoint_readback_error(self):
        context = make_context()
        upload_items = run_trial.build_mission_upload_items(
            [run_trial.MissionWaypoint(seq=0, x_m=10.0, y_m=0.0)]
        )
        home_lat, home_lon, _ = context.study_cfg["home_llh"]
        waypoint_lat, waypoint_lon = run_trial.mission_xy_to_geodetic(
            "gazebo_xy_m", home_lat, home_lon, 10.0, 0.0
        )
        adjusted_waypoint_lat = waypoint_lat + math.degrees(0.8 / run_trial.EARTH_RADIUS_M)
        master = FakeMaster(
            [
                FakeMessage("MISSION_COUNT", count=2),
                downloaded_item(0, home_lat, home_lon),
                downloaded_item(1, adjusted_waypoint_lat, waypoint_lon),
            ]
        )

        run_trial.verify_uploaded_mission(master, context, upload_items)

    def test_launch_command_converts_course_angle_for_y_forward_hull(self):
        context = SimpleNamespace(
            launch_file="/tmp/launch.py",
            study_cfg={
                "home_llh": [44.0, -124.0, 0.0],
                "namespace": "sailboat",
            },
            scenario_cfg={
                "spawn": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 0.2,
                    "yaw_deg": 26.565,
                },
            },
            files=SimpleNamespace(param_file="/tmp/trial.parm"),
        )

        launch_cmd = run_trial.build_launch_command(context, "dogleg_world")

        yaw_arg = next(item for item in launch_cmd if item.startswith("yaw:="))
        roll_arg = next(item for item in launch_cmd if item.startswith("roll:="))
        pitch_arg = next(item for item in launch_cmd if item.startswith("pitch:="))
        home_arg = next(item for item in launch_cmd if item.startswith("ardupilot_home:="))
        self.assertAlmostEqual(
            float(yaw_arg.split(":=", 1)[1]),
            math.radians(-63.435),
            places=6,
        )
        self.assertEqual(roll_arg, "roll:=0.0")
        self.assertEqual(pitch_arg, "pitch:=0.0")
        self.assertEqual(home_arg, "ardupilot_home:=44.0,-124.0,0.0,63.435")

    def test_launch_command_passes_explicit_gazebo_roll_and_pitch(self):
        context = SimpleNamespace(
            launch_file="/tmp/launch.py",
            study_cfg={
                "home_llh": [44.0, -124.0, 0.0],
                "namespace": "sailboat",
            },
            scenario_cfg={
                "spawn": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 0.2,
                    "heading_deg": 90.0,
                    "gazebo_roll_deg": -5.0,
                    "gazebo_pitch_deg": 10.0,
                },
            },
            files=SimpleNamespace(param_file="/tmp/trial.parm"),
        )

        launch_cmd = run_trial.build_launch_command(context, "frame_world")

        self.assertIn(f"roll:={math.radians(-5.0)}", launch_cmd)
        self.assertIn(f"pitch:={math.radians(10.0)}", launch_cmd)
        self.assertIn(f"yaw:={math.radians(-90.0)}", launch_cmd)

    def test_gazebo_odometry_does_not_trust_mavlink_reached_for_completion(self):
        context = make_context()
        self.assertFalse(run_trial.trust_mavlink_reached_for_completion(context))

        mavlink_context = make_context()
        mavlink_context.termination_cfg["position_source"] = "mavlink"
        self.assertTrue(run_trial.trust_mavlink_reached_for_completion(mavlink_context))

    def test_gazebo_capture_uses_local_distance_without_nav_wp_gate(self):
        capture_distance = run_trial.compute_capture_distance_m(
            local_distance_to_wp_m=1.8,
            nav_wp_dist_m=9.0,
            raw_waypoint_index=1,
            effective_target_index=1,
            use_nav_wp_dist=False,
        )

        self.assertEqual(capture_distance, 1.8)

    def test_gazebo_reached_completion_requires_local_distance_gate(self):
        context = make_context()
        self.assertTrue(
            run_trial.accept_mavlink_reached_for_completion(
                context,
                raw_reached_seq=2,
                waypoint_count=2,
                distance_to_wp_m=5.65,
            )
        )
        self.assertFalse(
            run_trial.accept_mavlink_reached_for_completion(
                context,
                raw_reached_seq=2,
                waypoint_count=2,
                distance_to_wp_m=55.0,
            )
        )

    def test_advance_uses_only_home_offset_raw_sequence(self):
        context = make_context()
        state = run_trial.TelemetryState(mission_seq=1)
        master = FakeMaster([FakeMessage("MISSION_CURRENT", seq=2)])

        advanced = run_trial.advance_mission_current_to_waypoint(
            master,
            context,
            state,
            next_waypoint_index=1,
            waypoint_count=3,
            home_lat_deg=44.65870,
            home_lon_deg=-124.06556,
            servo_params={},
        )

        self.assertTrue(advanced)
        requested_sequences = [
            call[2] for call in master.mav.calls["mission_set_current_send"]
        ]
        self.assertEqual(requested_sequences, [2])

    def test_local_capture_fallback_reuploads_full_mission(self):
        context = make_context()
        master = FakeMaster()
        waypoints = [
            run_trial.MissionWaypoint(seq=0, x_m=10.0, y_m=0.0),
            run_trial.MissionWaypoint(seq=1, x_m=20.0, y_m=0.0),
            run_trial.MissionWaypoint(seq=2, x_m=30.0, y_m=0.0),
        ]

        with (
            patch.object(run_trial, "upload_mission") as upload_mock,
            patch.object(run_trial, "set_vehicle_mode") as mode_mock,
        ):
            retargeted = run_trial.retarget_mission_after_local_capture(
                master,
                context,
                waypoints,
                next_waypoint_index=1,
            )

        self.assertTrue(retargeted)
        upload_mock.assert_called_once_with(
            master,
            context,
            waypoints,
            start_waypoint_index=1,
        )
        mode_mock.assert_called_once_with(master, "AUTO", timeout_s=10.0)


class StartupStateHoldTests(unittest.TestCase):
    def test_startup_hold_topic_defaults_to_model_namespace(self):
        context = SimpleNamespace(study_cfg={"namespace": "/test_boat/"})

        self.assertTrue(run_trial.startup_state_hold_enabled(context))
        self.assertEqual(
            run_trial.startup_state_hold_topic(context),
            "/model/test_boat/startup_hold",
        )

    def test_startup_hold_publish_repeats_the_requested_boolean(self):
        context = SimpleNamespace(
            study_cfg={
                "namespace": "sailboat",
                "startup_state_release_repeats": 2,
                "startup_state_release_repeat_interval_s": 0.0,
            }
        )
        execution = run_trial.ExecutionConfig(backend="local")
        completed = SimpleNamespace(returncode=0, stderr="")

        with patch.object(
            run_trial.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            run_trial.publish_startup_state_held(
                context,
                execution,
                False,
                reason="unit_test",
            )

        self.assertEqual(run_mock.call_count, 2)
        command = run_mock.call_args_list[0].args[0]
        self.assertEqual(command[:2], ["bash", "-lc"])
        self.assertIn("/model/sailboat/startup_hold", command[-1])
        self.assertIn("data: false", command[-1])

    def test_navigation_physics_releases_hold_before_enabling_sail(self):
        context = SimpleNamespace(
            study_cfg={
                "startup_state_release_settle_s": 0.2,
                "sail_force_enable_settle_s": 0.5,
            }
        )
        execution = run_trial.ExecutionConfig(backend="local")
        events: list[tuple] = []

        def record_hold(_context, _execution, held, *, reason):
            events.append(("hold", held, reason))

        def record_sail(_context, _execution, enabled, *, reason):
            events.append(("sail", enabled, reason))

        with (
            patch.object(
                run_trial,
                "publish_startup_state_held",
                side_effect=record_hold,
            ),
            patch.object(
                run_trial,
                "publish_sail_force_enabled",
                side_effect=record_sail,
            ),
            patch.object(
                run_trial.time,
                "sleep",
                side_effect=lambda seconds: events.append(("sleep", seconds)),
            ),
        ):
            run_trial.start_navigation_physics(context, execution)

        self.assertEqual(
            events,
            [
                ("hold", False, "armed_auto"),
                ("sleep", 0.2),
                ("sail", True, "armed_auto"),
                ("sleep", 0.5),
            ],
        )


if __name__ == "__main__":
    unittest.main()
