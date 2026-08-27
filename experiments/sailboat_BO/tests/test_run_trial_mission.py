from __future__ import annotations

import math
import unittest
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


if __name__ == "__main__":
    unittest.main()
