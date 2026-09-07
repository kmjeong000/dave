from __future__ import annotations

import csv
import math
from pathlib import Path

from experiments.sailboat_BO.tacking import (
    BoomObservation,
    TackDetectorConfig,
    attach_boom_observations,
    detect_tack_events,
    write_tack_events_csv,
)


def sample(
    time_s: float,
    *,
    x_m: float,
    yaw_deg: float,
    wind_from_deg: float = 0.0,
    waypoint_index: int = 0,
) -> dict[str, object]:
    return {
        "sim_time_s": time_s,
        "x_m": x_m,
        "y_m": 0.0,
        "yaw_mav_rad": math.radians(yaw_deg),
        "mavlink_heading_valid": True,
        "mavlink_wind_valid": True,
        "mavlink_wind_from_direction_rad": math.radians(wind_from_deg),
        "surge_speed_mps": 1.0,
        "rudder_cmd_rad": 0.0,
        "sail_cmd_rad": 0.5,
        "waypoint_index": waypoint_index,
        "distance_to_wp_m": 50.0 - x_m,
        "cross_track_error_m": 1.0,
    }


def config() -> TackDetectorConfig:
    return TackDetectorConfig(
        no_go_angle_deg=45.0,
        leg_margin_deg=1.0,
        head_to_wind_core_margin_deg=5.0,
        pre_leg_min_duration_s=2.0,
        post_leg_min_duration_s=2.0,
        pre_leg_min_distance_m=2.0,
        post_leg_min_distance_m=2.0,
        min_heading_change_deg=45.0,
        candidate_timeout_s=20.0,
    )


def test_complete_port_to_starboard_tack_requires_legs_and_head_to_wind():
    samples = [
        sample(0.0, x_m=0.0, yaw_deg=70.0),
        sample(2.0, x_m=2.0, yaw_deg=70.0),
        sample(3.0, x_m=3.0, yaw_deg=0.0),
        sample(4.0, x_m=4.0, yaw_deg=-70.0),
        sample(6.0, x_m=6.0, yaw_deg=-70.0),
    ]

    events = detect_tack_events(samples, config())

    assert len(events) == 1
    event = events[0]
    assert event["from_tack"] == "port"
    assert event["to_tack"] == "starboard"
    assert event["crossed_head_to_wind"] is True
    assert event["pre_leg_qualified"] is True
    assert event["post_leg_qualified"] is True
    assert event["heading_change_sufficient"] is True
    assert event["successful_tack"] is True
    assert event["failure_reason"] == ""
    assert samples[2]["tack_state"] == "HEAD_TO_WIND"
    assert samples[2]["tack_event_id"] == 1


def test_close_hauled_leg_can_track_just_inside_no_go_boundary():
    """The controller's boundary command is a leg, not head-to-wind core."""

    samples = [
        sample(0.0, x_m=0.0, yaw_deg=44.0),
        sample(2.0, x_m=2.0, yaw_deg=44.0),
        sample(3.0, x_m=3.0, yaw_deg=0.0),
        sample(4.0, x_m=4.0, yaw_deg=-44.0),
        sample(6.0, x_m=6.0, yaw_deg=-44.0),
    ]

    events = detect_tack_events(samples, config())

    assert samples[1]["tack_state"] == "PORT_TACK"
    assert samples[2]["tack_state"] == "HEAD_TO_WIND"
    assert samples[3]["tack_state"] == "STARBOARD_TACK"
    assert len(events) == 1
    assert events[0]["successful_tack"] is True


def test_short_return_to_origin_is_rejected_as_chatter_not_tack():
    samples = [
        sample(0.0, x_m=0.0, yaw_deg=70.0),
        sample(2.0, x_m=2.0, yaw_deg=70.0),
        sample(3.0, x_m=3.0, yaw_deg=0.0),
        sample(3.4, x_m=3.4, yaw_deg=-70.0),
        sample(3.8, x_m=3.8, yaw_deg=70.0),
    ]

    events = detect_tack_events(samples, config())

    assert len(events) == 1
    assert events[0]["successful_tack"] is False
    assert events[0]["failure_reason"] == "returned_to_origin_tack"


def test_wind_only_crossing_without_a_real_turn_is_rejected():
    samples = [
        sample(0.0, x_m=0.0, yaw_deg=70.0, wind_from_deg=0.0),
        sample(2.0, x_m=2.0, yaw_deg=70.0, wind_from_deg=0.0),
        sample(3.0, x_m=3.0, yaw_deg=70.0, wind_from_deg=70.0),
        sample(4.0, x_m=4.0, yaw_deg=70.0, wind_from_deg=140.0),
        sample(6.0, x_m=6.0, yaw_deg=70.0, wind_from_deg=140.0),
    ]

    events = detect_tack_events(samples, config())

    assert len(events) == 1
    assert events[0]["crossed_head_to_wind"] is True
    assert events[0]["post_leg_qualified"] is True
    assert events[0]["heading_change_sufficient"] is False
    assert events[0]["successful_tack"] is False
    assert events[0]["failure_reason"] == "insufficient_heading_change"


def test_boom_is_supporting_evidence_not_a_success_requirement(tmp_path: Path):
    samples = [
        sample(0.0, x_m=0.0, yaw_deg=70.0),
        sample(2.0, x_m=2.0, yaw_deg=70.0),
        sample(3.0, x_m=3.0, yaw_deg=0.0),
        sample(4.0, x_m=4.0, yaw_deg=-70.0),
        sample(6.0, x_m=6.0, yaw_deg=-70.0),
    ]
    detector_config = config()
    events = detect_tack_events(samples, detector_config)

    attach_boom_observations(
        samples,
        events,
        [BoomObservation(3.0, -30.0), BoomObservation(6.0, 30.0)],
        detector_config,
    )
    assert events[0]["successful_tack"] is True
    assert events[0]["boom_side_changed"] is True
    assert samples[-1]["actual_boom_angle_deg"] == 30.0

    output = tmp_path / "tack_events.csv"
    write_tack_events_csv(output, events)
    with output.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["successful_tack"] == "True"
    assert rows[0]["boom_side_changed"] == "True"


def test_missing_paired_mavlink_heading_or_wind_is_unknown_not_a_tack():
    samples = [
        sample(0.0, x_m=0.0, yaw_deg=70.0),
        sample(2.0, x_m=2.0, yaw_deg=70.0),
    ]
    samples[1]["mavlink_wind_valid"] = False

    events = detect_tack_events(samples, config())

    assert events == []
    assert samples[1]["tack_state"] == "UNKNOWN"
