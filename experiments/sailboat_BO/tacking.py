"""Offline, frame-consistent diagnostic detection of physical tack events.

This module is deliberately separate from :mod:`score`.  The BO objective's
``tack_count`` is a course-side metric, whereas this diagnostic answers the
stricter question: did a boat hold one windward leg, pass through
head-to-wind, turn, and then hold the opposite leg?  It must therefore never
change the BO cost or a trial's success/failure result.

All wind/heading comparisons use MAVLink values from the same NED/FRD
convention.  In particular, do not combine ``yaw_gz_odom_rad`` with a MAVLink
wind bearing: their sign conventions are different in this model.
"""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


PORT_TACK = "PORT_TACK"
STARBOARD_TACK = "STARBOARD_TACK"
HEAD_TO_WIND = "HEAD_TO_WIND"
TACKING = "TACKING"
UNKNOWN = "UNKNOWN"

SIDE_STATES = {PORT_TACK, STARBOARD_TACK}

TACK_EVENT_FIELDS = (
    "event_id",
    "start_time_s",
    "end_time_s",
    "from_tack",
    "to_tack",
    "heading_before_deg",
    "heading_after_deg",
    "heading_change_deg",
    "relative_wind_before_deg",
    "min_abs_relative_wind_deg",
    "relative_wind_after_deg",
    "pre_leg_duration_s",
    "post_leg_duration_s",
    "pre_leg_distance_m",
    "post_leg_distance_m",
    "max_yaw_rate_deg_s",
    "min_speed_during_tack_mps",
    "boom_before_deg",
    "boom_after_deg",
    "boom_side_changed",
    "waypoint_index_start",
    "waypoint_index_end",
    "distance_to_wp_start_m",
    "distance_to_wp_end_m",
    "cross_track_start_m",
    "cross_track_end_m",
    "pre_leg_qualified",
    "crossed_head_to_wind",
    "post_leg_qualified",
    "heading_change_sufficient",
    "successful_tack",
    "failure_reason",
)


def wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_x = _finite(left.get("x_m"))
    left_y = _finite(left.get("y_m"))
    right_x = _finite(right.get("x_m"))
    right_y = _finite(right.get("y_m"))
    if None in (left_x, left_y, right_x, right_y):
        return 0.0
    return math.hypot(right_x - left_x, right_y - left_y)


def _side_name(state: str) -> str:
    return "port" if state == PORT_TACK else "starboard"


@dataclass(frozen=True)
class TackDetectorConfig:
    """Thresholds for a diagnostic event, not controller behaviour."""

    no_go_angle_deg: float
    # A close-hauled controller can track at the configured no-go boundary.
    # Treat a small amount inside that boundary as a stable leg, while using
    # a wider inner band for an unambiguous head-to-wind crossing.
    leg_margin_deg: float = 1.0
    head_to_wind_core_margin_deg: float = 5.0
    pre_leg_min_duration_s: float = 5.0
    post_leg_min_duration_s: float = 5.0
    pre_leg_min_distance_m: float = 3.0
    post_leg_min_distance_m: float = 3.0
    min_heading_change_deg: float = 45.0
    candidate_timeout_s: float = 30.0
    boom_deadband_deg: float = 5.0
    boom_max_sample_age_s: float = 1.5

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        no_go_angle_deg: float,
    ) -> "TackDetectorConfig":
        values = values or {}

        def value(name: str, default: float) -> float:
            parsed = _finite(values.get(name))
            return default if parsed is None else parsed

        leg_margin_deg = max(0.0, value("leg_margin_deg", 1.0))
        head_to_wind_core_margin_deg = max(
            0.0,
            value("head_to_wind_core_margin_deg", 5.0),
        )
        if head_to_wind_core_margin_deg <= leg_margin_deg:
            raise ValueError(
                "tack_diagnostics.head_to_wind_core_margin_deg must be greater "
                "than tack_diagnostics.leg_margin_deg"
            )
        return cls(
            no_go_angle_deg=max(0.0, no_go_angle_deg),
            leg_margin_deg=leg_margin_deg,
            head_to_wind_core_margin_deg=head_to_wind_core_margin_deg,
            pre_leg_min_duration_s=max(0.0, value("pre_leg_min_duration_s", 5.0)),
            post_leg_min_duration_s=max(0.0, value("post_leg_min_duration_s", 5.0)),
            pre_leg_min_distance_m=max(0.0, value("pre_leg_min_distance_m", 3.0)),
            post_leg_min_distance_m=max(0.0, value("post_leg_min_distance_m", 3.0)),
            min_heading_change_deg=max(0.0, value("min_heading_change_deg", 45.0)),
            candidate_timeout_s=max(1.0, value("candidate_timeout_s", 30.0)),
            boom_deadband_deg=max(0.0, value("boom_deadband_deg", 5.0)),
            boom_max_sample_age_s=max(0.0, value("boom_max_sample_age_s", 1.5)),
        )


@dataclass(frozen=True)
class BoomObservation:
    sim_time_s: float
    boom_angle_deg: float


def _state_for_relative_wind(
    relative_wind_angle_deg: float | None,
    config: TackDetectorConfig,
) -> str:
    if relative_wind_angle_deg is None:
        return UNKNOWN
    magnitude = abs(relative_wind_angle_deg)
    head_to_wind_threshold = max(
        0.0,
        config.no_go_angle_deg - config.head_to_wind_core_margin_deg,
    )
    stable_leg_threshold = max(
        0.0,
        config.no_go_angle_deg - config.leg_margin_deg,
    )
    if magnitude <= head_to_wind_threshold:
        return HEAD_TO_WIND
    if relative_wind_angle_deg >= stable_leg_threshold:
        # In NED/FRD, a positive wind-from relative bearing is starboard.
        return STARBOARD_TACK
    if relative_wind_angle_deg <= -stable_leg_threshold:
        return PORT_TACK
    return TACKING


def _annotate_observations(
    samples: Sequence[MutableMapping[str, Any]],
    config: TackDetectorConfig,
) -> None:
    previous_time_s: float | None = None
    previous_heading_rad: float | None = None

    for sample in samples:
        time_s = _finite(sample.get("sim_time_s"))
        heading_valid = bool(sample.get("mavlink_heading_valid", False))
        wind_valid = bool(sample.get("mavlink_wind_valid", False))
        heading_rad = _finite(sample.get("yaw_mav_rad")) if heading_valid else None
        wind_from_rad = (
            _finite(sample.get("mavlink_wind_from_direction_rad"))
            if wind_valid
            else None
        )
        relative_wind_angle_deg: float | None = None
        if heading_rad is not None and wind_from_rad is not None:
            relative_wind_angle_deg = math.degrees(wrap_pi(wind_from_rad - heading_rad))

        yaw_rate_deg_s: float | None = None
        if (
            time_s is not None
            and heading_rad is not None
            and previous_time_s is not None
            and previous_heading_rad is not None
            and time_s > previous_time_s
        ):
            yaw_rate_deg_s = math.degrees(
                wrap_pi(heading_rad - previous_heading_rad) / (time_s - previous_time_s)
            )

        sample["mavlink_heading_valid"] = heading_rad is not None
        sample["mavlink_heading_deg"] = (
            math.degrees(heading_rad) if heading_rad is not None else None
        )
        sample["relative_true_wind_angle_deg"] = relative_wind_angle_deg
        sample["yaw_rate_deg_s"] = yaw_rate_deg_s
        sample["tack_state"] = _state_for_relative_wind(relative_wind_angle_deg, config)
        sample["tack_event_id"] = None
        sample["actual_boom_angle_deg"] = None
        sample["boom_sample_age_s"] = None

        if time_s is not None and heading_rad is not None:
            previous_time_s = time_s
            previous_heading_rad = heading_rad


def _new_candidate(
    *,
    event_id: int,
    source: str,
    source_start_time_s: float,
    source_end_sample: Mapping[str, Any],
    pre_leg_duration_s: float,
    pre_leg_distance_m: float,
    transition_sample: Mapping[str, Any],
) -> dict[str, Any]:
    start_time_s = _finite(transition_sample.get("sim_time_s")) or 0.0
    relative_before = _finite(source_end_sample.get("relative_true_wind_angle_deg"))
    heading_before = _finite(source_end_sample.get("mavlink_heading_deg"))
    return {
        "event_id": event_id,
        "start_time_s": start_time_s,
        "end_time_s": None,
        "from_tack": _side_name(source),
        "to_tack": _side_name(STARBOARD_TACK if source == PORT_TACK else PORT_TACK),
        "source": source,
        "target": STARBOARD_TACK if source == PORT_TACK else PORT_TACK,
        "heading_before_deg": heading_before,
        "heading_after_deg": None,
        "heading_change_deg": None,
        "relative_wind_before_deg": relative_before,
        "min_abs_relative_wind_deg": abs(relative_before) if relative_before is not None else None,
        "relative_wind_after_deg": None,
        "pre_leg_duration_s": pre_leg_duration_s,
        "post_leg_duration_s": 0.0,
        "pre_leg_distance_m": pre_leg_distance_m,
        "post_leg_distance_m": 0.0,
        "max_yaw_rate_deg_s": 0.0,
        "min_speed_during_tack_mps": None,
        "boom_before_deg": None,
        "boom_after_deg": None,
        "boom_side_changed": None,
        "waypoint_index_start": source_end_sample.get("waypoint_index"),
        "waypoint_index_end": None,
        "distance_to_wp_start_m": source_end_sample.get("distance_to_wp_m"),
        "distance_to_wp_end_m": None,
        "cross_track_start_m": source_end_sample.get("cross_track_error_m"),
        "cross_track_end_m": None,
        "pre_leg_qualified": True,
        "crossed_head_to_wind": False,
        "post_leg_qualified": False,
        "heading_change_sufficient": False,
        "successful_tack": False,
        "failure_reason": "",
        "_sample_indices": [],
        "_last_post_sample": None,
        "_post_started_at_s": None,
        "_source_start_time_s": source_start_time_s,
    }


def _update_candidate(candidate: MutableMapping[str, Any], sample: Mapping[str, Any]) -> None:
    candidate["_sample_indices"].append(sample.get("_tack_sample_index"))
    relative = _finite(sample.get("relative_true_wind_angle_deg"))
    if relative is not None:
        previous = candidate["min_abs_relative_wind_deg"]
        candidate["min_abs_relative_wind_deg"] = (
            abs(relative) if previous is None else min(float(previous), abs(relative))
        )
    yaw_rate = _finite(sample.get("yaw_rate_deg_s"))
    if yaw_rate is not None:
        candidate["max_yaw_rate_deg_s"] = max(float(candidate["max_yaw_rate_deg_s"]), abs(yaw_rate))
    speed = _finite(sample.get("surge_speed_mps"))
    if speed is not None:
        minimum = candidate["min_speed_during_tack_mps"]
        candidate["min_speed_during_tack_mps"] = speed if minimum is None else min(float(minimum), speed)


def _finalize_candidate(
    candidate: MutableMapping[str, Any],
    sample: Mapping[str, Any],
    *,
    successful: bool,
    failure_reason: str,
    config: TackDetectorConfig,
) -> dict[str, Any]:
    end_time_s = _finite(sample.get("sim_time_s")) or float(candidate["start_time_s"])
    heading_after = _finite(sample.get("mavlink_heading_deg"))
    heading_before = _finite(candidate["heading_before_deg"])
    heading_change = (
        abs(math.degrees(wrap_pi(math.radians(heading_after - heading_before))))
        if heading_after is not None and heading_before is not None
        else None
    )
    candidate["end_time_s"] = end_time_s
    candidate["heading_after_deg"] = heading_after
    candidate["heading_change_deg"] = heading_change
    candidate["relative_wind_after_deg"] = _finite(sample.get("relative_true_wind_angle_deg"))
    candidate["waypoint_index_end"] = sample.get("waypoint_index")
    candidate["distance_to_wp_end_m"] = sample.get("distance_to_wp_m")
    candidate["cross_track_end_m"] = sample.get("cross_track_error_m")
    candidate["heading_change_sufficient"] = bool(
        heading_change is not None and heading_change >= config.min_heading_change_deg
    )
    candidate["successful_tack"] = bool(successful and candidate["heading_change_sufficient"])
    candidate["failure_reason"] = (
        "" if candidate["successful_tack"] else failure_reason
    )
    if not candidate["successful_tack"] and not candidate["failure_reason"]:
        candidate["failure_reason"] = "insufficient_heading_change"
    return dict(candidate)


def detect_tack_events(
    samples: Sequence[MutableMapping[str, Any]],
    config: TackDetectorConfig,
) -> list[dict[str, Any]]:
    """Annotate samples and return robust tack events.

    A success needs a qualified source leg, an observed head-to-wind crossing,
    and a qualified opposite leg.  A boom sign is intentionally diagnostic
    only: the physical boom is sampled less often than telemetry and is not a
    safe primary condition for an event.
    """

    _annotate_observations(samples, config)
    events: list[dict[str, Any]] = []
    stable_side: str | None = None
    stable_started_at_s: float | None = None
    stable_distance_m = 0.0
    stable_last_sample: MutableMapping[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    next_event_id = 1

    for index, sample in enumerate(samples):
        sample["_tack_sample_index"] = index
        time_s = _finite(sample.get("sim_time_s"))
        state = str(sample.get("tack_state", UNKNOWN))
        previous_sample = stable_last_sample

        if candidate is not None:
            _update_candidate(candidate, sample)
            sample["tack_event_id"] = candidate["event_id"]
            if state == HEAD_TO_WIND:
                candidate["crossed_head_to_wind"] = True
            if time_s is not None and time_s - float(candidate["start_time_s"]) > config.candidate_timeout_s:
                events.append(
                    _finalize_candidate(
                        candidate,
                        sample,
                        successful=False,
                        failure_reason="candidate_timeout",
                        config=config,
                    )
                )
                candidate = None
                stable_side = None
                stable_started_at_s = None
                stable_distance_m = 0.0
                stable_last_sample = None
                continue

            if state == candidate["source"]:
                events.append(
                    _finalize_candidate(
                        candidate,
                        sample,
                        successful=False,
                        failure_reason="returned_to_origin_tack",
                        config=config,
                    )
                )
                candidate = None
                stable_side = state
                stable_started_at_s = time_s
                stable_distance_m = 0.0
                stable_last_sample = sample
                continue

            if state == candidate["target"]:
                previous_post = candidate.get("_last_post_sample")
                if previous_post is None:
                    candidate["_post_started_at_s"] = time_s
                else:
                    candidate["post_leg_distance_m"] = float(candidate["post_leg_distance_m"]) + _distance(previous_post, sample)
                candidate["_last_post_sample"] = sample
                post_started_at_s = candidate.get("_post_started_at_s")
                if time_s is not None and post_started_at_s is not None:
                    candidate["post_leg_duration_s"] = max(0.0, time_s - float(post_started_at_s))
                candidate["post_leg_qualified"] = bool(
                    float(candidate["post_leg_duration_s"]) >= config.post_leg_min_duration_s
                    and float(candidate["post_leg_distance_m"]) >= config.post_leg_min_distance_m
                )
                if candidate["post_leg_qualified"]:
                    reason = "" if candidate["crossed_head_to_wind"] else "missing_head_to_wind_crossing"
                    event = _finalize_candidate(
                        candidate,
                        sample,
                        successful=bool(candidate["crossed_head_to_wind"]),
                        failure_reason=reason,
                        config=config,
                    )
                    events.append(event)
                    candidate = None
                    stable_side = state
                    stable_started_at_s = post_started_at_s
                    stable_distance_m = float(event["post_leg_distance_m"])
                    stable_last_sample = sample
                continue

            # A no-go/tacking band is part of the same candidate.  It neither
            # resets the source leg nor proves the opposite leg.
            continue

        if state in SIDE_STATES:
            if stable_side == state and stable_last_sample is not None:
                stable_distance_m += _distance(stable_last_sample, sample)
                stable_last_sample = sample
                continue

            if stable_side in SIDE_STATES and stable_last_sample is not None:
                duration_s = (
                    max(0.0, (time_s or 0.0) - float(stable_started_at_s))
                    if stable_started_at_s is not None
                    else 0.0
                )
                if (
                    duration_s >= config.pre_leg_min_duration_s
                    and stable_distance_m >= config.pre_leg_min_distance_m
                ):
                    candidate = _new_candidate(
                        event_id=next_event_id,
                        source=stable_side,
                        source_start_time_s=float(stable_started_at_s or time_s or 0.0),
                        source_end_sample=stable_last_sample,
                        pre_leg_duration_s=duration_s,
                        pre_leg_distance_m=stable_distance_m,
                        transition_sample=sample,
                    )
                    next_event_id += 1
                    _update_candidate(candidate, sample)
                    sample["tack_event_id"] = candidate["event_id"]
                    if state == candidate["target"]:
                        candidate["_post_started_at_s"] = time_s
                        candidate["_last_post_sample"] = sample
                    stable_side = None
                    stable_started_at_s = None
                    stable_distance_m = 0.0
                    stable_last_sample = None
                    continue

            stable_side = state
            stable_started_at_s = time_s
            stable_distance_m = 0.0
            stable_last_sample = sample
            continue

        if stable_side in SIDE_STATES and stable_last_sample is not None:
            duration_s = (
                max(0.0, (time_s or 0.0) - float(stable_started_at_s))
                if stable_started_at_s is not None
                else 0.0
            )
            if (
                duration_s >= config.pre_leg_min_duration_s
                and stable_distance_m >= config.pre_leg_min_distance_m
            ):
                candidate = _new_candidate(
                    event_id=next_event_id,
                    source=stable_side,
                    source_start_time_s=float(stable_started_at_s or time_s or 0.0),
                    source_end_sample=stable_last_sample,
                    pre_leg_duration_s=duration_s,
                    pre_leg_distance_m=stable_distance_m,
                    transition_sample=sample,
                )
                next_event_id += 1
                _update_candidate(candidate, sample)
                sample["tack_event_id"] = candidate["event_id"]
                if state == HEAD_TO_WIND:
                    candidate["crossed_head_to_wind"] = True
            stable_side = None
            stable_started_at_s = None
            stable_distance_m = 0.0
            stable_last_sample = None

    if candidate is not None and samples:
        last_sample = samples[-1]
        if candidate.get("_last_post_sample") is not None:
            reason = "insufficient_post_tack_dwell_or_distance"
        elif candidate.get("crossed_head_to_wind"):
            reason = "missing_opposite_tack_leg"
        else:
            reason = "missing_head_to_wind_crossing"
        events.append(
            _finalize_candidate(
                candidate,
                last_sample,
                successful=False,
                failure_reason=reason,
                config=config,
            )
        )

    for sample in samples:
        sample.pop("_tack_sample_index", None)
    for event in events:
        event.pop("_sample_indices", None)
        event.pop("_last_post_sample", None)
        event.pop("_post_started_at_s", None)
        event.pop("_source_start_time_s", None)
    return events


def attach_boom_observations(
    samples: Sequence[MutableMapping[str, Any]],
    events: Sequence[MutableMapping[str, Any]],
    boom_observations: Iterable[BoomObservation],
    config: TackDetectorConfig,
) -> None:
    """Join rate-limited physical boom observations without making them a gate."""

    observations = sorted(boom_observations, key=lambda item: item.sim_time_s)
    times = [item.sim_time_s for item in observations]

    def nearest(time_s: float | None) -> BoomObservation | None:
        if time_s is None or not observations:
            return None
        position = bisect_left(times, time_s)
        candidates = observations[max(0, position - 1):position + 1]
        if not candidates:
            return None
        value = min(candidates, key=lambda item: abs(item.sim_time_s - time_s))
        return value if abs(value.sim_time_s - time_s) <= config.boom_max_sample_age_s else None

    for sample in samples:
        sample_time = _finite(sample.get("sim_time_s"))
        observation = nearest(sample_time)
        if observation is not None and sample_time is not None:
            sample["actual_boom_angle_deg"] = observation.boom_angle_deg
            sample["boom_sample_age_s"] = sample_time - observation.sim_time_s

    for event in events:
        before = nearest(_finite(event.get("start_time_s")))
        after = nearest(_finite(event.get("end_time_s")))
        event["boom_before_deg"] = before.boom_angle_deg if before is not None else None
        event["boom_after_deg"] = after.boom_angle_deg if after is not None else None
        if before is None or after is None:
            event["boom_side_changed"] = None
        elif (
            abs(before.boom_angle_deg) <= config.boom_deadband_deg
            or abs(after.boom_angle_deg) <= config.boom_deadband_deg
        ):
            event["boom_side_changed"] = None
        else:
            event["boom_side_changed"] = before.boom_angle_deg * after.boom_angle_deg < 0.0


def write_tack_events_csv(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TACK_EVENT_FIELDS)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event.get(field) for field in TACK_EVENT_FIELDS})


def write_tack_summary_json(
    path: Path,
    events: Sequence[Mapping[str, Any]],
    config: TackDetectorConfig,
) -> None:
    successful = [event for event in events if bool(event.get("successful_tack"))]
    failures: dict[str, int] = {}
    for event in events:
        reason = str(event.get("failure_reason") or "")
        if reason:
            failures[reason] = failures.get(reason, 0) + 1
    payload = {
        "definition": "stable_side_leg -> head_to_wind -> stable_opposite_leg",
        "frame_contract": "MAVLink NED/FRD heading paired with MAVLink wind-from bearing",
        "config": asdict(config),
        "candidate_event_count": len(events),
        "successful_tack_count": len(successful),
        "failed_candidate_count": len(events) - len(successful),
        "failure_counts": failures,
        "boom_is_diagnostic_only": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
