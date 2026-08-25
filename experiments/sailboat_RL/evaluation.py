from __future__ import annotations

from collections import Counter
from statistics import fmean, pstdev
from typing import Any, Iterable


def _finite_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _metric_summary(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, float | int | None]:
    values = _finite_values(records, key)
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_controller_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        record for record in records if record.get("evaluation_status") == "complete"
    ]
    reasons = Counter(str(record.get("reason", "")) for record in completed)
    errors = Counter(
        str(record.get("error_type", "unknown"))
        for record in records
        if record.get("evaluation_status") == "error"
    )
    mission_successes = sum(
        1 for record in completed if record.get("reason") == "mission_complete"
    )
    return {
        "episodes_requested": len(records),
        "episodes_complete": len(completed),
        "episodes_error": len(records) - len(completed),
        "mission_success_count": mission_successes,
        "mission_success_rate": (
            mission_successes / len(completed) if completed else None
        ),
        "termination_reasons": dict(sorted(reasons.items())),
        "error_types": dict(sorted(errors.items())),
        "episode_reward": _metric_summary(completed, "episode_reward"),
        "episode_steps": _metric_summary(completed, "episode_steps"),
        "mission_time_s": _metric_summary(completed, "bo_mission_time_s"),
        "progress_ratio": _metric_summary(completed, "bo_progress_ratio"),
        "final_distance_to_wp_m": _metric_summary(
            completed, "bo_final_distance_to_wp_m"
        ),
        "max_abs_roll_deg": _metric_summary(completed, "bo_max_abs_roll_deg"),
        "rudder_abs_saturation_ratio": _metric_summary(
            completed, "rudder_abs_saturation_ratio"
        ),
        "sail_abs_saturation_ratio": _metric_summary(
            completed, "sail_abs_saturation_ratio"
        ),
    }


def _paired_metric(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    key: str,
) -> dict[str, float | int | None]:
    deltas: list[float] = []
    for zero_record, policy_record in pairs:
        zero_value = zero_record.get(key)
        policy_value = policy_record.get(key)
        if isinstance(zero_value, (int, float)) and isinstance(
            policy_value, (int, float)
        ):
            deltas.append(float(policy_value) - float(zero_value))
    if not deltas:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(deltas),
        "mean": fmean(deltas),
        "std": pstdev(deltas),
        "min": min(deltas),
        "max": max(deltas),
    }


def summarize_evaluations(records: list[dict[str, Any]]) -> dict[str, Any]:
    controllers = sorted({str(record["controller"]) for record in records})
    controller_summaries = {
        controller: summarize_controller_records(
            [record for record in records if record["controller"] == controller]
        )
        for controller in controllers
    }

    complete_by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("evaluation_status") != "complete":
            continue
        pair_index = int(record["pair_index"])
        complete_by_pair.setdefault(pair_index, {})[
            str(record["controller"])
        ] = record
    pairs = [
        (entries["zero"], entries["policy"])
        for entries in complete_by_pair.values()
        if "zero" in entries and "policy" in entries
    ]
    paired_summary: dict[str, Any] | None = None
    if "zero" in controllers and "policy" in controllers:
        paired_summary = {
            "definition": "policy_minus_zero",
            "complete_pair_count": len(pairs),
            "episode_reward_delta": _paired_metric(pairs, "episode_reward"),
            "episode_steps_delta": _paired_metric(pairs, "episode_steps"),
            "mission_time_s_delta": _paired_metric(pairs, "bo_mission_time_s"),
            "progress_ratio_delta": _paired_metric(pairs, "bo_progress_ratio"),
            "final_distance_to_wp_m_delta": _paired_metric(
                pairs, "bo_final_distance_to_wp_m"
            ),
            "max_abs_roll_deg_delta": _paired_metric(
                pairs, "bo_max_abs_roll_deg"
            ),
            "policy_better_reward_count": sum(
                1
                for zero_record, policy_record in pairs
                if policy_record["episode_reward"] > zero_record["episode_reward"]
            ),
            "policy_faster_mission_count": sum(
                1
                for zero_record, policy_record in pairs
                if isinstance(zero_record.get("bo_mission_time_s"), (int, float))
                and isinstance(policy_record.get("bo_mission_time_s"), (int, float))
                and policy_record["bo_mission_time_s"]
                < zero_record["bo_mission_time_s"]
            ),
        }

    error_count = sum(
        1 for record in records if record.get("evaluation_status") == "error"
    )
    return {
        "status": "complete" if error_count == 0 else "partial",
        "record_count": len(records),
        "error_count": error_count,
        "controllers": controller_summaries,
        "paired": paired_summary,
    }
