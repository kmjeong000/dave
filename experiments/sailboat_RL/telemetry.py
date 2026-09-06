from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping


STEP_TELEMETRY_COLUMNS = (
    "record_type",
    "trial_id",
    "episode_index",
    "step",
    "sim_time_s",
    "action_rudder_normalized",
    "action_sail_normalized",
    "requested_residual_rudder_rad",
    "requested_residual_sail_rad",
    "residual_rudder_rad",
    "residual_sail_rad",
    "base_rudder_rad",
    "base_sail_rad",
    "final_rudder_rad",
    "final_sail_rad",
    "speed_mps",
    "roll_deg",
    "distance_to_waypoint_m",
    "cross_track_error_m",
    "waypoint_index",
    "waypoint_count",
    "reward",
    "reward_progress",
    "reward_cross_track",
    "reward_roll",
    "reward_residual",
    "reward_residual_change",
    "reward_waypoint",
    "reward_mission",
    "reward_terminal_accuracy",
    "reward_excessive_roll",
    "terminated",
    "truncated",
    "reason",
)

_FLOAT_COLUMNS = frozenset(
    {
        "sim_time_s",
        "action_rudder_normalized",
        "action_sail_normalized",
        "requested_residual_rudder_rad",
        "requested_residual_sail_rad",
        "residual_rudder_rad",
        "residual_sail_rad",
        "base_rudder_rad",
        "base_sail_rad",
        "final_rudder_rad",
        "final_sail_rad",
        "speed_mps",
        "roll_deg",
        "distance_to_waypoint_m",
        "cross_track_error_m",
        "reward",
        "reward_progress",
        "reward_cross_track",
        "reward_roll",
        "reward_residual",
        "reward_residual_change",
        "reward_waypoint",
        "reward_mission",
        "reward_terminal_accuracy",
        "reward_excessive_roll",
    }
)


class StepTelemetryWriter:
    """Durably write one aligned residual-RL record per environment step.

    The newest row is held until the following step or episode finalization.
    This lets a driver-limited ``environment_close`` be represented on its
    final control step without inventing a duplicate, non-physical step.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=STEP_TELEMETRY_COLUMNS,
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._stream.flush()
        self._pending: dict[str, Any] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _validate(self, row: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(row)
        missing = set(STEP_TELEMETRY_COLUMNS) - set(values)
        extra = set(values) - set(STEP_TELEMETRY_COLUMNS)
        if missing or extra:
            raise ValueError(
                "invalid RL step telemetry schema: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name in _FLOAT_COLUMNS:
            if not math.isfinite(float(values[name])):
                raise ValueError(f"non-finite RL telemetry value: {name}")
        return values

    def _write_pending(self) -> None:
        if self._pending is None:
            return
        self._writer.writerow(self._pending)
        self._stream.flush()
        self._pending = None

    def record(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot write closed RL step telemetry")
        values = self._validate(row)
        self._write_pending()
        self._pending = values
        if bool(values["terminated"]) or bool(values["truncated"]):
            self._write_pending()
            self.close()

    def finalize(self, reason: str) -> None:
        """Persist a non-terminal final step during reset/close cleanup."""

        if self._closed:
            return
        if self._pending is not None:
            self._pending["truncated"] = True
            self._pending["reason"] = str(reason)
            self._write_pending()
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._write_pending()
        self._stream.close()
        self._closed = True
