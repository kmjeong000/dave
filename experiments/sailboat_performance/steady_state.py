"""Sliding-window steady-state qualification for one polar cell."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from .core import circular_std_deg, linear_slope


@dataclass(frozen=True)
class SteadyStateConfig:
    window_s: float = 20.0
    qualification_hold_s: float = 10.0
    minimum_window_samples: int = 180
    minimum_valid_ratio: float = 0.95
    heading_rmse_max_deg: float = 5.0
    heading_circular_std_max_deg: float = 3.0
    speed_slope_max_mps2: float = 0.01
    speed_cv_max: float = 0.15
    speed_std_floor_mps: float = 0.03
    roll_slope_max_deg_s: float = 0.1
    roll_std_max_deg: float = 2.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SteadyStateConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "unknown steady_state fields: " + ", ".join(sorted(unknown))
            )
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.window_s <= 0.0 or self.qualification_hold_s < 0.0:
            raise ValueError("steady-state window must be positive and hold non-negative")
        if self.minimum_window_samples < 2:
            raise ValueError("minimum_window_samples must be at least 2")
        if not 0.0 < self.minimum_valid_ratio <= 1.0:
            raise ValueError("minimum_valid_ratio must be in (0, 1]")
        for name in (
            "heading_rmse_max_deg",
            "heading_circular_std_max_deg",
            "speed_slope_max_mps2",
            "speed_cv_max",
            "speed_std_floor_mps",
            "roll_slope_max_deg_s",
            "roll_std_max_deg",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class SteadyObservation:
    sim_time_s: float
    heading_error_deg: float
    heading_deg: float
    speed_mps: float
    roll_deg: float
    telemetry_valid: bool
    wind_settled: bool
    force_ramp_complete: bool
    tack_detected: bool = False


@dataclass(frozen=True)
class SteadyEvaluation:
    qualified: bool
    ready: bool
    reasons: tuple[str, ...]
    stable_for_s: float
    stats: dict[str, float | int | bool | None]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_window(
    observations: Sequence[SteadyObservation],
    config: SteadyStateConfig,
) -> tuple[bool, tuple[str, ...], dict[str, float | int | bool | None]]:
    if not observations:
        return False, ("insufficient_window",), {"sample_count": 0}

    duration_s = observations[-1].sim_time_s - observations[0].sim_time_s
    valid = [sample for sample in observations if sample.telemetry_valid]
    valid_ratio = len(valid) / len(observations)
    stats: dict[str, float | int | bool | None] = {
        "sample_count": len(observations),
        "valid_sample_count": len(valid),
        "valid_ratio": valid_ratio,
        "window_duration_s": duration_s,
    }
    reasons: list[str] = []
    if duration_s + 1e-9 < config.window_s:
        reasons.append("insufficient_window")
    if len(observations) < config.minimum_window_samples:
        reasons.append("insufficient_samples")
    if valid_ratio + 1e-12 < config.minimum_valid_ratio:
        reasons.append("telemetry_stale")
    if not observations[-1].wind_settled:
        reasons.append("wind_not_settled")
    if not observations[-1].force_ramp_complete:
        reasons.append("force_ramp_incomplete")
    if any(sample.tack_detected for sample in observations):
        reasons.append("tack_detected")

    if len(valid) < 2:
        reasons.append("insufficient_valid_samples")
        return False, tuple(dict.fromkeys(reasons)), stats

    heading_errors = [sample.heading_error_deg for sample in valid]
    headings = [sample.heading_deg for sample in valid]
    speeds = [sample.speed_mps for sample in valid]
    rolls = [sample.roll_deg for sample in valid]
    heading_rmse = math.sqrt(fmean(value * value for value in heading_errors))
    heading_std = circular_std_deg(headings)
    speed_mean = fmean(speeds)
    speed_std = pstdev(speeds)
    speed_cv = speed_std / abs(speed_mean) if abs(speed_mean) > 1e-6 else math.inf
    speed_slope = linear_slope(
        [(sample.sim_time_s, sample.speed_mps) for sample in valid]
    )
    roll_std = pstdev(rolls)
    roll_slope = linear_slope(
        [(sample.sim_time_s, sample.roll_deg) for sample in valid]
    )
    stats.update(
        {
            "heading_rmse_deg": heading_rmse,
            "heading_circular_std_deg": heading_std,
            "speed_mean_mps": speed_mean,
            "speed_std_mps": speed_std,
            "speed_cv": speed_cv,
            "speed_slope_mps2": speed_slope,
            "roll_mean_deg": fmean(rolls),
            "roll_std_deg": roll_std,
            "roll_slope_deg_s": roll_slope,
        }
    )

    if heading_rmse > config.heading_rmse_max_deg:
        reasons.append("heading_unstable")
    if heading_std > config.heading_circular_std_max_deg:
        reasons.append("heading_dispersion")
    if abs(speed_slope) > config.speed_slope_max_mps2:
        reasons.append("speed_trending")
    if speed_std > config.speed_std_floor_mps and speed_cv > config.speed_cv_max:
        reasons.append("speed_not_steady")
    if abs(roll_slope) > config.roll_slope_max_deg_s:
        reasons.append("roll_trending")
    if roll_std > config.roll_std_max_deg:
        reasons.append("roll_unstable")
    return not reasons, tuple(dict.fromkeys(reasons)), stats


class SteadyStateDetector:
    def __init__(self, config: SteadyStateConfig):
        config.validate()
        self.config = config
        self._observations: deque[SteadyObservation] = deque()
        self._qualified_since_s: float | None = None

    def update(self, observation: SteadyObservation) -> SteadyEvaluation:
        if self._observations and observation.sim_time_s <= self._observations[-1].sim_time_s:
            raise ValueError("steady-state observations must have increasing sim_time_s")
        self._observations.append(observation)
        cutoff = observation.sim_time_s - self.config.window_s
        # Retain the last sample on or before the cutoff.  At a finite sample
        # rate, removing every sample older than the exact cutoff leaves a
        # window just short of ``window_s`` forever (for example 19.9 s at
        # 10 Hz), so it could never qualify.
        while (
            len(self._observations) > 1
            and self._observations[1].sim_time_s <= cutoff
        ):
            self._observations.popleft()

        qualified, reasons, stats = _evaluate_window(
            list(self._observations), self.config
        )
        if qualified:
            if self._qualified_since_s is None:
                self._qualified_since_s = observation.sim_time_s
        else:
            self._qualified_since_s = None
        stable_for_s = (
            max(0.0, observation.sim_time_s - self._qualified_since_s)
            if self._qualified_since_s is not None
            else 0.0
        )
        ready = qualified and stable_for_s + 1e-9 >= self.config.qualification_hold_s
        return SteadyEvaluation(
            qualified=qualified,
            ready=ready,
            reasons=reasons,
            stable_for_s=stable_for_s,
            stats=stats,
        )
