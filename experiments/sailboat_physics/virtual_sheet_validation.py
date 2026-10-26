"""Prospective Layer 2 validation for the reduced-order virtual sheet.

Framework ``v1`` is preserved for reproducibility: every excursion above the
soft containment reference must show an inward boom reversal before recovery.

Framework ``v2`` recognizes two bounded recovery mechanisms:

* the boom reverses inward; or
* a contract-valid, immediate sheet-out release expands the commanded
  allowance and returns penetration to the nominal region.

The caller must select a framework version explicitly.  This prevents an old
trial from being silently reclassified when a new framework is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


FRAMEWORK_V1 = "v1"
FRAMEWORK_V2 = "v2"
FRAMEWORK_VERSIONS = (FRAMEWORK_V1, FRAMEWORK_V2)

BOUNDED_TRANSIENT = "BOUNDED TRANSIENT"
DEGRADED = "DEGRADED"
PROBLEMATIC = "PROBLEMATIC"
OPEN = "OPEN"

INWARD_REVERSAL = "INWARD_REVERSAL"
VALID_SHEET_OUT_RELEASE = "VALID_SHEET_OUT_RELEASE"
RIGHT_CENSORED = "RIGHT_CENSORED"
NO_VALID_RECOVERY_MECHANISM = "NO_VALID_RECOVERY_MECHANISM"

DEFAULT_REFERENCE_RAD = math.radians(15.0)
DEFAULT_MAXIMUM_ALLOWANCE_RAD = 0.7854
DEFAULT_TOLERANCE_RAD = 1.0e-7

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class VirtualSheetSample:
    """One physics-tick virtual-sheet telemetry sample, in SI units."""

    sim_time_s: float
    raw_allowance_rad: float
    effective_allowance_rad: float
    allowance_lag_rad: float
    boom_angle_rad: float
    boom_velocity_rad_s: float
    penetration_rad: float
    requested_torque_nm: float
    applied_torque_nm: float
    sheet_out_release: bool
    tightening: bool
    torque_capped: bool

    @property
    def outward_velocity_rad_s(self) -> float:
        side = -1.0 if self.boom_angle_rad < 0.0 else 1.0
        return side * self.boom_velocity_rad_s


@dataclass(frozen=True)
class ComplianceEpisode:
    start_time_s: float
    end_time_s: float | None
    duration_s: float
    peak_time_s: float
    peak_penetration_rad: float
    recovery_time_s: float | None
    recovery_mechanism: str
    inward_reversal: bool
    valid_sheet_out_release: bool


@dataclass(frozen=True)
class ComplianceEvaluation:
    framework_version: str
    prospective_only: bool
    classification: str
    reference_rad: float
    sample_count: int
    max_penetration_rad: float
    total_time_above_reference_s: float
    longest_time_above_reference_s: float
    excess_integral_rad_s: float
    episode_count: int
    inward_reversal_count: int
    sheet_out_recovery_count: int
    unrecovered_episode_count: int
    recovery_time_median_s: float | None
    recovery_time_max_s: float | None
    repeated_peak_growth: bool
    cap_total_s: float
    cap_longest_s: float
    episodes: tuple[ComplianceEpisode, ...]


def _number(line: str, name: str) -> float | None:
    match = re.search(
        rf"(?:^|\s){re.escape(name)}=({_NUMBER})(?=\s|$)", line
    )
    return float(match.group(1)) if match else None


def _boolean(line: str, name: str) -> bool | None:
    match = re.search(
        rf"(?:^|\s){re.escape(name)}=(true|false)(?=\s|$)", line
    )
    return None if match is None else match.group(1) == "true"


def parse_virtual_sheet_logs(paths: Iterable[Path]) -> list[VirtualSheetSample]:
    """Parse physics-rate telemetry from Gazebo launch logs."""

    samples: list[VirtualSheetSample] = []

    for path in paths:
        if not path.exists():
            continue

        with path.open(encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                line = _ANSI.sub("", raw_line)
                if "[VirtualSheetActuator][physics]" not in line:
                    continue

                values = {
                    "sim_time_s": _number(line, "simTimeS"),
                    "raw_allowance_rad": _number(line, "rawAllowanceRad"),
                    "effective_allowance_rad": _number(
                        line, "effectiveAllowanceRad"
                    ),
                    "allowance_lag_rad": _number(line, "allowanceLagRad"),
                    "boom_angle_rad": _number(line, "boomAngleRad"),
                    "boom_velocity_rad_s": _number(
                        line, "boomVelocityRadS"
                    ),
                    "penetration_rad": _number(
                        line, "effectivePenetrationRad"
                    ),
                    "requested_torque_nm": _number(
                        line, "requestedTorqueNm"
                    ),
                    "applied_torque_nm": _number(line, "appliedTorqueNm"),
                    "sheet_out_release": _boolean(
                        line, "allowanceSheetOutRelease"
                    ),
                    "tightening": _boolean(line, "allowanceTightening"),
                    "torque_capped": _boolean(line, "torqueCapped"),
                }

                if all(value is not None for value in values.values()):
                    samples.append(VirtualSheetSample(**values))

    samples.sort(key=lambda sample: sample.sim_time_s)
    return samples


def _active_ranges(mask: Sequence[bool]) -> list[tuple[int, int, int | None]]:
    ranges: list[tuple[int, int, int | None]] = []
    start: int | None = None

    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            ranges.append((start, index - 1, index))
            start = None

    if start is not None:
        ranges.append((start, len(mask) - 1, None))

    return ranges


def _longest_duration(
    samples: Sequence[VirtualSheetSample],
    ranges: Sequence[tuple[int, int, int | None]],
) -> float:
    durations = []
    for start, end, recovery in ranges:
        end_time = (
            samples[recovery].sim_time_s
            if recovery is not None
            else samples[end].sim_time_s
        )
        durations.append(end_time - samples[start].sim_time_s)
    return max(durations, default=0.0)


def _valid_sheet_out_recovery(
    previous: VirtualSheetSample,
    recovered: VirtualSheetSample,
    *,
    reference_rad: float,
    maximum_allowance_rad: float,
    tolerance_rad: float,
) -> bool:
    """Return whether a boundary expansion is a valid unilateral-sheet release."""

    finite = all(
        math.isfinite(value)
        for value in (
            recovered.raw_allowance_rad,
            recovered.effective_allowance_rad,
            recovered.allowance_lag_rad,
            recovered.boom_angle_rad,
            recovered.boom_velocity_rad_s,
            recovered.penetration_rad,
            recovered.requested_torque_nm,
            recovered.applied_torque_nm,
        )
    )
    allowance_contract = (
        recovered.raw_allowance_rad >= -tolerance_rad
        and recovered.raw_allowance_rad
        <= maximum_allowance_rad + tolerance_rad
        and recovered.effective_allowance_rad + tolerance_rad
        >= recovered.raw_allowance_rad
        and recovered.effective_allowance_rad
        <= maximum_allowance_rad + tolerance_rad
        and abs(
            recovered.effective_allowance_rad - recovered.raw_allowance_rad
        )
        <= tolerance_rad
        and abs(
            (
                recovered.effective_allowance_rad
                - recovered.raw_allowance_rad
            )
            - recovered.allowance_lag_rad
        )
        <= tolerance_rad
    )
    boundary_expanded = (
        recovered.effective_allowance_rad
        > previous.effective_allowance_rad + tolerance_rad
    )
    returned_to_nominal = (
        previous.penetration_rad > reference_rad
        and recovered.penetration_rad <= reference_rad
    )

    return (
        finite
        and recovered.sheet_out_release
        and not recovered.tightening
        and allowance_contract
        and boundary_expanded
        and returned_to_nominal
    )


def evaluate_compliance(
    samples: Sequence[VirtualSheetSample],
    *,
    framework_version: str,
    reference_rad: float = DEFAULT_REFERENCE_RAD,
    maximum_allowance_rad: float = DEFAULT_MAXIMUM_ALLOWANCE_RAD,
    tolerance_rad: float = DEFAULT_TOLERANCE_RAD,
) -> ComplianceEvaluation:
    """Evaluate Layer 2 using an explicitly selected frozen framework."""

    if framework_version not in FRAMEWORK_VERSIONS:
        raise ValueError(
            f"framework_version must be one of {FRAMEWORK_VERSIONS}"
        )
    if not samples:
        raise ValueError("at least one telemetry sample is required")
    if any(
        later.sim_time_s <= earlier.sim_time_s
        for earlier, later in zip(samples, samples[1:])
    ):
        raise ValueError("sample simulation times must be strictly increasing")

    dts = [
        later.sim_time_s - earlier.sim_time_s
        for earlier, later in zip(samples, samples[1:])
    ]
    above_mask = [sample.penetration_rad > reference_rad for sample in samples]
    capped_mask = [sample.torque_capped for sample in samples]
    ranges = _active_ranges(above_mask)
    cap_ranges = _active_ranges(capped_mask)

    total_time = sum(
        dt
        for sample, dt in zip(samples, dts)
        if sample.penetration_rad > reference_rad
    )
    excess_integral = sum(
        dt * (sample.penetration_rad - reference_rad)
        for sample, dt in zip(samples, dts)
        if sample.penetration_rad > reference_rad
    )
    cap_total = sum(
        dt
        for sample, dt in zip(samples, dts)
        if sample.torque_capped
    )

    episodes: list[ComplianceEpisode] = []
    peaks: list[float] = []

    for start, end, recovery in ranges:
        peak = max(
            range(start, end + 1),
            key=lambda index: samples[index].penetration_rad,
        )
        peaks.append(samples[peak].penetration_rad)

        search_end = recovery if recovery is not None else end
        inward_reversal = any(
            samples[index].outward_velocity_rad_s < -tolerance_rad
            for index in range(peak, search_end + 1)
        )
        sheet_out_recovery = (
            recovery is not None
            and _valid_sheet_out_recovery(
                samples[end],
                samples[recovery],
                reference_rad=reference_rad,
                maximum_allowance_rad=maximum_allowance_rad,
                tolerance_rad=tolerance_rad,
            )
        )

        if recovery is None:
            mechanism = RIGHT_CENSORED
            end_time = None
            recovery_time = None
            duration = samples[end].sim_time_s - samples[start].sim_time_s
        else:
            end_time = samples[recovery].sim_time_s
            recovery_time = end_time - samples[peak].sim_time_s
            duration = end_time - samples[start].sim_time_s

            if inward_reversal:
                mechanism = INWARD_REVERSAL
            elif framework_version == FRAMEWORK_V2 and sheet_out_recovery:
                mechanism = VALID_SHEET_OUT_RELEASE
            else:
                mechanism = NO_VALID_RECOVERY_MECHANISM

        episodes.append(
            ComplianceEpisode(
                start_time_s=samples[start].sim_time_s,
                end_time_s=end_time,
                duration_s=duration,
                peak_time_s=samples[peak].sim_time_s,
                peak_penetration_rad=samples[peak].penetration_rad,
                recovery_time_s=recovery_time,
                recovery_mechanism=mechanism,
                inward_reversal=inward_reversal,
                valid_sheet_out_release=sheet_out_recovery,
            )
        )

    repeated_peak_growth = (
        len(peaks) >= 3
        and all(
            later > earlier + tolerance_rad
            for earlier, later in zip(peaks, peaks[1:])
        )
    )
    unrecovered_count = sum(
        episode.recovery_mechanism == RIGHT_CENSORED
        for episode in episodes
    )
    invalid_recovery_count = sum(
        episode.recovery_mechanism == NO_VALID_RECOVERY_MECHANISM
        for episode in episodes
    )

    if unrecovered_count:
        classification = OPEN
    elif repeated_peak_growth:
        classification = PROBLEMATIC
    elif invalid_recovery_count:
        classification = DEGRADED
    else:
        classification = BOUNDED_TRANSIENT

    recovery_times = [
        episode.recovery_time_s
        for episode in episodes
        if episode.recovery_time_s is not None
    ]

    return ComplianceEvaluation(
        framework_version=framework_version,
        prospective_only=True,
        classification=classification,
        reference_rad=reference_rad,
        sample_count=len(samples),
        max_penetration_rad=max(
            sample.penetration_rad for sample in samples
        ),
        total_time_above_reference_s=total_time,
        longest_time_above_reference_s=_longest_duration(samples, ranges),
        excess_integral_rad_s=excess_integral,
        episode_count=len(episodes),
        inward_reversal_count=sum(
            episode.inward_reversal for episode in episodes
        ),
        sheet_out_recovery_count=sum(
            episode.recovery_mechanism == VALID_SHEET_OUT_RELEASE
            for episode in episodes
        ),
        unrecovered_episode_count=unrecovered_count,
        recovery_time_median_s=(
            median(recovery_times) if recovery_times else None
        ),
        recovery_time_max_s=(max(recovery_times) if recovery_times else None),
        repeated_peak_growth=repeated_peak_growth,
        cap_total_s=cap_total,
        cap_longest_s=_longest_duration(samples, cap_ranges),
        episodes=tuple(episodes),
    )


def _serializable(evaluation: ComplianceEvaluation) -> dict:
    data = asdict(evaluation)
    data["reference_deg"] = math.degrees(evaluation.reference_rad)
    data["max_penetration_deg"] = math.degrees(
        evaluation.max_penetration_rad
    )
    data["excess_integral_deg_s"] = math.degrees(
        evaluation.excess_integral_rad_s
    )
    for episode in data["episodes"]:
        episode["peak_penetration_deg"] = math.degrees(
            episode["peak_penetration_rad"]
        )
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument(
        "--framework-version",
        choices=FRAMEWORK_VERSIONS,
        required=True,
        help="Explicit prospective framework; no implicit version default.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    log_paths = sorted((args.trial_dir / "logs").glob("launch.*.log"))
    samples = parse_virtual_sheet_logs(log_paths)
    evaluation = evaluate_compliance(
        samples,
        framework_version=args.framework_version,
    )
    output = _serializable(evaluation)

    print(json.dumps(output, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0 if evaluation.classification == BOUNDED_TRANSIENT else 1


if __name__ == "__main__":
    raise SystemExit(main())
