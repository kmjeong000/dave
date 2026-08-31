#!/usr/bin/env python3
"""Align BO commands with the measured rudder angle and hydrodynamic wrench."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from .rudder_response import (
    align_rudder_samples,
    build_rudder_command_samples,
    parse_rudder_wrench,
    summarize_rudder_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the rudder command -> actual angle -> hydrodynamic yaw "
            "moment sign chain in one or more completed BO trials"
        )
    )
    parser.add_argument(
        "--trial-dir",
        action="append",
        required=True,
        help="BO trial directory; repeat this option to combine trials",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Defaults to <trial>/physics_diagnostics for "
            "one trial; required when combining multiple trials."
        ),
    )
    parser.add_argument("--maximum-time-delta-s", type=float, default=0.75)
    parser.add_argument("--command-deadband-deg", type=float, default=5.0)
    parser.add_argument("--actual-angle-deadband-deg", type=float, default=2.0)
    parser.add_argument("--minimum-speed-mps", type=float, default=0.25)
    parser.add_argument("--minimum-force-n", type=float, default=0.1)
    parser.add_argument("--minimum-samples-per-side", type=int, default=3)
    parser.add_argument("--minimum-sign-agreement-ratio", type=float, default=0.8)
    parser.add_argument("--yaw-moment-deadband-nm", type=float, default=0.5)
    parser.add_argument("--wrench-math-tolerance-nm", type=float, default=0.05)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 if any diagnostic gate fails",
    )
    return parser.parse_args()


def _read_logs(paths: Sequence[Path]) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in paths
        if path.exists()
    )


def main() -> int:
    args = parse_args()
    trial_dirs = [Path(path).expanduser().resolve() for path in args.trial_dir]
    if len(trial_dirs) > 1 and not args.output_dir:
        raise SystemExit("--output-dir is required with multiple --trial-dir values")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else trial_dirs[0] / "physics_diagnostics"
    )

    aligned = []
    per_trial: list[dict[str, object]] = []
    for trial_dir in trial_dirs:
        raw_path = trial_dir / "raw/samples.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        text = _read_logs(
            [
                trial_dir / "logs/launch.stdout.log",
                trial_dir / "logs/launch.stderr.log",
            ]
        )
        if not text:
            raise FileNotFoundError(
                f"no launch logs found below {trial_dir / 'logs'}"
            )
        wrench_samples = parse_rudder_wrench(text)
        with raw_path.open(encoding="utf-8", newline="") as stream:
            command_samples = build_rudder_command_samples(
                list(csv.DictReader(stream))
            )
        trial_aligned = align_rudder_samples(
            trial_dir.name,
            wrench_samples,
            command_samples,
            maximum_time_delta_s=args.maximum_time_delta_s,
        )
        aligned.extend(trial_aligned)
        per_trial.append(
            {
                "trial_id": trial_dir.name,
                "wrench_sample_count": len(wrench_samples),
                "command_sample_count": len(command_samples),
                "aligned_sample_count": len(trial_aligned),
            }
        )

    summary = summarize_rudder_response(
        aligned,
        command_deadband_deg=args.command_deadband_deg,
        actual_angle_deadband_deg=args.actual_angle_deadband_deg,
        minimum_speed_mps=args.minimum_speed_mps,
        minimum_force_n=args.minimum_force_n,
        minimum_samples_per_side=args.minimum_samples_per_side,
        minimum_sign_agreement_ratio=args.minimum_sign_agreement_ratio,
        yaw_moment_deadband_nm=args.yaw_moment_deadband_nm,
        wrench_math_tolerance_nm=args.wrench_math_tolerance_nm,
    )
    summary["trials"] = per_trial

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "rudder_response_samples.csv"
    summary_path = output_dir / "rudder_response_summary.json"
    if aligned:
        with samples_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aligned[0].as_dict()))
            writer.writeheader()
            writer.writerows(sample.as_dict() for sample in aligned)
    else:
        samples_path.write_text("", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"sample_csv: {samples_path}")
    print(f"summary_json: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not aligned:
        print(
            "rudder_response_diagnostic: NO_SAMPLES "
            "(rebuild the image and run a new trial with the new plugin log format)"
        )
        return 2 if args.require_pass else 0
    if summary["passed"]:
        print("rudder_response_diagnostic: PASSED")
        return 0
    print("rudder_response_diagnostic: FAILED")
    return 2 if args.require_pass else 0


if __name__ == "__main__":
    raise SystemExit(main())
