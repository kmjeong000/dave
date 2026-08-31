#!/usr/bin/env python3
"""Summarize the sail wrench recorded by an existing BO trial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from .tacking_wrench import parse_sail_wrench, summarize_sail_wrench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether SailLiftDragSystem applies opposite roll moments "
            "on the two sustained boom sides"
        )
    )
    parser.add_argument(
        "--trial-dir",
        required=True,
        help="BO trial directory containing logs/launch.stdout.log",
    )
    parser.add_argument("--boom-deadband-deg", type=float, default=5.0)
    parser.add_argument("--minimum-force-n", type=float, default=0.1)
    parser.add_argument("--minimum-samples-per-side", type=int, default=3)
    parser.add_argument("--roll-moment-deadband-nm", type=float, default=0.5)
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
    trial_dir = Path(args.trial_dir).expanduser().resolve()
    log_paths = [
        trial_dir / "logs/launch.stdout.log",
        trial_dir / "logs/launch.stderr.log",
    ]
    text = _read_logs(log_paths)
    if not text:
        raise FileNotFoundError(
            f"no launch logs found below {trial_dir / 'logs'}"
        )

    samples = parse_sail_wrench(text)
    summary = summarize_sail_wrench(
        samples,
        boom_deadband_deg=args.boom_deadband_deg,
        minimum_force_n=args.minimum_force_n,
        minimum_samples_per_side=args.minimum_samples_per_side,
        roll_moment_deadband_nm=args.roll_moment_deadband_nm,
        wrench_math_tolerance_nm=args.wrench_math_tolerance_nm,
    )

    output_dir = trial_dir / "physics_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "sail_wrench_samples.csv"
    summary_path = output_dir / "sail_wrench_summary.json"

    if samples:
        with samples_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0].as_dict()))
            writer.writeheader()
            writer.writerows(sample.as_dict() for sample in samples)
    else:
        samples_path.write_text("", encoding="utf-8")

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"trial_dir: {trial_dir}")
    print(f"sample_csv: {samples_path}")
    print(f"summary_json: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not samples:
        print(
            "tacking_wrench_diagnostic: NO_SAMPLES "
            "(rebuild the image and run a new trial with the new plugin log format)"
        )
        return 2 if args.require_pass else 0
    if summary["passed"]:
        print("tacking_wrench_diagnostic: PASSED")
        return 0
    print("tacking_wrench_diagnostic: FAILED")
    return 2 if args.require_pass else 0


if __name__ == "__main__":
    raise SystemExit(main())
