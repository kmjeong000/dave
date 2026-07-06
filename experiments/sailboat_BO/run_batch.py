from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

try:
    from .common import DEFAULT_PARAMS_PATH, DEFAULT_RESULTS_DIR, DEFAULT_SCENARIO_PATH, load_yaml
except ImportError:
    from common import DEFAULT_PARAMS_PATH, DEFAULT_RESULTS_DIR, DEFAULT_SCENARIO_PATH, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of sailboat BO scenarios")
    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO_PATH),
        help="Path to scenario.yaml",
    )
    parser.add_argument(
        "--params-file",
        default=str(DEFAULT_PARAMS_PATH),
        help="Path to JSON parameter file used for every trial",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "eval"],
        default="all",
        help="Scenario split to run",
    )
    parser.add_argument(
        "--scenario-ids",
        nargs="*",
        help="Optional explicit scenario ids. If omitted, scenarios are selected by --split.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Override repeats_per_scenario from scenario.yaml",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Shared results directory",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["local", "docker-exec"],
        help="Forwarded to run_trial.py. Use docker-exec to launch each scenario inside an already-running container.",
    )
    parser.add_argument(
        "--docker-container",
        help="Forwarded to run_trial.py when --execution-backend docker-exec is used.",
    )
    parser.add_argument(
        "--container-repo-root",
        help="Forwarded to run_trial.py when --execution-backend docker-exec is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run through to each trial",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining scenarios even if one fails",
    )
    return parser.parse_args()


def select_scenarios(config: dict[str, Any], split: str, scenario_ids: list[str] | None) -> list[dict[str, Any]]:
    scenarios = config.get("scenarios", [])
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [scenario for scenario in scenarios if scenario.get("id") in wanted]
        missing = wanted - {scenario.get("id") for scenario in selected}
        if missing:
            raise KeyError(f"Unknown scenario ids: {', '.join(sorted(missing))}")
        return selected
    if split == "all":
        return list(scenarios)
    return [scenario for scenario in scenarios if scenario.get("split") == split]


def build_trial_command(
    *,
    python_executable: str,
    scenario_file: str,
    scenario_id: str,
    params_file: str,
    repeat_idx: int,
    results_dir: str,
    execution_backend: str | None,
    docker_container: str | None,
    container_repo_root: str | None,
    dry_run: bool,
) -> list[str]:
    cmd = [
        python_executable,
        "experiments/sailboat_BO/run_trial.py",
        "--scenario",
        scenario_file,
        "--scenario-id",
        scenario_id,
        "--params-file",
        params_file,
        "--repeat-idx",
        str(repeat_idx),
        "--results-dir",
        results_dir,
    ]
    if execution_backend:
        cmd.extend(["--execution-backend", execution_backend])
    if docker_container:
        cmd.extend(["--docker-container", docker_container])
    if container_repo_root:
        cmd.extend(["--container-repo-root", container_repo_root])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> int:
    args = parse_args()
    config = load_yaml(args.scenario)
    selected_scenarios = select_scenarios(config, args.split, args.scenario_ids)
    repeats = args.repeats if args.repeats is not None else int(config["study"].get("repeats_per_scenario", 1))

    if not selected_scenarios:
        raise SystemExit("No scenarios selected")

    failures = 0
    for scenario in selected_scenarios:
        scenario_id = scenario["id"]
        for repeat_idx in range(repeats):
            cmd = build_trial_command(
                python_executable=sys.executable,
                scenario_file=args.scenario,
                scenario_id=scenario_id,
                params_file=args.params_file,
                repeat_idx=repeat_idx,
                results_dir=args.results_dir,
                execution_backend=args.execution_backend,
                docker_container=args.docker_container,
                container_repo_root=args.container_repo_root,
                dry_run=args.dry_run,
            )
            print("running:", " ".join(cmd))
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                failures += 1
                print(f"trial failed: scenario={scenario_id} repeat={repeat_idx} rc={completed.returncode}")
                if not args.keep_going:
                    return completed.returncode

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
