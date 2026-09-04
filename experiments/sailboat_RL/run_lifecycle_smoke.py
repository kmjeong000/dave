from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from experiments.sailboat_BO.run_trial import (
    DEFAULT_CONTAINER_REPO_ROOT,
    get_repo_root,
    resolve_repo_path,
)
from experiments.sailboat_RL.runtime import (
    LifecycleEnvironmentSettings,
    build_lifecycle_environment,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automatically started and cleaned Gymnasium sailboat episodes"
    )
    parser.add_argument(
        "--scenario",
        default="experiments/sailboat_BO/scenario.yaml",
    )
    parser.add_argument("--scenario-id", default="eval_long_oblique")
    parser.add_argument(
        "--params-file",
        default="experiments/sailboat_BO/optimized_baseline_params.json",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/sailboat_RL/results/lifecycle_smoke",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=20,
        help="Driver smoke limit; 0 lets each episode run to an environment terminal state",
    )
    parser.add_argument(
        "--action",
        default="[0.0, 0.0]",
        help="Fixed normalized JSON action [rudder, sail] in [-1, 1]",
    )
    parser.add_argument("--control-period-s", type=float, default=0.5)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=45.0)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--execution-backend",
        choices=["local", "docker-exec"],
        default="local",
    )
    parser.add_argument("--docker-container")
    parser.add_argument(
        "--container-repo-root",
        default=DEFAULT_CONTAINER_REPO_ROOT,
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.max_steps_per_episode < 0:
        raise SystemExit("--max-steps-per-episode must be non-negative")
    if args.control_period_s <= 0.0:
        raise SystemExit("--control-period-s must be positive")

    action = np.asarray(json.loads(args.action), dtype=np.float32)
    if action.shape != (2,):
        raise SystemExit("--action must be a JSON two-vector")

    repo_root = get_repo_root()
    scenario_path = resolve_repo_path(repo_root, args.scenario)
    params_file = resolve_repo_path(repo_root, args.params_file)
    results_dir = resolve_repo_path(repo_root, args.results_dir)
    env, backend = build_lifecycle_environment(
        LifecycleEnvironmentSettings(
            repo_root=repo_root,
            scenario_path=scenario_path,
            scenario_id=args.scenario_id,
            params_file=params_file,
            results_dir=results_dir,
            execution_backend=args.execution_backend,
            docker_container=args.docker_container,
            container_repo_root=(
                args.container_repo_root
                if args.execution_backend == "docker-exec"
                else None
            ),
            startup_timeout_s=float(args.startup_timeout_s),
            shutdown_timeout_s=float(args.shutdown_timeout_s),
            repeat_start=int(args.repeat_start),
            control_period_s=float(args.control_period_s),
        ),
    )

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            print(
                "episode_reset:",
                {
                    "episode": episode,
                    "trial_id": backend.current_trial_id,
                    "summary_json": backend.current_summary_json,
                    "observation": observation.tolist(),
                    "info": info,
                },
            )
            step_index = 0
            while True:
                observation, reward, terminated, truncated, info = env.step(action)
                if step_index % max(1, args.log_every) == 0 or terminated or truncated:
                    print(
                        "episode_step:",
                        {
                            "episode": episode,
                            "step": step_index,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "reason": info["reason"],
                            "distance_to_waypoint_m": info[
                                "distance_to_waypoint_m"
                            ],
                            "waypoint_index": info["waypoint_index"],
                            "sim_time_s": info["sim_time_s"],
                        },
                    )
                step_index += 1
                if terminated or truncated:
                    break
                if (
                    args.max_steps_per_episode > 0
                    and step_index >= args.max_steps_per_episode
                ):
                    print(
                        "episode_driver_limit:",
                        {"episode": episode, "steps": step_index},
                    )
                    break
    finally:
        env.close()

    if backend.last_cleanup_error:
        print("cleanup_warning:", backend.last_cleanup_error)
        return 1
    print("lifecycle_smoke: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
