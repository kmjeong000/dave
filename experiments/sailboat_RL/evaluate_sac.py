from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from experiments.sailboat_BO.run_trial import (
    DEFAULT_CONTAINER_REPO_ROOT,
    get_repo_root,
    resolve_repo_path,
)
from experiments.sailboat_RL.core import (
    REWARD_COMPONENT_KEYS,
    reward_component_info_key,
)
from experiments.sailboat_RL.evaluation import summarize_evaluations


ActionFunction = Callable[[np.ndarray], np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved SAC residual policy and optionally compare it "
            "with zero residual under paired trial conditions"
        )
    )
    parser.add_argument(
        "--model",
        help="Saved Stable-Baselines3 SAC .zip model",
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "policy", "zero"],
        default="compare",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--results-dir",
        help=(
            "New output directory. The default is a timestamped directory under "
            "experiments/sailboat_RL/results/evaluation."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="experiments/sailboat_BO/scenario.yaml",
    )
    parser.add_argument("--scenario-id", default="eval_long_oblique")
    parser.add_argument(
        "--params-file",
        default="experiments/sailboat_BO/optimized_baseline_params.json",
        help="Frozen BO controller parameters used by both controllers",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--repeat-start", type=int, default=1000)
    parser.add_argument("--control-period-s", type=float, default=0.5)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=45.0)
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
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample policy actions instead of deterministic SAC actions",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining paired trials after an episode error",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    return parser.parse_args(argv)


def validate_evaluation_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.mode in {"compare", "policy"} and not args.model:
        raise ValueError("--model is required for policy evaluation")
    if args.control_period_s <= 0.0:
        raise ValueError("--control-period-s must be positive")
    if args.startup_timeout_s <= 0.0:
        raise ValueError("--startup-timeout-s must be positive")
    if args.shutdown_timeout_s <= 0.0:
        raise ValueError("--shutdown-timeout-s must be positive")


def default_results_dir(repo_root: Path, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo_root
        / "experiments"
        / "sailboat_RL"
        / "results"
        / "evaluation"
        / f"{timestamp}_{os.getpid()}"
    )


def prepare_results_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"results path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"results directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _bo_metrics(summary_path: str) -> dict[str, Any]:
    if not summary_path:
        return {}
    path = Path(summary_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    metadata = data.get("metadata", {})
    metrics = data.get("metrics", {})
    constraints = data.get("constraints", {})
    return {
        "bo_status": metadata.get("status"),
        "bo_failure_reason": metadata.get("failure_reason"),
        "bo_mission_complete": constraints.get("constraint_mission_complete"),
        "bo_mission_time_s": metrics.get("mission_time_s"),
        "bo_progress_ratio": metrics.get("progress_ratio"),
        "bo_final_distance_to_wp_m": metrics.get("final_distance_to_wp_m"),
        "bo_max_abs_roll_deg": metrics.get("max_abs_roll_deg"),
        "bo_xte_rms_m": metrics.get("xte_rms_m"),
    }


def _finalize_episode_record(record: dict[str, Any]) -> None:
    """Refresh BO metrics and reject contradictory terminal outcomes."""

    if record.get("summary_json"):
        record.update(_bo_metrics(str(record["summary_json"])))
    if record.get("evaluation_status") != "complete":
        return

    bo_status = record.get("bo_status")
    gym_reason = str(record.get("reason", ""))
    error = ""
    if bo_status is None:
        error = "BO summary was not available after lifecycle cleanup"
    elif gym_reason == "mission_complete" and not (
        bo_status == "success" and record.get("bo_mission_complete") is True
    ):
        error = (
            "Gym reported mission_complete but BO finalized as "
            f"status={bo_status!r}, failure_reason="
            f"{record.get('bo_failure_reason')!r}, mission_complete="
            f"{record.get('bo_mission_complete')!r}"
        )
    elif bo_status == "failure" and gym_reason != str(
        record.get("bo_failure_reason", "")
    ):
        error = (
            f"Gym terminal reason {gym_reason!r} does not match BO failure "
            f"reason {record.get('bo_failure_reason')!r}"
        )

    if error:
        record.update(
            {
                "evaluation_status": "error",
                "error_type": "EpisodeFinalizationError",
                "error": error,
            }
        )


def run_episode(
    env: Any,
    backend: Any,
    action_function: ActionFunction,
    *,
    controller: str,
    pair_index: int,
    seed: int,
    repeat_idx: int,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    started_at = time.monotonic()
    observation, _reset_info = env.reset(
        seed=seed,
        options={"repeat_idx": repeat_idx},
    )
    episode_reward = 0.0
    episode_components = {
        name: 0.0 for name in REWARD_COMPONENT_KEYS
    }
    episode_steps = 0
    progress_saturation_count = 0
    progress_normalized_abs_sum = 0.0
    progress_normalized_abs_max = 0.0
    actions: list[np.ndarray] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.asarray(action_function(observation), dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(
                f"controller {controller} returned action shape {action.shape}"
            )
        observation, reward, terminated, truncated, final_info = env.step(action)
        step_components = final_info.get("reward_components", {})
        for name in REWARD_COMPONENT_KEYS:
            episode_components[name] += float(step_components[name])
        diagnostics = final_info.get("reward_diagnostics", {})
        progress_saturation_count += int(
            float(diagnostics.get("progress_saturated", 0.0)) > 0.5
        )
        progress_abs = abs(
            float(diagnostics.get("progress_normalized", 0.0))
        )
        progress_normalized_abs_sum += progress_abs
        progress_normalized_abs_max = max(
            progress_normalized_abs_max,
            progress_abs,
        )
        actions.append(action.copy())
        episode_reward += float(reward)
        episode_steps += 1

    action_array = np.stack(actions) if actions else np.zeros((0, 2))
    trial_id = str(backend.current_trial_id)
    summary_json = str(backend.current_summary_json)
    record: dict[str, Any] = {
        "evaluation_status": "complete",
        "controller": controller,
        "pair_index": pair_index,
        "seed": seed,
        "repeat_idx": repeat_idx,
        "trial_id": trial_id,
        "summary_json": summary_json,
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reason": str(final_info.get("reason", "")),
        "episode_sim_time_s": float(final_info.get("sim_time_s", 0.0))
        - float(_reset_info.get("sim_time_s", 0.0)),
        "final_distance_to_waypoint_m": final_info.get(
            "distance_to_waypoint_m"
        ),
        "rudder_action_abs_mean": (
            float(np.abs(action_array[:, 0]).mean()) if actions else 0.0
        ),
        "sail_action_abs_mean": (
            float(np.abs(action_array[:, 1]).mean()) if actions else 0.0
        ),
        "rudder_abs_saturation_ratio": (
            float((np.abs(action_array[:, 0]) >= saturation_threshold).mean())
            if actions
            else 0.0
        ),
        "sail_abs_saturation_ratio": (
            float((np.abs(action_array[:, 1]) >= saturation_threshold).mean())
            if actions
            else 0.0
        ),
        "elapsed_wall_s": time.monotonic() - started_at,
    }
    for name, value in episode_components.items():
        record[reward_component_info_key(name)] = float(value)
    component_sum = float(sum(episode_components.values()))
    record["reward_component_sum"] = component_sum
    record["reward_component_sum_error"] = component_sum - episode_reward
    sample_count = max(episode_steps, 1)
    record["progress_saturation_ratio"] = (
        progress_saturation_count / sample_count
    )
    record["progress_normalized_abs_mean"] = (
        progress_normalized_abs_sum / sample_count
    )
    record["progress_normalized_abs_max"] = progress_normalized_abs_max
    record.update(_bo_metrics(summary_json))
    return record


def _controller_order(mode: str, pair_index: int) -> list[str]:
    if mode == "zero":
        return ["zero"]
    if mode == "policy":
        return ["policy"]
    # Alternate the first controller to reduce time/order bias in paired runs.
    return ["zero", "policy"] if pair_index % 2 == 0 else ["policy", "zero"]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_evaluation_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        from stable_baselines3 import SAC
        from experiments.sailboat_RL.runtime import (
            LifecycleEnvironmentSettings,
            build_lifecycle_environment,
        )
    except ImportError as exc:
        raise SystemExit(
            "Stable-Baselines3 is not installed. Rebuild dave:sailboat-rl."
        ) from exc

    repo_root = get_repo_root()
    scenario_path = resolve_repo_path(repo_root, args.scenario)
    params_file = resolve_repo_path(repo_root, args.params_file)
    model_path = resolve_repo_path(repo_root, args.model) if args.model else None
    results_dir = (
        resolve_repo_path(repo_root, args.results_dir)
        if args.results_dir
        else default_results_dir(repo_root)
    )
    if model_path is not None and not model_path.exists():
        raise SystemExit(f"model does not exist: {model_path}")
    try:
        prepare_results_dir(results_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    model = (
        SAC.load(model_path, device=args.device) if model_path is not None else None
    )
    _write_json(
        results_dir / "evaluation_config.json",
        {
            "mode": args.mode,
            "model": str(model_path) if model_path is not None else None,
            "deterministic_policy": not args.stochastic_policy,
            "scenario": str(scenario_path),
            "scenario_id": args.scenario_id,
            "params_file": str(params_file),
            "episodes": args.episodes,
            "seed": args.seed,
            "repeat_start": args.repeat_start,
            "arguments": vars(args),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"results_dir: {results_dir}")

    def zero_action(_observation: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def policy_action(observation: np.ndarray) -> np.ndarray:
        if model is None:
            raise RuntimeError("policy model is not loaded")
        action, _state = model.predict(
            observation,
            deterministic=not args.stochastic_policy,
        )
        return np.asarray(action, dtype=np.float32)

    action_functions = {"zero": zero_action, "policy": policy_action}
    records: list[dict[str, Any]] = []
    exit_code = 0
    interrupted = False
    for pair_index in range(args.episodes):
        seed = args.seed + pair_index
        repeat_idx = args.repeat_start + pair_index
        for controller in _controller_order(args.mode, pair_index):
            env = None
            backend = None
            record: dict[str, Any] | None = None
            try:
                env, backend = build_lifecycle_environment(
                    LifecycleEnvironmentSettings(
                        repo_root=repo_root,
                        scenario_path=scenario_path,
                        scenario_id=args.scenario_id,
                        params_file=params_file,
                        results_dir=results_dir / controller / "trials",
                        control_period_s=float(args.control_period_s),
                        execution_backend=args.execution_backend,
                        docker_container=args.docker_container,
                        container_repo_root=(
                            args.container_repo_root
                            if args.execution_backend == "docker-exec"
                            else None
                        ),
                        startup_timeout_s=float(args.startup_timeout_s),
                        shutdown_timeout_s=float(args.shutdown_timeout_s),
                        repeat_start=repeat_idx,
                    )
                )
                record = run_episode(
                    env,
                    backend,
                    action_functions[controller],
                    controller=controller,
                    pair_index=pair_index,
                    seed=seed,
                    repeat_idx=repeat_idx,
                )
            except KeyboardInterrupt:
                interrupted = True
                exit_code = 130
                print("sac_evaluation: INTERRUPTED")
                break
            except Exception as exc:
                record = {
                    "evaluation_status": "error",
                    "controller": controller,
                    "pair_index": pair_index,
                    "seed": seed,
                    "repeat_idx": repeat_idx,
                    "trial_id": (
                        str(backend.current_trial_id) if backend is not None else ""
                    ),
                    "summary_json": (
                        str(backend.current_summary_json)
                        if backend is not None
                        else ""
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                exit_code = 1
                print("evaluation_error:", record)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        if record is None:
                            record = {
                                "controller": controller,
                                "pair_index": pair_index,
                                "seed": seed,
                                "repeat_idx": repeat_idx,
                            }
                        record.update(
                            {
                                "evaluation_status": "error",
                                "cleanup_error_type": type(exc).__name__,
                                "cleanup_error": str(exc),
                            }
                        )
                        exit_code = 1
                    if record is not None:
                        _finalize_episode_record(record)
                        if record.get("evaluation_status") == "error":
                            exit_code = 1

            if interrupted:
                break
            if record is None:
                raise RuntimeError("evaluation episode produced no record")
            records.append(record)
            _write_json(results_dir / "episodes.json", records)
            _write_records_csv(results_dir / "episodes.csv", records)
            _write_json(
                results_dir / "evaluation_summary.json",
                summarize_evaluations(records),
            )
            print(
                "evaluation_episode:",
                {
                    "controller": controller,
                    "pair_index": pair_index,
                    "status": record["evaluation_status"],
                    "reason": record.get("reason", ""),
                    "reward": record.get("episode_reward"),
                    "steps": record.get("episode_steps"),
                    "trial_id": record.get("trial_id", ""),
                },
            )
            if record["evaluation_status"] == "error" and not args.keep_going:
                break
        if interrupted or (exit_code and not args.keep_going):
            break

    summary = summarize_evaluations(records)
    if interrupted:
        summary["status"] = "interrupted"
    _write_json(results_dir / "episodes.json", records)
    _write_records_csv(results_dir / "episodes.csv", records)
    _write_json(results_dir / "evaluation_summary.json", summary)
    print(f"evaluation_summary: {results_dir / 'evaluation_summary.json'}")
    if exit_code == 0:
        print("sac_evaluation: PASSED")
    elif not interrupted:
        print("sac_evaluation: PARTIAL")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
