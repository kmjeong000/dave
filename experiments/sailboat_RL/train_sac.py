from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from experiments.sailboat_BO.run_trial import (
    DEFAULT_CONTAINER_REPO_ROOT,
    get_repo_root,
    resolve_repo_path,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SAC policy for bounded residual sailboat control"
    )
    parser.add_argument(
        "--scenario",
        default="experiments/sailboat_BO/scenario.yaml",
    )
    parser.add_argument("--scenario-id", default="eval_long_oblique")
    parser.add_argument(
        "--params-file",
        default="experiments/sailboat_BO/verified_incumbent_params.json",
        help="Frozen BO controller parameters used as the base policy",
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "New output directory. The default is a timestamped directory under "
            "experiments/sailboat_RL/results/sac."
        ),
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--control-period-s", type=float, default=0.5)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=45.0)
    parser.add_argument("--repeat-start", type=int, default=0)
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
        "--skip-env-check",
        action="store_true",
        help="Skip Stable-Baselines3 check_env before training",
    )
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        help="Run the live Stable-Baselines3 environment check and exit",
    )
    return parser.parse_args(argv)


def validate_training_args(args: argparse.Namespace) -> None:
    if args.check_env_only and args.skip_env_check:
        raise ValueError(
            "--check-env-only cannot be combined with --skip-env-check"
        )
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative")
    if not args.check_env_only and args.learning_starts >= args.total_timesteps:
        raise ValueError(
            "--learning-starts must be smaller than --total-timesteps"
        )
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.batch_size > args.buffer_size:
        raise ValueError("--batch-size cannot exceed --buffer-size")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must be in (0, 1]")
    if args.checkpoint_freq < 0:
        raise ValueError("--checkpoint-freq must be non-negative")
    if args.control_period_s <= 0.0:
        raise ValueError("--control-period-s must be positive")
    if args.startup_timeout_s <= 0.0:
        raise ValueError("--startup-timeout-s must be positive")
    if args.shutdown_timeout_s <= 0.0:
        raise ValueError("--shutdown-timeout-s must be positive")


def default_run_dir(repo_root: Path, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo_root
        / "experiments"
        / "sailboat_RL"
        / "results"
        / "sac"
        / f"{timestamp}_{os.getpid()}"
    )


def prepare_run_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"run path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "models", "tensorboard", "trials"):
        (path / child).mkdir(exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _training_config(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    scenario_path: Path,
    params_file: Path,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "residual_control": True,
        "repo_root": str(repo_root),
        "scenario_path": str(scenario_path),
        "scenario_id": args.scenario_id,
        "params_file": str(params_file),
        "run_dir": str(run_dir),
        "arguments": vars(args),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "gymnasium": _package_version("gymnasium"),
            "stable_baselines3": _package_version("stable-baselines3"),
            "torch": _package_version("torch"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_training_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from experiments.sailboat_RL.runtime import (
            LifecycleEnvironmentSettings,
            build_lifecycle_environment,
        )
    except ImportError as exc:
        raise SystemExit(
            "Stable-Baselines3 is not installed. Rebuild dave:sailboat-rl "
            "from .docker/sailboat.amd64.dockerfile."
        ) from exc

    repo_root = get_repo_root()
    scenario_path = resolve_repo_path(repo_root, args.scenario)
    params_file = resolve_repo_path(repo_root, args.params_file)
    run_dir = (
        resolve_repo_path(repo_root, args.run_dir)
        if args.run_dir
        else default_run_dir(repo_root)
    )
    try:
        prepare_run_dir(run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    _write_json(
        run_dir / "training_config.json",
        _training_config(
            args,
            repo_root=repo_root,
            scenario_path=scenario_path,
            params_file=params_file,
            run_dir=run_dir,
        ),
    )
    print(f"run_dir: {run_dir}")

    env, _backend = build_lifecycle_environment(
        LifecycleEnvironmentSettings(
            repo_root=repo_root,
            scenario_path=scenario_path,
            scenario_id=args.scenario_id,
            params_file=params_file,
            results_dir=run_dir / "trials",
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
            repeat_start=int(args.repeat_start),
        )
    )
    wrapped_env = None
    model = None
    started_at = time.monotonic()
    try:
        if not args.skip_env_check:
            print("env_check: starting")
            check_env(env, warn=True, skip_render_check=True)
            print("env_check: PASSED")

        if args.check_env_only:
            _write_json(
                run_dir / "training_summary.json",
                {
                    "status": "environment_check_passed",
                    "timesteps": 0,
                    "elapsed_wall_s": time.monotonic() - started_at,
                },
            )
            return 0

        wrapped_env = Monitor(
            env,
            filename=str(run_dir / "monitor.csv"),
            info_keywords=("reason",),
        )
        callback = None
        if args.checkpoint_freq > 0:
            callback = CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="sac_sailboat",
                save_replay_buffer=True,
            )

        model = SAC(
            "MlpPolicy",
            wrapped_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=args.gamma,
            tau=args.tau,
            ent_coef="auto",
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=1,
            tb_log_name="sac_sailboat",
        )
        final_model = run_dir / "models" / "sac_final"
        replay_buffer = run_dir / "models" / "sac_replay_buffer.pkl"
        model.save(final_model)
        model.save_replay_buffer(replay_buffer)
        _write_json(
            run_dir / "training_summary.json",
            {
                "status": "complete",
                "timesteps": int(model.num_timesteps),
                "elapsed_wall_s": time.monotonic() - started_at,
                "model": str(final_model.with_suffix(".zip")),
                "replay_buffer": str(replay_buffer),
            },
        )
        print(f"final_model: {final_model.with_suffix('.zip')}")
        print("sac_training: PASSED")
        return 0
    except KeyboardInterrupt:
        summary: dict[str, Any] = {
            "status": "interrupted",
            "elapsed_wall_s": time.monotonic() - started_at,
        }
        if model is not None:
            interrupted_model = run_dir / "models" / "sac_interrupted"
            interrupted_buffer = (
                run_dir / "models" / "sac_interrupted_replay_buffer.pkl"
            )
            model.save(interrupted_model)
            model.save_replay_buffer(interrupted_buffer)
            summary.update(
                {
                    "timesteps": int(model.num_timesteps),
                    "model": str(interrupted_model.with_suffix(".zip")),
                    "replay_buffer": str(interrupted_buffer),
                }
            )
        _write_json(run_dir / "training_summary.json", summary)
        print("sac_training: INTERRUPTED")
        return 130
    except Exception as exc:
        _write_json(
            run_dir / "training_summary.json",
            {
                "status": "failed",
                "timesteps": int(model.num_timesteps) if model is not None else 0,
                "elapsed_wall_s": time.monotonic() - started_at,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        else:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
