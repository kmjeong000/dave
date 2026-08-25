from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.sailboat_BO.common import load_yaml
from experiments.sailboat_BO.run_trial import get_scenario

from .core import EnvironmentConfig
from .env import SailboatResidualEnv
from .lifecycle_backend import EpisodeLifecycleBackend, LifecycleConfig
from .ros_backend import Ros2AttachBackend


@dataclass(frozen=True)
class LifecycleEnvironmentSettings:
    """Inputs shared by lifecycle smoke checks and RL training."""

    repo_root: Path
    scenario_path: Path
    scenario_id: str
    params_file: Path
    results_dir: Path
    control_period_s: float = 0.5
    execution_backend: str = "local"
    docker_container: str | None = None
    container_repo_root: str | None = None
    startup_timeout_s: float = 180.0
    shutdown_timeout_s: float = 45.0
    repeat_start: int = 0

    def __post_init__(self) -> None:
        if self.control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive")


def build_lifecycle_environment(
    settings: LifecycleEnvironmentSettings,
) -> tuple[SailboatResidualEnv, EpisodeLifecycleBackend]:
    """Build the canonical residual environment around managed BO trials."""

    config = load_yaml(settings.scenario_path)
    scenario = get_scenario(config, settings.scenario_id)
    termination = dict(config.get("termination", {}))
    termination.update(scenario.get("termination", {}))
    namespace = str(config["study"].get("namespace", "sailboat"))

    def make_attach_backend() -> Ros2AttachBackend:
        return Ros2AttachBackend(
            waypoints=scenario["mission"]["waypoints"],
            namespace=namespace,
            wind_world_xyz_mps=scenario["world"]["wind_world_xyz_mps"],
            waypoint_capture_radius_m=float(
                termination.get("waypoint_capture_radius_m", 5.0)
            ),
            waypoint_capture_hold_s=float(
                termination.get("waypoint_capture_hold_s", 1.0)
            ),
        )

    backend = EpisodeLifecycleBackend(
        LifecycleConfig(
            repo_root=settings.repo_root,
            scenario_path=settings.scenario_path,
            scenario_id=settings.scenario_id,
            params_file=settings.params_file,
            results_dir=settings.results_dir,
            execution_backend=settings.execution_backend,
            docker_container=settings.docker_container,
            container_repo_root=settings.container_repo_root,
            startup_timeout_s=settings.startup_timeout_s,
            shutdown_timeout_s=settings.shutdown_timeout_s,
            repeat_start=settings.repeat_start,
        ),
        make_attach_backend,
    )
    env = SailboatResidualEnv(
        backend,
        EnvironmentConfig(
            success_radius_m=float(termination.get("success_radius_m", 5.0)),
            max_roll_deg=float(termination.get("max_roll_deg", 45.0)),
            episode_timeout_s=float(termination.get("timeout_s", 240.0)),
            control_period_s=settings.control_period_s,
            # run_trial.py owns the sustained roll, timeout, progress, and
            # mission-completion rules in lifecycle mode.  The attach backend
            # still supplies observations and reward inputs, but must not end
            # the Gym episode ahead of the runner's finalized outcome.
            backend_authoritative_termination=True,
        ),
    )
    return env, backend
