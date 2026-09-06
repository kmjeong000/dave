from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


REWARD_COMPONENT_KEYS = (
    "progress",
    "cross_track",
    "roll",
    "residual",
    "residual_change",
    "waypoint",
    "mission",
    "terminal_accuracy",
    "excessive_roll",
)


def reward_component_info_key(name: str) -> str:
    if name not in REWARD_COMPONENT_KEYS:
        raise KeyError(f"unknown reward component: {name}")
    return f"reward_{name}"


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def wrap_pi(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class RawState:
    """Physical state used by the residual-RL environment."""

    sim_time_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float
    roll_deg: float
    target_x_m: float
    target_y_m: float
    wind_x_mps: float
    wind_y_mps: float
    base_rudder_rad: float
    base_sail_rad: float
    residual_rudder_rad: float
    residual_sail_rad: float
    cross_track_error_m: float
    waypoint_index: int = 0
    waypoint_count: int = 1
    mission_complete: bool = False
    termination_reason: str = ""
    termination_truncated: bool = False
    # The command adapter's actual bounded output. Attach backends that do
    # not observe the final actuator topics may leave these unset; managed
    # lifecycle episodes require them before writing per-step telemetry.
    final_rudder_rad: float | None = None
    final_sail_rad: float | None = None

    @property
    def distance_to_waypoint_m(self) -> float:
        return math.hypot(self.target_x_m - self.x_m, self.target_y_m - self.y_m)

    @property
    def target_bearing_rad(self) -> float:
        # Match the gazebo_xy_m convention used by sailboat_BO/run_trial.py.
        return math.atan2(self.target_x_m - self.x_m, self.target_y_m - self.y_m)

    @property
    def heading_error_rad(self) -> float:
        return wrap_pi(self.target_bearing_rad - self.yaw_rad)

    @property
    def wind_from_direction_rad(self) -> float:
        if self.wind_x_mps == 0.0 and self.wind_y_mps == 0.0:
            return 0.0
        wind_toward = math.atan2(self.wind_x_mps, self.wind_y_mps)
        return wrap_pi(wind_toward + math.pi)

    @property
    def relative_wind_angle_rad(self) -> float:
        return wrap_pi(self.wind_from_direction_rad - self.yaw_rad)


@dataclass(frozen=True)
class RewardConfig:
    # Match progress_scale_m so unsaturated progress reward remains equal to
    # progress in metres, as it was before normalization.
    progress_weight: float = 3.0
    cross_track_weight: float = 0.10
    roll_weight: float = 0.45
    residual_weight: float = 0.20
    residual_change_weight: float = 0.40
    waypoint_bonus: float = 10.0
    mission_bonus: float = 50.0
    terminal_accuracy_weight: float = 10.0
    excessive_roll_penalty: float = 50.0


@dataclass(frozen=True)
class EnvironmentConfig:
    rudder_residual_limit_rad: float = math.radians(5.0)
    sail_residual_limit_rad: float = math.radians(5.0)
    distance_scale_m: float = 100.0
    # A 1.5 m scale saturated about 21% of live 0.5 s control steps.  Three
    # metres retains headroom while preserving metre-for-metre reward below it.
    progress_scale_m: float = 3.0
    cross_track_scale_m: float = 5.0
    speed_scale_mps: float = 3.0
    roll_scale_deg: float = 45.0
    actuator_scale_rad: float = math.radians(45.0)
    success_radius_m: float = 5.0
    max_roll_deg: float = 45.0
    episode_timeout_s: float = 240.0
    control_period_s: float = 0.5
    backend_authoritative_termination: bool = False
    reward: RewardConfig = RewardConfig()


@dataclass(frozen=True)
class TransitionResult:
    reward: float
    terminated: bool
    truncated: bool
    reason: str
    components: dict[str, float]
    diagnostics: dict[str, float]


def scale_action(
    action: Sequence[float] | np.ndarray,
    config: EnvironmentConfig,
) -> np.ndarray:
    """Map normalized rudder/sail actions in [-1, 1] to radians."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("action values must be finite")
    clipped = np.clip(values, -1.0, 1.0)
    return clipped * np.asarray(
        [
            config.rudder_residual_limit_rad,
            config.sail_residual_limit_rad,
        ],
        dtype=np.float32,
    )


def build_observation(state: RawState, config: EnvironmentConfig) -> np.ndarray:
    """Build a bounded, dimensionless observation vector."""
    distance = state.distance_to_waypoint_m
    heading_error = state.heading_error_rad
    relative_wind = state.relative_wind_angle_rad
    waypoint_progress = (
        float(state.waypoint_index) / max(float(state.waypoint_count - 1), 1.0)
    )

    return np.asarray(
        [
            _clip(distance / config.distance_scale_m, 0.0, 2.0),
            _clip(
                state.cross_track_error_m
                / max(config.cross_track_scale_m, 1e-9),
                0.0,
                2.0,
            ),
            math.sin(heading_error),
            math.cos(heading_error),
            _clip(state.speed_mps / config.speed_scale_mps, 0.0, 2.0),
            _clip(state.roll_deg / config.roll_scale_deg, -2.0, 2.0),
            math.sin(relative_wind),
            math.cos(relative_wind),
            _clip(
                state.base_rudder_rad / config.actuator_scale_rad,
                -1.0,
                1.0,
            ),
            _clip(
                state.base_sail_rad / config.actuator_scale_rad,
                -1.0,
                1.0,
            ),
            _clip(
                state.residual_rudder_rad / config.rudder_residual_limit_rad,
                -1.0,
                1.0,
            ),
            _clip(
                state.residual_sail_rad / config.sail_residual_limit_rad,
                -1.0,
                1.0,
            ),
            _clip(waypoint_progress, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def evaluate_transition(
    previous: RawState,
    current: RawState,
    previous_action_rad: Sequence[float] | np.ndarray,
    action_rad: Sequence[float] | np.ndarray,
    episode_elapsed_s: float,
    config: EnvironmentConfig,
) -> TransitionResult:
    """Compute reward and Gymnasium termination signals for one control step."""
    previous_action = np.asarray(previous_action_rad, dtype=np.float32)
    action = np.asarray(action_rad, dtype=np.float32)
    if previous_action.shape != (2,) or action.shape != (2,):
        raise ValueError("physical actions must both have shape (2,)")

    waypoint_advanced = current.waypoint_index > previous.waypoint_index
    if waypoint_advanced:
        # The target coordinates change as soon as a waypoint is captured.
        # Compare the new position with the previous target for this one
        # transition; comparing distances to two different targets creates a
        # large artificial negative reward at every waypoint handoff.
        current_distance_for_progress = math.hypot(
            previous.target_x_m - current.x_m,
            previous.target_y_m - current.y_m,
        )
    else:
        current_distance_for_progress = current.distance_to_waypoint_m
    progress_m = previous.distance_to_waypoint_m - current_distance_for_progress
    progress_normalized = _clip(
        progress_m / max(config.progress_scale_m, 1e-9),
        -1.0,
        1.0,
    )
    cross_track_normalized = _clip(
        current.cross_track_error_m / max(config.cross_track_scale_m, 1e-9),
        0.0,
        1.0,
    )
    roll_normalized = _clip(
        abs(current.roll_deg) / max(config.roll_scale_deg, 1e-9),
        0.0,
        1.0,
    )
    mission_complete = bool(current.mission_complete)
    excessive_roll = abs(current.roll_deg) > config.max_roll_deg
    local_excessive_roll = (
        excessive_roll and not config.backend_authoritative_termination
    )
    timeout = (
        float(episode_elapsed_s) >= config.episode_timeout_s
        and not config.backend_authoritative_termination
    )

    normalized_residual = np.clip(
        np.asarray(
            [
                action[0] / config.rudder_residual_limit_rad,
                action[1] / config.sail_residual_limit_rad,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )
    # Moving from -limit to +limit is the largest possible one-step change.
    # Dividing by 2*limit keeps each change input in [-1, 1].
    normalized_change = np.clip(
        np.asarray(
            [
                (action[0] - previous_action[0])
                / (2.0 * config.rudder_residual_limit_rad),
                (action[1] - previous_action[1])
                / (2.0 * config.sail_residual_limit_rad),
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )
    terminal_accuracy_normalized = (
        1.0
        - _clip(
            current.distance_to_waypoint_m
            / max(config.success_radius_m, 1e-9),
            0.0,
            1.0,
        )
        if mission_complete
        else 0.0
    )

    components = {
        "progress": config.reward.progress_weight * progress_normalized,
        "cross_track": -config.reward.cross_track_weight
        * cross_track_normalized,
        "roll": -config.reward.roll_weight * roll_normalized,
        "residual": -config.reward.residual_weight
        * float(np.square(normalized_residual).mean()),
        "residual_change": -config.reward.residual_change_weight
        * float(np.square(normalized_change).mean()),
        "waypoint": config.reward.waypoint_bonus if waypoint_advanced else 0.0,
        "mission": config.reward.mission_bonus if mission_complete else 0.0,
        "terminal_accuracy": config.reward.terminal_accuracy_weight
        * terminal_accuracy_normalized,
        "excessive_roll": (
            -config.reward.excessive_roll_penalty if excessive_roll else 0.0
        ),
    }
    reward = float(sum(components.values()))
    diagnostics = {
        "progress_m": float(progress_m),
        "progress_normalized": float(progress_normalized),
        "progress_saturated": float(abs(progress_normalized) >= 1.0 - 1e-9),
        "cross_track_normalized": float(cross_track_normalized),
        "roll_normalized": float(roll_normalized),
        "terminal_accuracy_normalized": float(terminal_accuracy_normalized),
    }

    if mission_complete:
        return TransitionResult(
            reward,
            True,
            False,
            "mission_complete",
            components,
            diagnostics,
        )
    if local_excessive_roll:
        return TransitionResult(
            reward,
            True,
            False,
            "excessive_roll",
            components,
            diagnostics,
        )
    if current.termination_reason:
        return TransitionResult(
            reward,
            not current.termination_truncated,
            current.termination_truncated,
            current.termination_reason,
            components,
            diagnostics,
        )
    if timeout:
        return TransitionResult(
            reward,
            False,
            True,
            "timeout",
            components,
            diagnostics,
        )
    return TransitionResult(
        reward,
        False,
        False,
        "",
        components,
        diagnostics,
    )
