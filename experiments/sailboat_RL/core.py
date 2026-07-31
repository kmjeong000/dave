from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


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
    waypoint_index: int = 0
    waypoint_count: int = 1
    mission_complete: bool = False

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
    progress_weight: float = 1.0
    cross_track_weight: float = 0.02
    roll_weight: float = 0.01
    residual_weight: float = 0.10
    residual_change_weight: float = 0.05
    waypoint_bonus: float = 10.0
    mission_bonus: float = 50.0
    excessive_roll_penalty: float = 50.0


@dataclass(frozen=True)
class EnvironmentConfig:
    rudder_residual_limit_rad: float = math.radians(5.0)
    sail_residual_limit_rad: float = math.radians(5.0)
    distance_scale_m: float = 100.0
    cross_track_scale_m: float = 20.0
    speed_scale_mps: float = 3.0
    roll_scale_deg: float = 45.0
    actuator_scale_rad: float = math.radians(45.0)
    success_radius_m: float = 5.0
    max_roll_deg: float = 45.0
    episode_timeout_s: float = 240.0
    control_period_s: float = 0.5
    reward: RewardConfig = RewardConfig()


@dataclass(frozen=True)
class TransitionResult:
    reward: float
    terminated: bool
    truncated: bool
    reason: str
    components: dict[str, float]


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

    progress_m = previous.distance_to_waypoint_m - current.distance_to_waypoint_m
    waypoint_advanced = current.waypoint_index > previous.waypoint_index
    mission_complete = bool(current.mission_complete)
    excessive_roll = abs(current.roll_deg) > config.max_roll_deg
    timeout = float(episode_elapsed_s) >= config.episode_timeout_s

    normalized_residual = np.asarray(
        [
            action[0] / config.rudder_residual_limit_rad,
            action[1] / config.sail_residual_limit_rad,
        ],
        dtype=np.float32,
    )
    normalized_change = np.asarray(
        [
            (action[0] - previous_action[0]) / config.rudder_residual_limit_rad,
            (action[1] - previous_action[1]) / config.sail_residual_limit_rad,
        ],
        dtype=np.float32,
    )

    components = {
        "progress": config.reward.progress_weight * progress_m,
        # Cross-track requires a path-segment estimate and is reserved for the
        # lifecycle backend. Keep the component explicit for stable logging.
        "cross_track": 0.0,
        "roll": -config.reward.roll_weight * abs(current.roll_deg),
        "residual": -config.reward.residual_weight
        * float(np.square(normalized_residual).sum()),
        "residual_change": -config.reward.residual_change_weight
        * float(np.square(normalized_change).sum()),
        "waypoint": config.reward.waypoint_bonus if waypoint_advanced else 0.0,
        "mission": config.reward.mission_bonus if mission_complete else 0.0,
        "excessive_roll": (
            -config.reward.excessive_roll_penalty if excessive_roll else 0.0
        ),
    }
    reward = float(sum(components.values()))

    if mission_complete:
        return TransitionResult(reward, True, False, "mission_complete", components)
    if excessive_roll:
        return TransitionResult(reward, True, False, "excessive_roll", components)
    if timeout:
        return TransitionResult(reward, False, True, "timeout", components)
    return TransitionResult(reward, False, False, "", components)
