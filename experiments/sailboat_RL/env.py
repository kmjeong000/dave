from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised by runtime setup
    raise RuntimeError(
        "Gymnasium is required for SailboatResidualEnv. "
        "Install experiments/sailboat_RL/requirements.txt."
    ) from exc

from .core import (
    EnvironmentConfig,
    RawState,
    REWARD_COMPONENT_KEYS,
    build_observation,
    evaluate_transition,
    reward_component_info_key,
    scale_action,
)
from .telemetry import StepTelemetryWriter


class ResidualBackend(Protocol):
    """Runtime seam implemented by ROS attach and future lifecycle backends."""

    def reset(self, *, seed: int | None, options: dict[str, Any]) -> RawState:
        ...

    def step(
        self,
        rudder_residual_rad: float,
        sail_residual_rad: float,
        control_period_s: float,
    ) -> RawState:
        ...

    def close(self) -> None:
        ...


class SailboatResidualEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment that learns bounded corrections to BO commands."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: ResidualBackend,
        config: EnvironmentConfig | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config or EnvironmentConfig()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray(
                [0, 0, -1, -1, 0, -2, -1, -1, -1, -1, -1, -1, 0],
                dtype=np.float32,
            ),
            high=np.asarray(
                [2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self._state: RawState | None = None
        self._previous_action_rad = np.zeros(2, dtype=np.float32)
        self._episode_start_sim_s = 0.0
        self._episode_reward_components = self._empty_reward_components()
        self._progress_sample_count = 0
        self._progress_saturation_count = 0
        self._progress_normalized_abs_sum = 0.0
        self._progress_normalized_abs_max = 0.0
        self._episode_index = 0
        self._telemetry: StepTelemetryWriter | None = None

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        return {name: 0.0 for name in REWARD_COMPONENT_KEYS}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._finalize_telemetry("environment_reset")
        super().reset(seed=seed)
        state = self.backend.reset(seed=seed, options=dict(options or {}))
        self._state = state
        self._previous_action_rad = np.zeros(2, dtype=np.float32)
        self._episode_start_sim_s = float(state.sim_time_s)
        self._episode_reward_components = self._empty_reward_components()
        self._progress_sample_count = 0
        self._progress_saturation_count = 0
        self._progress_normalized_abs_sum = 0.0
        self._progress_normalized_abs_max = 0.0
        self._start_telemetry()
        self._episode_index += 1
        observation = build_observation(state, self.config)
        info = self._build_info(state, reason="reset")
        info["telemetry_path"] = (
            str(self._telemetry.path) if self._telemetry is not None else ""
        )
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")

        normalized_action = np.asarray(action, dtype=np.float32)
        physical_action = scale_action(normalized_action, self.config)
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        next_state = self.backend.step(
            float(physical_action[0]),
            float(physical_action[1]),
            self.config.control_period_s,
        )
        elapsed_s = max(
            0.0,
            float(next_state.sim_time_s) - self._episode_start_sim_s,
        )
        result = evaluate_transition(
            self._state,
            next_state,
            self._previous_action_rad,
            physical_action,
            elapsed_s,
            self.config,
        )
        self._state = next_state
        self._previous_action_rad = physical_action
        info = self._build_info(next_state, reason=result.reason)
        info["reward_components"] = result.components
        info["reward_diagnostics"] = result.diagnostics
        for name in REWARD_COMPONENT_KEYS:
            self._episode_reward_components[name] += float(
                result.components[name]
            )
        progress_abs = abs(float(result.diagnostics["progress_normalized"]))
        self._progress_sample_count += 1
        self._progress_saturation_count += int(
            result.diagnostics["progress_saturated"] > 0.5
        )
        self._progress_normalized_abs_sum += progress_abs
        self._progress_normalized_abs_max = max(
            self._progress_normalized_abs_max,
            progress_abs,
        )
        if result.terminated or result.truncated:
            info["episode_reward_components"] = dict(
                self._episode_reward_components
            )
            for name, value in self._episode_reward_components.items():
                info[reward_component_info_key(name)] = float(value)
            sample_count = max(self._progress_sample_count, 1)
            info["progress_saturation_ratio"] = (
                self._progress_saturation_count / sample_count
            )
            info["progress_normalized_abs_mean"] = (
                self._progress_normalized_abs_sum / sample_count
            )
            info["progress_normalized_abs_max"] = (
                self._progress_normalized_abs_max
            )
        info["physical_action_rad"] = physical_action.copy()
        info["normalized_action"] = normalized_action.copy()
        self._record_telemetry(
            normalized_action=normalized_action,
            physical_action=physical_action,
            state=next_state,
            reward=result.reward,
            components=result.components,
            terminated=result.terminated,
            truncated=result.truncated,
            reason=result.reason,
        )
        return (
            build_observation(next_state, self.config),
            result.reward,
            result.terminated,
            result.truncated,
            info,
        )

    def _build_info(self, state: RawState, *, reason: str) -> dict[str, Any]:
        return {
            "reason": reason,
            "sim_time_s": state.sim_time_s,
            "distance_to_waypoint_m": state.distance_to_waypoint_m,
            "cross_track_error_m": state.cross_track_error_m,
            "waypoint_index": state.waypoint_index,
            "waypoint_count": state.waypoint_count,
            "base_rudder_rad": state.base_rudder_rad,
            "base_sail_rad": state.base_sail_rad,
            "residual_rudder_rad": state.residual_rudder_rad,
            "residual_sail_rad": state.residual_sail_rad,
            "final_rudder_rad": state.final_rudder_rad,
            "final_sail_rad": state.final_sail_rad,
            "actual_residual_rudder_rad": (
                state.final_rudder_rad - state.base_rudder_rad
                if state.final_rudder_rad is not None
                else None
            ),
            "actual_residual_sail_rad": (
                state.final_sail_rad - state.base_sail_rad
                if state.final_sail_rad is not None
                else None
            ),
            "backend_termination_reason": state.termination_reason,
            "backend_termination_truncated": state.termination_truncated,
        }

    def close(self) -> None:
        self._finalize_telemetry("environment_close")
        self.backend.close()
        self._state = None

    def _start_telemetry(self) -> None:
        trial_dir = getattr(self.backend, "current_trial_dir", None)
        trial_id = str(getattr(self.backend, "current_trial_id", ""))
        if trial_dir is None or not trial_id:
            self._telemetry = None
            return
        self._telemetry = StepTelemetryWriter(
            Path(trial_dir) / "rl" / "rl_steps.csv"
        )

    def _finalize_telemetry(self, reason: str) -> None:
        if self._telemetry is None:
            return
        self._telemetry.finalize(reason)
        self._telemetry = None

    def _record_telemetry(
        self,
        *,
        normalized_action: np.ndarray,
        physical_action: np.ndarray,
        state: RawState,
        reward: float,
        components: dict[str, float],
        terminated: bool,
        truncated: bool,
        reason: str,
    ) -> None:
        if self._telemetry is None:
            return
        if state.final_rudder_rad is None or state.final_sail_rad is None:
            raise RuntimeError(
                "managed RL telemetry requires final adapter actuator commands"
            )
        row: dict[str, Any] = {
            "record_type": "step",
            "trial_id": str(getattr(self.backend, "current_trial_id", "")),
            "episode_index": self._episode_index - 1,
            "step": self._progress_sample_count - 1,
            "sim_time_s": state.sim_time_s,
            "action_rudder_normalized": normalized_action[0],
            "action_sail_normalized": normalized_action[1],
            "requested_residual_rudder_rad": physical_action[0],
            "requested_residual_sail_rad": physical_action[1],
            "residual_rudder_rad": (
                state.final_rudder_rad - state.base_rudder_rad
            ),
            "residual_sail_rad": state.final_sail_rad - state.base_sail_rad,
            "base_rudder_rad": state.base_rudder_rad,
            "base_sail_rad": state.base_sail_rad,
            "final_rudder_rad": state.final_rudder_rad,
            "final_sail_rad": state.final_sail_rad,
            "speed_mps": state.speed_mps,
            "roll_deg": state.roll_deg,
            "distance_to_waypoint_m": state.distance_to_waypoint_m,
            "cross_track_error_m": state.cross_track_error_m,
            "waypoint_index": state.waypoint_index,
            "waypoint_count": state.waypoint_count,
            "reward": reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reason": reason,
        }
        for name in REWARD_COMPONENT_KEYS:
            row[reward_component_info_key(name)] = components[name]
        self._telemetry.record(row)
