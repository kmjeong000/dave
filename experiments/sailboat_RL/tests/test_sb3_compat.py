from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC

from experiments.sailboat_RL.core import EnvironmentConfig, RawState
from experiments.sailboat_RL.env import SailboatResidualEnv


class FakeBackend:
    def __init__(self):
        self.sim_time_s = 0.0
        self.y_m = 0.0
        self.closed = False

    def _state(self, rudder=0.0, sail=0.0):
        return RawState(
            sim_time_s=self.sim_time_s,
            x_m=0.0,
            y_m=self.y_m,
            yaw_rad=0.0,
            speed_mps=1.0,
            roll_deg=0.0,
            target_x_m=0.0,
            target_y_m=100.0,
            wind_x_mps=0.0,
            wind_y_mps=8.0,
            base_rudder_rad=0.1,
            base_sail_rad=0.2,
            residual_rudder_rad=rudder,
            residual_sail_rad=sail,
        )

    def reset(self, *, seed, options):
        del seed, options
        self.sim_time_s = 0.0
        self.y_m = 0.0
        return self._state()

    def step(self, rudder_residual_rad, sail_residual_rad, control_period_s):
        self.sim_time_s += control_period_s
        self.y_m += 0.5
        return self._state(rudder_residual_rad, sail_residual_rad)

    def close(self):
        self.closed = True


def test_environment_passes_stable_baselines3_checker():
    backend = FakeBackend()
    env = SailboatResidualEnv(
        backend,
        EnvironmentConfig(control_period_s=0.1),
    )

    check_env(env, warn=True, skip_render_check=True)

    action = env.action_space.sample().astype(np.float32)
    env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reason" in info
    env.close()
    assert backend.closed


def test_sac_collects_transitions_and_updates_on_environment():
    backend = FakeBackend()
    env = SailboatResidualEnv(
        backend,
        EnvironmentConfig(control_period_s=0.01),
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=2,
        buffer_size=50,
        batch_size=2,
        train_freq=(1, "step"),
        gradient_steps=1,
        device="cpu",
        verbose=0,
        seed=42,
    )

    model.learn(total_timesteps=8)

    assert model.num_timesteps == 8
    assert model.replay_buffer.size() == 8
    env.close()
    assert backend.closed
