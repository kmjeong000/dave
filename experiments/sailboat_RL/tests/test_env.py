import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from experiments.sailboat_RL.core import EnvironmentConfig, RawState
from experiments.sailboat_RL.env import SailboatResidualEnv


class FakeBackend:
    def __init__(self):
        self.actions = []
        self.closed = False
        self.state = self._state(sim_time_s=0.0, y_m=0.0)

    @staticmethod
    def _state(sim_time_s, y_m, *, mission_complete=False):
        return RawState(
            sim_time_s=sim_time_s,
            x_m=0.0,
            y_m=y_m,
            yaw_rad=0.0,
            speed_mps=1.0,
            roll_deg=0.0,
            target_x_m=0.0,
            target_y_m=100.0,
            wind_x_mps=0.0,
            wind_y_mps=8.0,
            base_rudder_rad=0.1,
            base_sail_rad=0.2,
            residual_rudder_rad=0.0,
            residual_sail_rad=0.0,
            cross_track_error_m=0.0,
            mission_complete=mission_complete,
        )

    def reset(self, *, seed, options):
        self.state = self._state(sim_time_s=0.0, y_m=0.0)
        return self.state

    def step(self, rudder_residual_rad, sail_residual_rad, control_period_s):
        self.actions.append(
            (rudder_residual_rad, sail_residual_rad, control_period_s)
        )
        self.state = self._state(sim_time_s=1.0, y_m=1.0)
        return self.state

    def close(self):
        self.closed = True


def test_environment_conforms_to_gymnasium_api():
    backend = FakeBackend()
    config = EnvironmentConfig(control_period_s=0.25)
    env = SailboatResidualEnv(backend, config)

    observation, info = env.reset(seed=7)
    result = env.step(np.asarray([0.5, -0.5], dtype=np.float32))

    assert env.observation_space.contains(observation)
    assert len(result) == 5
    assert backend.actions[0][2] == pytest.approx(0.25)
    assert backend.actions[0][0] == pytest.approx(
        0.5 * config.rudder_residual_limit_rad
    )
    assert backend.actions[0][1] == pytest.approx(
        -0.5 * config.sail_residual_limit_rad
    )
    assert info["reason"] == "reset"

    env.close()
    assert backend.closed


def test_terminal_step_exposes_episode_reward_components_and_progress_stats():
    backend = FakeBackend()
    backend.step = lambda rudder, sail, period: backend._state(
        sim_time_s=1.0,
        y_m=1.0,
        mission_complete=True,
    )
    env = SailboatResidualEnv(backend, EnvironmentConfig())
    env.reset(seed=7)

    _observation, reward, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )

    assert terminated
    assert not truncated
    assert info["reward_progress"] != 0.0
    assert sum(info["episode_reward_components"].values()) == pytest.approx(
        reward
    )
    assert 0.0 <= info["progress_saturation_ratio"] <= 1.0
