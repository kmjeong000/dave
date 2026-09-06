from __future__ import annotations

import csv
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from experiments.sailboat_RL.core import EnvironmentConfig, RawState
from experiments.sailboat_RL.env import SailboatResidualEnv
from experiments.sailboat_RL.telemetry import STEP_TELEMETRY_COLUMNS


class TelemetryBackend:
    current_trial_id = "trial_telemetry"

    def __init__(self, trial_dir, *, terminal_on_step: bool):
        self.current_trial_dir = trial_dir
        self.terminal_on_step = terminal_on_step
        self.closed = False
        self.step_count = 0

    @staticmethod
    def _state(
        *,
        sim_time_s: float,
        y_m: float,
        rudder_residual_rad: float = 0.0,
        sail_residual_rad: float = 0.0,
        actual_rudder_residual_rad: float | None = None,
        actual_sail_residual_rad: float | None = None,
        mission_complete: bool = False,
    ) -> RawState:
        base_rudder = 0.1
        base_sail = 0.7
        actual_rudder = (
            rudder_residual_rad
            if actual_rudder_residual_rad is None
            else actual_rudder_residual_rad
        )
        actual_sail = (
            sail_residual_rad
            if actual_sail_residual_rad is None
            else actual_sail_residual_rad
        )
        return RawState(
            sim_time_s=sim_time_s,
            x_m=0.0,
            y_m=y_m,
            yaw_rad=0.0,
            speed_mps=1.25,
            roll_deg=2.0,
            target_x_m=0.0,
            target_y_m=100.0,
            wind_x_mps=0.0,
            wind_y_mps=8.0,
            base_rudder_rad=base_rudder,
            base_sail_rad=base_sail,
            residual_rudder_rad=rudder_residual_rad,
            residual_sail_rad=sail_residual_rad,
            final_rudder_rad=base_rudder + actual_rudder,
            final_sail_rad=base_sail + actual_sail,
            cross_track_error_m=0.25,
            mission_complete=mission_complete,
        )

    def reset(self, *, seed, options):
        del seed, options
        return self._state(sim_time_s=10.0, y_m=0.0)

    def step(self, rudder_residual_rad, sail_residual_rad, control_period_s):
        del control_period_s
        self.step_count += 1
        requested_rudder = float(rudder_residual_rad)
        requested_sail = float(sail_residual_rad)
        return self._state(
            sim_time_s=10.5,
            y_m=1.0,
            rudder_residual_rad=requested_rudder,
            sail_residual_rad=requested_sail,
            # Preserve a rate-limit-like requested/actual difference for the
            # nonzero-action telemetry test, but model zero command as an
            # already settled zero output for the cleanup test.
            actual_rudder_residual_rad=(
                math.copysign(0.01, requested_rudder)
                if requested_rudder != 0.0
                else 0.0
            ),
            actual_sail_residual_rad=(
                math.copysign(0.01, requested_sail)
                if requested_sail != 0.0
                else 0.0
            ),
            mission_complete=self.terminal_on_step,
        )

    def close(self):
        self.closed = True


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_terminal_step_telemetry_records_requested_and_actual_commands(tmp_path):
    backend = TelemetryBackend(tmp_path / "trial_telemetry", terminal_on_step=True)
    config = EnvironmentConfig(control_period_s=0.5)
    env = SailboatResidualEnv(backend, config)

    _observation, reset_info = env.reset(seed=7)
    telemetry_path = tmp_path / "trial_telemetry" / "rl" / "rl_steps.csv"
    assert reset_info["telemetry_path"] == str(telemetry_path)

    _observation, reward, terminated, truncated, info = env.step(
        np.asarray([0.2, -0.2], dtype=np.float32)
    )
    env.close()

    rows = _read_rows(telemetry_path)
    assert tuple(rows[0]) == STEP_TELEMETRY_COLUMNS
    assert len(rows) == 1
    row = rows[0]
    assert terminated and not truncated
    assert row["reason"] == "mission_complete"
    assert row["terminated"] == "True"
    assert float(row["action_rudder_normalized"]) == pytest.approx(0.2)
    assert float(row["action_sail_normalized"]) == pytest.approx(-0.2)
    assert float(row["requested_residual_rudder_rad"]) == pytest.approx(
        0.2 * config.rudder_residual_limit_rad
    )
    assert float(row["requested_residual_sail_rad"]) == pytest.approx(
        -0.2 * config.sail_residual_limit_rad
    )
    assert float(row["residual_rudder_rad"]) == pytest.approx(0.01)
    assert float(row["residual_sail_rad"]) == pytest.approx(-0.01)
    assert float(row["reward"]) == pytest.approx(reward)
    assert float(row["reward_progress"]) == pytest.approx(
        info["reward_components"]["progress"]
    )

    rudder = {name: float(row[name]) for name in (
        "base_rudder_rad", "residual_rudder_rad", "final_rudder_rad"
    )}
    sail = {name: float(row[name]) for name in (
        "base_sail_rad", "residual_sail_rad", "final_sail_rad"
    )}
    assert rudder["final_rudder_rad"] == pytest.approx(
        min(max(rudder["base_rudder_rad"] + rudder["residual_rudder_rad"], -0.7854), 0.7854)
    )
    assert sail["final_sail_rad"] == pytest.approx(
        min(max(sail["base_sail_rad"] + sail["residual_sail_rad"], 0.0), 0.7854)
    )
    assert all(
        math.isfinite(float(value))
        for name, value in row.items()
        if name not in {"record_type", "trial_id", "episode_index", "step", "terminated", "truncated", "reason", "waypoint_index", "waypoint_count"}
    )


def test_environment_close_preserves_final_nonterminal_step(tmp_path):
    backend = TelemetryBackend(tmp_path / "trial_telemetry", terminal_on_step=False)
    env = SailboatResidualEnv(backend, EnvironmentConfig())

    env.reset(seed=7)
    env.step(np.zeros(2, dtype=np.float32))
    env.close()

    rows = _read_rows(tmp_path / "trial_telemetry" / "rl" / "rl_steps.csv")
    assert len(rows) == 1
    assert rows[0]["truncated"] == "True"
    assert rows[0]["terminated"] == "False"
    assert rows[0]["reason"] == "environment_close"
    assert float(rows[0]["requested_residual_rudder_rad"]) == pytest.approx(0.0)
    assert float(rows[0]["requested_residual_sail_rad"]) == pytest.approx(0.0)
    assert float(rows[0]["residual_rudder_rad"]) == pytest.approx(0.0)
    assert float(rows[0]["residual_sail_rad"]) == pytest.approx(0.0)
    assert float(rows[0]["final_rudder_rad"]) == pytest.approx(
        float(rows[0]["base_rudder_rad"])
    )
    assert float(rows[0]["final_sail_rad"]) == pytest.approx(
        float(rows[0]["base_sail_rad"])
    )
    assert backend.closed
