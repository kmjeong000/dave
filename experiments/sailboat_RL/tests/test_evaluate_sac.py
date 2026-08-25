from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from experiments.sailboat_RL.evaluate_sac import (
    _controller_order,
    _finalize_episode_record,
    default_results_dir,
    parse_args,
    run_episode,
    validate_evaluation_args,
)


class FakeBackend:
    current_trial_id = "fake_trial"

    def __init__(self, summary_json):
        self.current_summary_json = str(summary_json)


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, *, seed, options):
        assert seed == 7
        assert options == {"repeat_idx": 11}
        return np.zeros(12, dtype=np.float32), {"sim_time_s": 10.0}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count == 2
        return (
            np.zeros(12, dtype=np.float32),
            1.5,
            terminated,
            False,
            {
                "reason": "mission_complete" if terminated else "",
                "sim_time_s": 10.0 + self.step_count,
                "distance_to_waypoint_m": 2.0,
            },
        )


def test_run_episode_records_rewards_actions_and_bo_metrics(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata": {"status": "success", "failure_reason": ""},
                "constraints": {"constraint_mission_complete": True},
                "metrics": {
                    "mission_time_s": 120.0,
                    "progress_ratio": 0.99,
                    "final_distance_to_wp_m": 2.0,
                    "max_abs_roll_deg": 5.0,
                    "xte_rms_m": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )
    env = FakeEnv()
    backend = FakeBackend(summary_path)

    record = run_episode(
        env,
        backend,
        lambda observation: np.asarray([1.0, -1.0], dtype=np.float32),
        controller="policy",
        pair_index=0,
        seed=7,
        repeat_idx=11,
    )

    assert record["evaluation_status"] == "complete"
    assert record["episode_reward"] == 3.0
    assert record["episode_steps"] == 2
    assert record["reason"] == "mission_complete"
    assert record["rudder_abs_saturation_ratio"] == 1.0
    assert record["sail_abs_saturation_ratio"] == 1.0
    assert record["bo_status"] == "success"
    assert record["bo_mission_time_s"] == 120.0


def test_policy_and_compare_modes_require_model():
    with pytest.raises(ValueError, match="model is required"):
        validate_evaluation_args(parse_args(["--mode", "compare"]))
    validate_evaluation_args(parse_args(["--mode", "zero"]))


def test_default_results_dir_is_timestamped(tmp_path):
    now = datetime(2026, 8, 25, 8, 9, 10, tzinfo=timezone.utc)

    path = default_results_dir(tmp_path, now=now)

    assert path.parent == tmp_path / "experiments/sailboat_RL/results/evaluation"
    assert path.name.startswith("20260825T080910Z_")


def test_compare_mode_alternates_controller_order():
    assert _controller_order("compare", 0) == ["zero", "policy"]
    assert _controller_order("compare", 1) == ["policy", "zero"]


def test_finalize_record_rejects_gym_bo_terminal_mismatch(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "status": "failure",
                    "failure_reason": "environment_close",
                },
                "constraints": {"constraint_mission_complete": False},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    record = {
        "evaluation_status": "complete",
        "reason": "mission_complete",
        "summary_json": str(summary_path),
    }

    _finalize_episode_record(record)

    assert record["evaluation_status"] == "error"
    assert record["error_type"] == "EpisodeFinalizationError"
    assert "environment_close" in record["error"]


def test_finalize_record_accepts_matching_runner_failure(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "status": "failure",
                    "failure_reason": "no_progress",
                },
                "constraints": {"constraint_mission_complete": False},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    record = {
        "evaluation_status": "complete",
        "reason": "no_progress",
        "summary_json": str(summary_path),
    }

    _finalize_episode_record(record)

    assert record["evaluation_status"] == "complete"
    assert record["bo_failure_reason"] == "no_progress"
