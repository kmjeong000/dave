import math

import numpy as np
import pytest

from experiments.sailboat_RL.core import (
    EnvironmentConfig,
    RawState,
    build_observation,
    evaluate_transition,
    scale_action,
)


def make_state(**overrides):
    values = {
        "sim_time_s": 10.0,
        "x_m": 0.0,
        "y_m": 0.0,
        "yaw_rad": 0.0,
        "speed_mps": 1.0,
        "roll_deg": 5.0,
        "target_x_m": 0.0,
        "target_y_m": 100.0,
        "wind_x_mps": 0.0,
        "wind_y_mps": 8.0,
        "base_rudder_rad": 0.1,
        "base_sail_rad": 0.2,
        "residual_rudder_rad": 0.0,
        "residual_sail_rad": 0.0,
        "cross_track_error_m": 0.0,
        "waypoint_index": 0,
        "waypoint_count": 3,
        "mission_complete": False,
    }
    values.update(overrides)
    return RawState(**values)


def test_scale_action_maps_normalized_action_to_radians():
    config = EnvironmentConfig(
        rudder_residual_limit_rad=0.05,
        sail_residual_limit_rad=0.10,
    )

    result = scale_action(np.asarray([0.5, -1.0]), config)

    assert result == pytest.approx([0.025, -0.10])


def test_scale_action_clips_out_of_range_values():
    result = scale_action(np.asarray([2.0, -2.0]), EnvironmentConfig())

    assert result == pytest.approx(
        [math.radians(5.0), -math.radians(5.0)]
    )


def test_observation_has_expected_shape_and_bounds():
    observation = build_observation(make_state(), EnvironmentConfig())

    assert observation.shape == (13,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -2.0)
    assert np.all(observation <= 2.0)


def test_forward_progress_produces_positive_reward():
    config = EnvironmentConfig()
    previous = make_state(y_m=0.0)
    current = make_state(sim_time_s=11.0, y_m=2.0)

    result = evaluate_transition(
        previous,
        current,
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert result.reward > 0.0
    assert not result.terminated
    assert not result.truncated


def test_cross_track_reward_changes_between_zero_and_five_metres():
    config = EnvironmentConfig()
    previous = make_state(roll_deg=0.0, cross_track_error_m=0.0)
    on_path = make_state(
        sim_time_s=11.0,
        roll_deg=0.0,
        cross_track_error_m=0.0,
    )
    off_path = make_state(
        sim_time_s=11.0,
        roll_deg=0.0,
        cross_track_error_m=5.0,
    )

    zero = evaluate_transition(
        previous, on_path, [0.0, 0.0], [0.0, 0.0], 1.0, config
    )
    five = evaluate_transition(
        previous, off_path, [0.0, 0.0], [0.0, 0.0], 1.0, config
    )

    assert zero.components["cross_track"] == pytest.approx(0.0)
    assert five.components["cross_track"] == pytest.approx(-0.10)
    assert five.reward < zero.reward
    assert zero.reward == pytest.approx(sum(zero.components.values()))
    assert five.reward == pytest.approx(sum(five.components.values()))


def test_progress_diagnostic_reports_normalization_saturation():
    config = EnvironmentConfig(progress_scale_m=1.5)
    result = evaluate_transition(
        make_state(y_m=0.0, roll_deg=0.0),
        make_state(sim_time_s=11.0, y_m=2.0, roll_deg=0.0),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert result.diagnostics["progress_normalized"] == pytest.approx(1.0)
    assert result.diagnostics["progress_saturated"] == pytest.approx(1.0)


def test_waypoint_handoff_does_not_compare_distances_to_different_targets():
    config = EnvironmentConfig()
    previous = make_state(
        y_m=0.0,
        roll_deg=0.0,
        target_y_m=1.0,
        waypoint_index=0,
    )
    current = make_state(
        sim_time_s=11.0,
        y_m=1.0,
        roll_deg=0.0,
        target_y_m=100.0,
        waypoint_index=1,
    )

    result = evaluate_transition(
        previous,
        current,
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert result.components["progress"] == pytest.approx(1.0)
    assert result.components["waypoint"] == config.reward.waypoint_bonus
    assert result.reward == pytest.approx(11.0)


def test_residual_effort_reduces_reward():
    config = EnvironmentConfig()
    previous = make_state()
    current = make_state(sim_time_s=11.0)

    zero = evaluate_transition(
        previous, current, [0.0, 0.0], [0.0, 0.0], 1.0, config
    )
    effort = evaluate_transition(
        previous,
        current,
        [0.0, 0.0],
        [
            config.rudder_residual_limit_rad,
            config.sail_residual_limit_rad,
        ],
        1.0,
        config,
    )

    assert effort.reward < zero.reward


def test_mission_completion_terminates_with_bonus():
    config = EnvironmentConfig()
    result = evaluate_transition(
        make_state(),
        make_state(sim_time_s=11.0, mission_complete=True),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert result.terminated
    assert not result.truncated
    assert result.reason == "mission_complete"
    assert result.components["mission"] == config.reward.mission_bonus


def test_terminal_accuracy_rewards_smaller_final_distance():
    config = EnvironmentConfig(success_radius_m=5.0)
    previous = make_state(
        x_m=0.0,
        y_m=100.0,
        target_y_m=100.0,
        roll_deg=0.0,
    )
    exact = evaluate_transition(
        previous,
        make_state(
            sim_time_s=11.0,
            x_m=0.0,
            y_m=100.0,
            target_y_m=100.0,
            roll_deg=0.0,
            mission_complete=True,
        ),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )
    radius = evaluate_transition(
        previous,
        make_state(
            sim_time_s=11.0,
            x_m=5.0,
            y_m=100.0,
            target_y_m=100.0,
            roll_deg=0.0,
            mission_complete=True,
        ),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert exact.components["terminal_accuracy"] == pytest.approx(10.0)
    assert radius.components["terminal_accuracy"] == pytest.approx(0.0)
    assert exact.reward == pytest.approx(sum(exact.components.values()))
    assert radius.reward == pytest.approx(sum(radius.components.values()))


def test_timeout_truncates_episode():
    config = EnvironmentConfig(episode_timeout_s=20.0)
    result = evaluate_transition(
        make_state(),
        make_state(sim_time_s=30.0),
        [0.0, 0.0],
        [0.0, 0.0],
        20.0,
        config,
    )

    assert not result.terminated
    assert result.truncated
    assert result.reason == "timeout"


def test_excessive_roll_terminates_episode():
    config = EnvironmentConfig(max_roll_deg=45.0)
    result = evaluate_transition(
        make_state(),
        make_state(sim_time_s=11.0, roll_deg=46.0),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert result.terminated
    assert result.reason == "excessive_roll"


def test_backend_authority_defers_local_roll_and_timeout_checks():
    config = EnvironmentConfig(
        max_roll_deg=45.0,
        episode_timeout_s=20.0,
        backend_authoritative_termination=True,
    )
    result = evaluate_transition(
        make_state(),
        make_state(sim_time_s=30.0, roll_deg=46.0),
        [0.0, 0.0],
        [0.0, 0.0],
        20.0,
        config,
    )

    assert not result.terminated
    assert not result.truncated
    assert result.reason == ""
    assert result.components["excessive_roll"] < 0.0


def test_backend_terminal_failure_terminates_episode():
    config = EnvironmentConfig()
    result = evaluate_transition(
        make_state(),
        make_state(sim_time_s=11.0, termination_reason="no_progress"),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert result.terminated
    assert not result.truncated
    assert result.reason == "no_progress"


def test_backend_infrastructure_stop_truncates_episode():
    config = EnvironmentConfig()
    result = evaluate_transition(
        make_state(),
        make_state(
            sim_time_s=11.0,
            termination_reason="runner_process_exit",
            termination_truncated=True,
        ),
        [0.0, 0.0],
        [0.0, 0.0],
        1.0,
        config,
    )

    assert not result.terminated
    assert result.truncated
    assert result.reason == "runner_process_exit"
