import math

import pytest

from experiments.sailboat_performance.steering_rate import (
    summarize_turn_rate_steps,
    symmetric_turn_rate_sequence,
)


def sample(step, elapsed, target, actual, rudder_deg, *, speed=1.0):
    return {
        "step_index": step,
        "step_elapsed_s": elapsed,
        "target_turn_rate_deg_s": target,
        "mavlink_yaw_rate_deg_s": actual,
        "gazebo_angular_yaw_rate_deg_s": actual - 0.2,
        "final_rudder_rad": math.radians(rudder_deg),
        "actual_rudder_angle_rad": math.radians(rudder_deg - 0.1),
        "speed_over_ground_mps": speed,
        "telemetry_valid": True,
    }


def test_symmetric_sequence_has_zero_net_commanded_heading():
    sequence = symmetric_turn_rate_sequence(12.0, 3.0, 5.0)
    assert sequence == ((12.0, 3.0), (-12.0, 6.0), (12.0, 3.0), (0.0, 5.0))
    assert sum(rate * duration for rate, duration in sequence) == pytest.approx(0.0)


def test_summary_reports_tracking_saturation_symmetry_and_zero_settling():
    records = [
        sample(0, 0.0, 10.0, 0.0, 2.0),
        sample(0, 0.5, 10.0, 9.5, 5.0),
        sample(0, 1.0, 10.0, 11.0, 6.0),
        sample(1, 0.0, -10.0, 4.0, -2.0),
        sample(1, 0.5, -10.0, -9.5, -5.0),
        sample(1, 1.0, -10.0, -11.0, -6.0),
        sample(2, 0.0, 0.0, 5.0, 1.0),
        sample(2, 0.5, 0.0, 1.0, 0.5),
        sample(2, 1.0, 0.0, 0.5, 0.0),
        sample(2, 1.5, 0.0, 0.2, 0.0),
    ]

    summary = summarize_turn_rate_steps(records)

    assert summary["valid_sample_count"] == 10
    assert summary["positive_peak_mean_deg_s"] == pytest.approx(11.0)
    assert summary["negative_peak_mean_deg_s"] == pytest.approx(11.0)
    assert summary["reversal_peak_asymmetry_fraction"] == pytest.approx(0.0)
    assert summary["steps"][0]["rise_time_90_s"] == pytest.approx(0.5)
    assert summary["steps"][0]["rudder_saturation_fraction"] == pytest.approx(0.0)
    assert summary["steps"][2]["zero_rate_settling_time_s"] == pytest.approx(0.5)


def test_invalid_rate_sequence_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        symmetric_turn_rate_sequence(0.0, 3.0, 5.0)
