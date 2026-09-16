import pytest

from experiments.sailboat_performance.steady_state import (
    SteadyObservation,
    SteadyStateConfig,
    SteadyStateDetector,
)


def observation(t, **overrides):
    values = {
        "sim_time_s": t,
        "heading_error_deg": 1.0,
        "heading_deg": 0.5,
        "speed_mps": 1.5,
        "roll_deg": 2.0,
        "telemetry_valid": True,
        "wind_settled": True,
        "force_ramp_complete": True,
        "tack_detected": False,
    }
    values.update(overrides)
    return SteadyObservation(**values)


def feed(detector, duration_s, factory=observation):
    result = None
    for index in range(int(duration_s * 10) + 1):
        result = detector.update(factory(index / 10.0))
    return result


def test_stable_case_qualifies_then_holds_for_ten_seconds():
    detector = SteadyStateDetector(SteadyStateConfig())
    result = feed(detector, 30.1)
    assert result is not None
    assert result.qualified
    assert result.ready
    assert result.stable_for_s >= 10.0
    assert result.stats["window_duration_s"] >= 20.0


@pytest.mark.parametrize(
    ("factory", "reason"),
    [
        (
            lambda t: observation(t, heading_deg=(12.0 if int(t) % 2 else 348.0), heading_error_deg=12.0),
            "heading_unstable",
        ),
        (lambda t: observation(t, speed_mps=0.5 + 0.03 * t), "speed_trending"),
        (lambda t: observation(t, roll_deg=0.3 * t), "roll_trending"),
        (lambda t: observation(t, telemetry_valid=(int(t * 10) % 5 != 0)), "telemetry_stale"),
        (lambda t: observation(t, tack_detected=t >= 29.0), "tack_detected"),
    ],
)
def test_unstable_or_invalid_case_does_not_qualify(factory, reason):
    detector = SteadyStateDetector(SteadyStateConfig())
    result = feed(detector, 30.0, factory)
    assert result is not None
    assert not result.qualified
    assert reason in result.reasons


def test_invalid_detector_config_is_rejected():
    with pytest.raises(ValueError):
        SteadyStateConfig.from_mapping({"window_s": 0.0})
    with pytest.raises(ValueError, match="unknown"):
        SteadyStateConfig.from_mapping({"made_up": 1})
