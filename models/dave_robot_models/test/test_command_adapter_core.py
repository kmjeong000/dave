import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "command_adapter_core.py"
SPEC = importlib.util.spec_from_file_location("command_adapter_core", MODULE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def test_disabled_residual_is_exact_pass_through():
    result = CORE.mix_command(
        0.321,
        0.05,
        residual_enabled=False,
        command_min=-0.7854,
        command_max=0.7854,
        residual_limit=math.radians(5.0),
    )

    assert result == pytest.approx(0.321)


def test_enabled_residual_is_added():
    result = CORE.mix_command(
        0.2,
        -0.03,
        residual_enabled=True,
        command_min=-0.7854,
        command_max=0.7854,
        residual_limit=math.radians(5.0),
    )

    assert result == pytest.approx(0.17)


def test_residual_and_final_command_are_bounded():
    result = CORE.mix_command(
        0.75,
        1.0,
        residual_enabled=True,
        command_min=-0.7854,
        command_max=0.7854,
        residual_limit=0.05,
    )

    assert result == pytest.approx(0.7854)


def test_rate_limit_bounds_positive_change():
    result = CORE.rate_limit(
        0.08,
        0.0,
        max_rate=0.2,
        elapsed_s=0.1,
    )

    assert result == pytest.approx(0.02)


def test_rate_limit_bounds_negative_change():
    result = CORE.rate_limit(
        -0.08,
        0.04,
        max_rate=0.2,
        elapsed_s=0.1,
    )

    assert result == pytest.approx(0.02)


def test_rate_limit_reaches_nearby_target_without_overshoot():
    result = CORE.rate_limit(
        0.01,
        0.0,
        max_rate=0.2,
        elapsed_s=0.1,
    )

    assert result == pytest.approx(0.01)


def test_disabled_diagnostic_override_preserves_controller_command():
    result = CORE.select_diagnostic_override(
        -0.2,
        0.4,
        override_enabled=False,
        command_min=-0.7854,
        command_max=0.7854,
    )

    assert result == pytest.approx(-0.2)


def test_enabled_diagnostic_override_selects_bounded_explicit_command():
    result = CORE.select_diagnostic_override(
        -0.2,
        2.0,
        override_enabled=True,
        command_min=-0.7854,
        command_max=0.7854,
    )

    assert result == pytest.approx(0.7854)


@pytest.mark.parametrize(
    ("max_rate", "elapsed_s"),
    [(-0.1, 0.1), (0.1, -0.1)],
)
def test_rate_limit_rejects_negative_limits(max_rate, elapsed_s):
    with pytest.raises(ValueError):
        CORE.rate_limit(
            0.1,
            0.0,
            max_rate=max_rate,
            elapsed_s=elapsed_s,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_commands_are_rejected(value):
    with pytest.raises(ValueError):
        CORE.mix_command(
            value,
            0.0,
            residual_enabled=True,
            command_min=-0.7854,
            command_max=0.7854,
            residual_limit=0.05,
        )
