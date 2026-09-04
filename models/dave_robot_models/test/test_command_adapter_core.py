import ast
import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "command_adapter_core.py"
ADAPTER_PATH = Path(__file__).parents[1] / "scripts" / "sailboat_command_adapter.py"
ROBOT_CONFIG_PATH = (
    Path(__file__).parents[1] / "config" / "sailboat" / "robot_config.py"
)
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


@pytest.mark.parametrize(
    ("base_sheet_deg", "delta_sheet_deg", "residual_enabled", "expected_deg"),
    [
        (20.0, 0.0, True, 20.0),
        (20.0, 5.0, True, 25.0),
        (2.0, -5.0, True, 0.0),
        (43.0, 5.0, True, 45.0),
        (20.0, 5.0, False, 20.0),
    ],
)
def test_sheet_allowance_mixer_preserves_unsigned_physical_range(
    base_sheet_deg,
    delta_sheet_deg,
    residual_enabled,
    expected_deg,
):
    result = CORE.mix_command(
        math.radians(base_sheet_deg),
        math.radians(delta_sheet_deg),
        residual_enabled=residual_enabled,
        command_min=0.0,
        command_max=math.pi / 4,
        residual_limit=math.radians(5.0),
    )

    assert result == pytest.approx(math.radians(expected_deg))


def _numeric_literal(node):
    value = ast.literal_eval(node)
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric literal, got {value!r}")
    return float(value)


def _adapter_default(parameter_name):
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "declare_parameter" or len(node.args) < 2:
            continue
        if ast.literal_eval(node.args[0]) == parameter_name:
            return _numeric_literal(node.args[1])
    raise AssertionError(f"missing adapter parameter {parameter_name}")


def _launch_parameter_values():
    tree = ast.parse(ROBOT_CONFIG_PATH.read_text(encoding="utf-8"))
    required = {"sail_min_rad", "sail_max_rad", "rudder_min_rad", "rudder_max_rad"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        values = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                continue
            try:
                name = ast.literal_eval(key)
            except (ValueError, TypeError):
                continue
            if name in required:
                values[name] = _numeric_literal(value)
        if required.issubset(values):
            return values
    raise AssertionError("missing sailboat command-adapter launch parameters")


def test_adapter_and_launch_config_preserve_sheet_and_rudder_contracts():
    launch_parameters = _launch_parameter_values()

    assert _adapter_default("sail_min_rad") == pytest.approx(0.0)
    assert _adapter_default("sail_max_rad") == pytest.approx(math.pi / 4, abs=1e-5)
    assert launch_parameters["sail_min_rad"] == pytest.approx(0.0)
    assert launch_parameters["sail_max_rad"] == pytest.approx(math.pi / 4, abs=1e-5)
    assert _adapter_default("rudder_min_rad") == pytest.approx(-math.pi / 4, abs=1e-5)
    assert _adapter_default("rudder_max_rad") == pytest.approx(math.pi / 4, abs=1e-5)
    assert launch_parameters["rudder_min_rad"] == pytest.approx(-math.pi / 4, abs=1e-5)
    assert launch_parameters["rudder_max_rad"] == pytest.approx(math.pi / 4, abs=1e-5)


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
