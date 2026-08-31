#!/usr/bin/env python3

"""Pure command-mixing helpers for the sailboat command adapter."""

import math


def clamp(value, lower, upper):
    if not math.isfinite(value):
        raise ValueError("command values must be finite")
    if lower > upper:
        raise ValueError("lower command limit must not exceed upper limit")
    return min(max(value, lower), upper)


def rate_limit(target, previous, *, max_rate, elapsed_s):
    """Move previous toward target by at most max_rate * elapsed_s."""
    target_value = float(target)
    previous_value = float(previous)
    rate = float(max_rate)
    elapsed = float(elapsed_s)

    if not all(
        math.isfinite(value)
        for value in (target_value, previous_value, rate, elapsed)
    ):
        raise ValueError("rate-limit values must be finite")
    if rate < 0.0:
        raise ValueError("maximum rate must be non-negative")
    if elapsed < 0.0:
        raise ValueError("elapsed time must be non-negative")

    maximum_change = rate * elapsed
    return previous_value + clamp(
        target_value - previous_value,
        -maximum_change,
        maximum_change,
    )


def mix_command(
    base_command,
    residual_command,
    *,
    residual_enabled,
    command_min,
    command_max,
    residual_limit,
):
    """Return a bounded base-plus-residual actuator command."""
    base = float(base_command)
    residual = float(residual_command)
    limit = float(residual_limit)

    if limit < 0.0:
        raise ValueError("residual limit must be non-negative")

    if not residual_enabled:
        residual = 0.0
    residual = clamp(residual, -limit, limit)
    return clamp(base + residual, float(command_min), float(command_max))


def select_diagnostic_override(
    command,
    override_command,
    *,
    override_enabled,
    command_min,
    command_max,
):
    """Select a bounded explicit command when a diagnostic override is enabled."""

    selected = float(override_command) if override_enabled else float(command)
    return clamp(selected, float(command_min), float(command_max))
