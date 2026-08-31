from __future__ import annotations

import pytest

from experiments.sailboat_physics.tacking_wrench import (
    SailWrenchSample,
    parse_sail_wrench,
    summarize_sail_wrench,
)


def make_sample(
    *,
    sim_time_s: float,
    boom_angle_deg: float,
    force_x_n: float,
    roll_moment_nm: float,
) -> SailWrenchSample:
    return SailWrenchSample(
        sim_time_s=sim_time_s,
        force_scale=0.35,
        boom_angle_deg=boom_angle_deg,
        raw_force_x_n=force_x_n / 0.35,
        raw_force_y_n=0.0,
        raw_force_z_n=0.0,
        applied_force_x_n=force_x_n,
        applied_force_y_n=0.0,
        applied_force_z_n=0.0,
        cp_world_x_m=0.0,
        cp_world_y_m=0.0,
        cp_world_z_m=1.0,
        lever_world_x_m=0.0,
        lever_world_y_m=0.0,
        lever_world_z_m=1.0,
        moment_world_x_nm=0.0,
        moment_world_y_nm=roll_moment_nm,
        moment_world_z_nm=0.0,
        roll_axis_world_x=0.0,
        roll_axis_world_y=1.0,
        roll_axis_world_z=0.0,
        roll_moment_nm=roll_moment_nm,
    )


def test_parse_sail_wrench_accepts_rate_limited_diagnostic_line():
    text = """
[gazebo-1] \x1b[1;31m[SailLiftDragSystem] link=sail_link simTimeS=42.5 windSource=world_component forceScale=0.35 forceLimitScale=1 rawForceN=20 windWorld=0 8 0 Vapp=1 7 0 speed=7.1 alpha_deg=20 cl=0.5 cd=0.1 lift=1 2 0 drag=3 4 0 force=4 6 0 boomAngleDeg=-35 rawForceWorld=10 15 0 appliedForceWorld=4 6 0 cpWorld=0 0 1 leverFromBaseWorld=0 0 1 momentAboutBaseWorld=-6 4 0 rollAxisWorld=0 1 0 rollMomentNm=4\x1b[0m
"""

    records = parse_sail_wrench(text)

    assert len(records) == 1
    record = records[0]
    assert record.sim_time_s == pytest.approx(42.5)
    assert record.boom_angle_deg == pytest.approx(-35.0)
    assert record.applied_force_y_n == pytest.approx(6.0)
    assert record.moment_world_x_nm == pytest.approx(-6.0)
    assert record.roll_moment_nm == pytest.approx(4.0)


def test_summary_passes_when_applied_roll_moment_reverses():
    samples = [
        make_sample(
            sim_time_s=float(index),
            boom_angle_deg=-35.0,
            force_x_n=-10.0,
            roll_moment_nm=-10.0,
        )
        for index in range(3)
    ] + [
        make_sample(
            sim_time_s=3.0 + index,
            boom_angle_deg=35.0,
            force_x_n=10.0,
            roll_moment_nm=10.0,
        )
        for index in range(3)
    ]

    summary = summarize_sail_wrench(samples)

    assert summary["negative_boom_mean_roll_moment_nm"] == pytest.approx(-10.0)
    assert summary["positive_boom_mean_roll_moment_nm"] == pytest.approx(10.0)
    assert summary["gates"]["wrench_math_consistent"] is True
    assert summary["gates"]["roll_moment_reverses_with_boom_side"] is True
    assert summary["passed"] is True


def test_summary_fails_when_both_boom_sides_apply_same_roll_moment():
    samples = [
        make_sample(
            sim_time_s=float(index),
            boom_angle_deg=-35.0,
            force_x_n=10.0,
            roll_moment_nm=10.0,
        )
        for index in range(3)
    ] + [
        make_sample(
            sim_time_s=3.0 + index,
            boom_angle_deg=35.0,
            force_x_n=10.0,
            roll_moment_nm=10.0,
        )
        for index in range(3)
    ]

    summary = summarize_sail_wrench(samples)

    assert summary["gates"]["wrench_math_consistent"] is True
    assert summary["gates"]["roll_moment_reverses_with_boom_side"] is False
    assert summary["passed"] is False
