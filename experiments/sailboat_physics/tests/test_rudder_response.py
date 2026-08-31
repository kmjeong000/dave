from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

from experiments.sailboat_physics.rudder_response import (
    AlignedRudderSample,
    RudderCommandSample,
    align_rudder_samples,
    build_rudder_command_samples,
    parse_rudder_wrench,
    summarize_rudder_response,
)


def make_aligned(
    *,
    sim_time_s: float,
    command_deg: float,
    actual_angle_deg: float,
    force_x_n: float,
    yaw_rate_deg_s: float,
) -> AlignedRudderSample:
    # With r=(0, -1.5, 0), F=(Fx, 0, 0), r x F=(0, 0, 1.5 Fx).
    yaw_moment_nm = 1.5 * force_x_n
    return AlignedRudderSample(
        trial_id="trial",
        sim_time_s=sim_time_s,
        alignment_delta_s=0.01,
        command_deg=command_deg,
        actual_angle_deg=actual_angle_deg,
        speed_mps=1.0,
        force_world_x_n=force_x_n,
        force_world_y_n=0.0,
        force_world_z_n=0.0,
        lever_world_x_m=0.0,
        lever_world_y_m=-1.5,
        lever_world_z_m=0.0,
        moment_world_x_nm=0.0,
        moment_world_y_nm=0.0,
        moment_world_z_nm=yaw_moment_nm,
        yaw_axis_world_x=0.0,
        yaw_axis_world_y=0.0,
        yaw_axis_world_z=1.0,
        yaw_moment_nm=yaw_moment_nm,
        yaw_rate_deg_s=yaw_rate_deg_s,
    )


def test_parse_rudder_wrench_accepts_rate_limited_diagnostic_line():
    text = """
[gazebo-1] \x1b[1;31m[FoilLiftDragSystem] link=rudder_link simTimeS=42.5 Vfluid=-1 0 0 speed=1 alpha_deg=12 cl=0.2 cd=0.05 lift=0 0 0 drag=0 0 0 forceWorld=2 0 0 jointAngleDeg=20 commandDeg=18 cpWorld=0 -1.5 0 leverFromBaseWorld=0 -1.5 0 momentAboutBaseWorld=0 0 3 yawAxisWorld=0 0 1 yawMomentNm=3\x1b[0m
"""

    records = parse_rudder_wrench(text)

    assert len(records) == 1
    record = records[0]
    assert record.sim_time_s == pytest.approx(42.5)
    assert record.joint_angle_deg == pytest.approx(20.0)
    assert record.command_deg == pytest.approx(18.0)
    assert record.force_world_x_n == pytest.approx(2.0)
    assert record.yaw_moment_nm == pytest.approx(3.0)


def test_command_rows_compute_wrapped_central_yaw_rate():
    rows = list(
        csv.DictReader(
            io.StringIO(
                "sim_time_s,rudder_cmd_rad,yaw_rad\n"
                "0.0,-0.2,3.124139361\n"
                "1.0,0.0,-3.124139361\n"
                "2.0,0.2,-3.089232776\n"
            )
        )
    )

    samples = build_rudder_command_samples(rows)

    assert samples[1].command_deg == pytest.approx(0.0)
    assert samples[1].yaw_rate_deg_s == pytest.approx(2.0, abs=1e-5)


def test_align_uses_nearest_command_within_time_limit():
    wrench = parse_rudder_wrench(
        "[FoilLiftDragSystem] link=rudder_link simTimeS=10.2 "
        "Vfluid=-1 0 0 speed=1 alpha_deg=12 cl=0.2 cd=0.05 "
        "forceWorld=2 0 0 jointAngleDeg=20 commandDeg=17 "
        "cpWorld=0 -1.5 0 "
        "leverFromBaseWorld=0 -1.5 0 momentAboutBaseWorld=0 0 3 "
        "yawAxisWorld=0 0 1 yawMomentNm=3"
    )
    commands = [
        RudderCommandSample(10.0, 15.0, 3.0),
        RudderCommandSample(11.0, -15.0, -3.0),
    ]

    aligned = align_rudder_samples(
        "trial-a",
        wrench,
        commands,
        maximum_time_delta_s=0.25,
    )

    assert len(aligned) == 1
    assert aligned[0].trial_id == "trial-a"
    assert aligned[0].command_deg == pytest.approx(17.0)
    assert aligned[0].alignment_delta_s == pytest.approx(0.2)


def test_summary_passes_for_bilateral_command_angle_and_yaw_moment_chain():
    samples = [
        make_aligned(
            sim_time_s=float(index),
            command_deg=-20.0,
            actual_angle_deg=-19.0,
            force_x_n=-2.0,
            yaw_rate_deg_s=-4.0,
        )
        for index in range(3)
    ] + [
        make_aligned(
            sim_time_s=3.0 + index,
            command_deg=20.0,
            actual_angle_deg=19.0,
            force_x_n=2.0,
            yaw_rate_deg_s=4.0,
        )
        for index in range(3)
    ]

    summary = summarize_rudder_response(samples)

    assert summary["command_actual_sign_agreement_ratio"] == pytest.approx(1.0)
    assert summary["negative_rudder_mean_yaw_moment_nm"] == pytest.approx(-3.0)
    assert summary["positive_rudder_mean_yaw_moment_nm"] == pytest.approx(3.0)
    assert summary["gates"]["actual_rudder_tracks_command_sign"] is True
    assert summary["gates"]["yaw_moment_reverses_with_rudder_side"] is True
    assert summary["passed"] is True


def test_summary_does_not_pass_with_only_one_command_side():
    samples = [
        make_aligned(
            sim_time_s=float(index),
            command_deg=-20.0,
            actual_angle_deg=-19.0,
            force_x_n=-2.0,
            yaw_rate_deg_s=-4.0,
        )
        for index in range(4)
    ]

    summary = summarize_rudder_response(samples)

    assert summary["gates"]["minimum_samples_each_command_side"] is False
    assert summary["passed"] is False


def test_rudder_foil_enables_base_referenced_yaw_diagnostics():
    repo_root = Path(__file__).resolve().parents[3]
    model_sdf = (
        repo_root
        / "models"
        / "dave_robot_models"
        / "description"
        / "sailboat"
        / "model.sdf"
    )
    model_text = model_sdf.read_text(encoding="utf-8")
    plugin_blocks = re.findall(
        r"<plugin\b[^>]*\bfilename=['\"]libFoilLiftDragSystem\.so['\"]"
        r"[^>]*>.*?</plugin>",
        model_text,
        flags=re.DOTALL,
    )
    rudder_plugin = next(
        plugin
        for plugin in (ET.fromstring(block) for block in plugin_blocks)
        if plugin.findtext("daggerboard_link") == "rudder_link"
    )

    assert rudder_plugin.findtext("base_link") == "base_link"
    assert tuple(
        float(value) for value in rudder_plugin.findtext("yaw_axis", "").split()
    ) == pytest.approx((0.0, 0.0, 1.0))
    assert rudder_plugin.findtext("command_topic") == (
        "/model/sailboat/joint/rudder_joint/cmd_pos"
    )
