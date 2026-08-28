from __future__ import annotations

import pytest

from experiments.sailboat_physics.wind import (
    SailPluginWindSample,
    Vector3,
    WindSample,
    summarize_wind_case,
    wind_from_compass_deg,
    wind_to_compass_deg,
    world_enu_to_model_sensor,
)


def make_sample(
    *,
    x: float = 0.0,
    y: float = 8.0,
    direction: float = 180.0,
    speed: float = 8.0,
    heading: float = 0.0,
    index: int = 0,
) -> WindSample:
    return WindSample(
        elapsed_wall_s=float(index),
        sim_time_s=20.0 + index,
        anemometer_x_mps=x,
        anemometer_y_mps=y,
        anemometer_z_mps=0.0,
        mav_wind_direction_deg=direction,
        mav_wind_speed_mps=speed,
        mav_wind_speed_z_mps=0.0,
        mav_heading_deg=heading,
    )


def make_plugin(
    *,
    x: float = 0.0,
    y: float = 8.0,
    apparent_x: float | None = None,
    apparent_y: float | None = None,
    index: int = 0,
) -> SailPluginWindSample:
    apparent_x = x if apparent_x is None else apparent_x
    apparent_y = y if apparent_y is None else apparent_y
    return SailPluginWindSample(
        sim_time_s=20.0 + index,
        source="world_component",
        force_scale=0.0,
        wind_x_mps=x,
        wind_y_mps=y,
        wind_z_mps=0.0,
        apparent_x_mps=apparent_x,
        apparent_y_mps=apparent_y,
        apparent_z_mps=0.0,
        aerodynamic_speed_mps=(apparent_x**2 + apparent_y**2) ** 0.5,
        alpha_deg=10.0,
        lift_coefficient=0.5,
        drag_coefficient=0.1,
    )


@pytest.mark.parametrize(
    ("vector", "to_deg", "from_deg"),
    [
        (Vector3(0.0, 8.0), 0.0, 180.0),
        (Vector3(8.0, 0.0), 90.0, 270.0),
        (Vector3(0.0, -8.0), 180.0, 0.0),
        (Vector3(-8.0, 0.0), 270.0, 90.0),
    ],
)
def test_wind_compass_conventions(vector, to_deg, from_deg):
    assert wind_to_compass_deg(vector) == pytest.approx(to_deg)
    assert wind_from_compass_deg(vector) == pytest.approx(from_deg)


def test_world_to_model_sensor_uses_plus_y_bow_and_plus_x_starboard():
    north = Vector3(0.0, 8.0, 0.0)

    heading_north = world_enu_to_model_sensor(north, 0.0)
    heading_east = world_enu_to_model_sensor(north, 90.0)

    assert heading_north == Vector3(0.0, 8.0, 0.0)
    assert heading_east.x == pytest.approx(-8.0)
    assert heading_east.y == pytest.approx(0.0, abs=1e-12)
    assert heading_east.z == 0.0


def test_static_summary_passes_and_reports_world_contract():
    samples = [make_sample(index=index) for index in range(8)]
    plugin = [make_plugin(index=index) for index in range(3)]

    summary = summarize_wind_case(
        samples,
        plugin,
        expected_world=Vector3(0.0, 8.0, 0.0),
        expected_heading_deg=0.0,
    )

    assert summary["passed"] is True
    assert summary["anemometer_frame_contract"] == "ambiguous_aligned"
    assert all(summary["gates"].values())
    assert summary["error"]["mav_direction_deg"] == pytest.approx(0.0)


def test_rotated_case_distinguishes_sensor_frame_contract():
    samples = [
        make_sample(
            x=-8.0,
            y=0.0,
            direction=180.0,
            speed=8.0,
            heading=90.0,
            index=index,
        )
        for index in range(8)
    ]
    plugin = [make_plugin(index=index) for index in range(3)]

    summary = summarize_wind_case(
        samples,
        plugin,
        expected_world=Vector3(0.0, 8.0, 0.0),
        expected_heading_deg=90.0,
    )

    assert summary["passed"] is True
    assert summary["anemometer_frame_contract"] == "model_sensor"
    assert summary["error"]["anemometer_vs_sensor_mps"] == pytest.approx(0.0)
    assert summary["error"]["anemometer_vs_world_mps"] > 10.0


def test_static_summary_fails_stale_sail_plugin_wind():
    samples = [make_sample(index=index) for index in range(8)]
    plugin = [make_plugin(x=0.0, y=3.0, index=index) for index in range(3)]

    summary = summarize_wind_case(
        samples,
        plugin,
        expected_world=Vector3(0.0, 8.0, 0.0),
        expected_heading_deg=0.0,
    )

    assert summary["passed"] is False
    assert summary["gates"]["sail_plugin_matches_configured_world"] is False


def test_dynamic_summary_requires_all_three_paths_to_change():
    speeds = [6.0, 7.0, 8.0, 9.0, 10.0, 9.0, 8.0, 7.0]
    samples = [
        make_sample(y=speed, speed=speed, index=index)
        for index, speed in enumerate(speeds)
    ]
    plugin = [
        make_plugin(y=8.0, apparent_y=speed, index=index)
        for index, speed in enumerate([6.0, 8.0, 10.0])
    ]

    summary = summarize_wind_case(
        samples,
        plugin,
        expected_world=Vector3(0.0, 8.0, 0.0),
        expected_heading_deg=0.0,
        dynamic=True,
    )

    assert summary["passed"] is True
    assert summary["gates"]["anemometer_wind_changes"] is True
    assert summary["gates"]["sail_plugin_wind_changes"] is True
    assert summary["gates"]["mavlink_wind_changes"] is True


def test_dynamic_summary_fails_when_mavlink_value_is_stale():
    samples = [
        make_sample(y=6.0 + index, speed=8.0, index=index)
        for index in range(8)
    ]
    plugin = [make_plugin(y=6.0 + index, index=index) for index in range(3)]

    summary = summarize_wind_case(
        samples,
        plugin,
        expected_world=Vector3(0.0, 8.0, 0.0),
        expected_heading_deg=0.0,
        dynamic=True,
    )

    assert summary["passed"] is False
    assert summary["gates"]["mavlink_wind_changes"] is False
