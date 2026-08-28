"""Pure wind-frame calculations for the sailboat physics diagnostic.

The Gazebo world uses ENU coordinates (``x=east, y=north, z=up``).  The
sailboat hull points along model ``+Y`` and model ``+X`` is starboard.  Wind
vectors in the world file point *towards* the direction of travel, while
ArduPilot's wind direction is the direction the wind comes *from*.

This module deliberately has no Gazebo, ROS or MAVLink imports so the frame
contract and result gates can be unit-tested without a running simulator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Sequence

from .frames import attitude_error_deg, normalize_degrees_360


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float = 0.0

    def horizontal_speed(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class WindSample:
    """One raw anemometer sample with the latest MAVLink wind estimate."""

    elapsed_wall_s: float
    sim_time_s: float | None
    anemometer_x_mps: float
    anemometer_y_mps: float
    anemometer_z_mps: float
    mav_wind_direction_deg: float | None = None
    mav_wind_speed_mps: float | None = None
    mav_wind_speed_z_mps: float | None = None
    mav_heading_deg: float | None = None


@dataclass(frozen=True)
class SailPluginWindSample:
    """Wind and aerodynamic input evaluated by ``SailLiftDragSystem``."""

    sim_time_s: float
    source: str
    force_scale: float
    wind_x_mps: float
    wind_y_mps: float
    wind_z_mps: float
    apparent_x_mps: float
    apparent_y_mps: float
    apparent_z_mps: float
    aerodynamic_speed_mps: float
    alpha_deg: float
    lift_coefficient: float
    drag_coefficient: float


def vector_error_mps(actual: Vector3, expected: Vector3) -> float:
    return math.sqrt(
        (actual.x - expected.x) ** 2
        + (actual.y - expected.y) ** 2
        + (actual.z - expected.z) ** 2
    )


def world_enu_to_model_sensor(
    wind_world: Vector3,
    heading_deg: float,
) -> Vector3:
    """Rotate a world ENU vector into this sailboat's model/sensor axes.

    At heading 0 degrees, model ``+Y`` points north and model ``+X`` points
    east.  At heading 90 degrees, ``+Y`` points east and ``+X`` points south.
    The anemometer link has no additional rotation relative to the model.
    """

    heading_rad = math.radians(float(heading_deg))
    cos_heading = math.cos(heading_rad)
    sin_heading = math.sin(heading_rad)
    return Vector3(
        x=wind_world.x * cos_heading - wind_world.y * sin_heading,
        y=wind_world.x * sin_heading + wind_world.y * cos_heading,
        z=wind_world.z,
    )


def wind_to_compass_deg(wind_world: Vector3) -> float:
    """Return compass bearing of an ENU wind vector's destination."""

    if wind_world.horizontal_speed() <= 1e-12:
        return 0.0
    return normalize_degrees_360(math.degrees(math.atan2(wind_world.x, wind_world.y)))


def wind_from_compass_deg(wind_world: Vector3) -> float:
    """Return meteorological/MAVLink direction from which wind arrives."""

    return normalize_degrees_360(wind_to_compass_deg(wind_world) + 180.0)


def _mean_vector(vectors: Sequence[Vector3]) -> Vector3 | None:
    if not vectors:
        return None
    return Vector3(
        x=fmean(vector.x for vector in vectors),
        y=fmean(vector.y for vector in vectors),
        z=fmean(vector.z for vector in vectors),
    )


def _circular_mean_deg(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sine = fmean(math.sin(math.radians(value)) for value in values)
    cosine = fmean(math.cos(math.radians(value)) for value in values)
    if abs(sine) <= 1e-12 and abs(cosine) <= 1e-12:
        return normalize_degrees_360(values[0])
    return normalize_degrees_360(math.degrees(math.atan2(sine, cosine)))


def _range(values: Sequence[float]) -> float:
    return max(values) - min(values) if values else 0.0


def summarize_wind_case(
    samples: Sequence[WindSample],
    plugin_samples: Sequence[SailPluginWindSample],
    *,
    expected_world: Vector3,
    expected_heading_deg: float,
    dynamic: bool = False,
    minimum_samples: int = 8,
    minimum_plugin_samples: int = 2,
    vector_tolerance_mps: float = 1.5,
    plugin_tolerance_mps: float = 0.5,
    direction_tolerance_deg: float = 8.0,
    speed_tolerance_mps: float = 1.5,
    heading_tolerance_deg: float = 3.0,
    dynamic_minimum_range_mps: float = 0.5,
) -> dict[str, object]:
    """Summarize one held-pose wind case and evaluate independent gates.

    The raw anemometer is compared with both upstream contracts because the
    asv_sim README describes world-frame output while the ArduPilot Gazebo
    plugin source describes sensor-frame input.  The rotated-heading case
    identifies which contract the installed versions actually implement.
    """

    raw_vectors = [
        Vector3(
            sample.anemometer_x_mps,
            sample.anemometer_y_mps,
            sample.anemometer_z_mps,
        )
        for sample in samples
    ]
    sample_sim_times = [
        float(sample.sim_time_s) for sample in samples if sample.sim_time_s is not None
    ]
    if sample_sim_times:
        # Debug logging starts before MAVLink and the anemometer are ready.  Do
        # not let the WindEffects rise transient from process startup bias a
        # static case; compare only plugin records from the actual sample window.
        window_start = min(sample_sim_times) - 1.0
        window_end = max(sample_sim_times) + 1.0
        selected_plugin_samples = [
            sample
            for sample in plugin_samples
            if window_start <= sample.sim_time_s <= window_end
        ]
    else:
        selected_plugin_samples = list(plugin_samples)
    plugin_vectors = [
        Vector3(sample.wind_x_mps, sample.wind_y_mps, sample.wind_z_mps)
        for sample in selected_plugin_samples
    ]
    expected_sensor = world_enu_to_model_sensor(expected_world, expected_heading_deg)
    raw_mean = _mean_vector(raw_vectors)
    plugin_mean = _mean_vector(plugin_vectors)

    raw_world_error = (
        vector_error_mps(raw_mean, expected_world) if raw_mean is not None else None
    )
    raw_sensor_error = (
        vector_error_mps(raw_mean, expected_sensor) if raw_mean is not None else None
    )
    plugin_world_error = (
        vector_error_mps(plugin_mean, expected_world) if plugin_mean is not None else None
    )
    plugin_direction_error = (
        abs(
            attitude_error_deg(
                wind_to_compass_deg(plugin_mean),
                wind_to_compass_deg(expected_world),
            )
        )
        if plugin_mean is not None and plugin_mean.horizontal_speed() > 1e-12
        else None
    )

    if raw_world_error is None or raw_sensor_error is None:
        raw_contract = "unavailable"
    elif abs(raw_world_error - raw_sensor_error) <= 0.1:
        raw_contract = "ambiguous_aligned"
    elif raw_world_error < raw_sensor_error:
        raw_contract = "world_enu"
    else:
        raw_contract = "model_sensor"

    raw_contract_error = None
    if raw_mean is not None:
        if dynamic and expected_world.horizontal_speed() > 1e-12:
            # Only magnitude varies in the dynamic case.  Scale both expected
            # frame hypotheses to the observed mean speed so this gate tests
            # axes/signs without incorrectly demanding the configured mean.
            scale = raw_mean.horizontal_speed() / expected_world.horizontal_speed()
            expected_world_for_contract = Vector3(
                expected_world.x * scale,
                expected_world.y * scale,
                expected_world.z * scale,
            )
            expected_sensor_for_contract = Vector3(
                expected_sensor.x * scale,
                expected_sensor.y * scale,
                expected_sensor.z * scale,
            )
            raw_contract_error = min(
                vector_error_mps(raw_mean, expected_world_for_contract),
                vector_error_mps(raw_mean, expected_sensor_for_contract),
            )
        elif raw_world_error is not None and raw_sensor_error is not None:
            raw_contract_error = min(raw_world_error, raw_sensor_error)

    mav_directions = [
        float(sample.mav_wind_direction_deg)
        for sample in samples
        if sample.mav_wind_direction_deg is not None
    ]
    mav_speeds = [
        float(sample.mav_wind_speed_mps)
        for sample in samples
        if sample.mav_wind_speed_mps is not None
    ]
    mav_headings = [
        float(sample.mav_heading_deg)
        for sample in samples
        if sample.mav_heading_deg is not None
    ]
    mav_direction_mean = _circular_mean_deg(mav_directions)
    mav_speed_mean = fmean(mav_speeds) if mav_speeds else None
    mav_heading_mean = _circular_mean_deg(mav_headings)
    expected_from_deg = wind_from_compass_deg(expected_world)
    direction_error = (
        abs(attitude_error_deg(mav_direction_mean, expected_from_deg))
        if mav_direction_mean is not None
        else None
    )
    speed_error = (
        abs(mav_speed_mean - expected_world.horizontal_speed())
        if mav_speed_mean is not None
        else None
    )
    heading_error = (
        abs(attitude_error_deg(mav_heading_mean, expected_heading_deg))
        if mav_heading_mean is not None
        else None
    )

    raw_speeds = [vector.horizontal_speed() for vector in raw_vectors]
    # Dynamic propagation must be checked at the actual input to the
    # lift-drag model, not merely at the selected world-wind component. This
    # also proves that a force-gated diagnostic still executed Compute().
    plugin_speeds = [
        math.hypot(sample.apparent_x_mps, sample.apparent_y_mps)
        for sample in selected_plugin_samples
    ]
    plugin_aerodynamic_speeds = [
        sample.aerodynamic_speed_mps for sample in selected_plugin_samples
    ]
    plugin_sources = sorted({sample.source for sample in selected_plugin_samples})
    component_source_ok = bool(plugin_sources) and all(
        source in {"world_component", "world_seed"} for source in plugin_sources
    )

    gates: dict[str, bool] = {
        "minimum_anemometer_samples": len(samples) >= int(minimum_samples),
        "minimum_sail_plugin_samples": (
            len(selected_plugin_samples) >= int(minimum_plugin_samples)
        ),
        "anemometer_matches_known_contract": (
            raw_contract_error is not None
            and raw_contract_error <= float(vector_tolerance_mps)
        ),
        "sail_plugin_uses_gazebo_wind_component": component_source_ok,
        "mavlink_wind_available": bool(mav_directions) and bool(mav_speeds),
        "mavlink_heading_aligned": (
            heading_error is not None and heading_error <= float(heading_tolerance_deg)
        ),
        "mavlink_wind_direction": (
            direction_error is not None
            and direction_error <= float(direction_tolerance_deg)
        ),
    }

    if dynamic:
        gates.update(
            {
                "anemometer_wind_changes": (
                    _range(raw_speeds) >= float(dynamic_minimum_range_mps)
                ),
                "sail_plugin_wind_changes": (
                    _range(plugin_speeds) >= float(dynamic_minimum_range_mps)
                ),
                "sail_plugin_uses_live_world_component": (
                    plugin_sources == ["world_component"]
                ),
                "sail_plugin_wind_direction": (
                    plugin_direction_error is not None
                    and plugin_direction_error <= float(direction_tolerance_deg)
                ),
                "mavlink_wind_changes": (
                    _range(mav_speeds) >= float(dynamic_minimum_range_mps)
                ),
            }
        )
    else:
        gates.update(
            {
                "sail_plugin_matches_configured_world": (
                    plugin_world_error is not None
                    and plugin_world_error <= float(plugin_tolerance_mps)
                ),
                "mavlink_wind_speed": (
                    speed_error is not None
                    and speed_error <= float(speed_tolerance_mps)
                ),
            }
        )

    return {
        "passed": all(gates.values()),
        "dynamic": bool(dynamic),
        "sample_count": len(samples),
        "mavlink_sample_count": min(len(mav_directions), len(mav_speeds)),
        "sail_plugin_sample_count": len(selected_plugin_samples),
        "sail_plugin_total_log_count": len(plugin_samples),
        "expected": {
            "world_enu_mps": asdict(expected_world),
            "model_sensor_mps": asdict(expected_sensor),
            "heading_deg": normalize_degrees_360(expected_heading_deg),
            "wind_from_deg": expected_from_deg,
            "horizontal_speed_mps": expected_world.horizontal_speed(),
        },
        "mean": {
            "anemometer_mps": asdict(raw_mean) if raw_mean is not None else None,
            "sail_plugin_world_mps": (
                asdict(plugin_mean) if plugin_mean is not None else None
            ),
            "sail_plugin_apparent_horizontal_speed_mps": (
                fmean(plugin_speeds) if plugin_speeds else None
            ),
            "sail_plugin_aerodynamic_speed_mps": (
                fmean(plugin_aerodynamic_speeds)
                if plugin_aerodynamic_speeds
                else None
            ),
            "mav_wind_direction_deg": mav_direction_mean,
            "mav_wind_speed_mps": mav_speed_mean,
            "mav_heading_deg": mav_heading_mean,
        },
        "error": {
            "anemometer_vs_world_mps": raw_world_error,
            "anemometer_vs_sensor_mps": raw_sensor_error,
            "anemometer_contract_mps": raw_contract_error,
            "sail_plugin_vs_world_mps": plugin_world_error,
            "sail_plugin_direction_deg": plugin_direction_error,
            "mav_direction_deg": direction_error,
            "mav_speed_mps": speed_error,
            "mav_heading_deg": heading_error,
        },
        "range": {
            "anemometer_horizontal_speed_mps": _range(raw_speeds),
            "sail_plugin_horizontal_speed_mps": _range(plugin_speeds),
            "mav_wind_speed_mps": _range(mav_speeds),
        },
        "anemometer_frame_contract": raw_contract,
        "sail_plugin_sources": plugin_sources,
        "gates": gates,
    }
