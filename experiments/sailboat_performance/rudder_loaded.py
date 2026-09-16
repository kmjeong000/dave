"""Pure helpers for the diagnostic-only loaded-rudder isolation test."""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from experiments.sailboat_physics.rudder_response import RudderWrenchSample


_FOIL_PLUGIN = re.compile(
    r"<plugin\b[^>]*filename=[\"']libFoilLiftDragSystem\.so[\"'][^>]*>"
    r".*?</plugin>",
    re.DOTALL,
)
_RUDDER_DIAGNOSTIC_PLUGIN = re.compile(
    r"<plugin\b[^>]*filename=[\"']libRudderDiagnosticLoadSystem\.so[\"'][^>]*>"
    r".*?</plugin>",
    re.DOTALL,
)
_WATER_VELOCITY = re.compile(
    r"(<water_velocity>\s*)[^<]*(\s*</water_velocity>)"
)
_DEBUG_PERIOD = re.compile(r"(<debug_period>\s*)[^<]*(\s*</debug_period>)")
_DIAGNOSTIC_FLOW_TRANSITION = re.compile(
    r"\[RudderDiagnosticLoadSystem\]\s+diagnostic flow transition:\s+"
    r"link=rudder_link,\s+false\s+->\s+true,\s+"
    r"simTimeS=([-+0-9.eE]+)"
)

DEFAULT_DIAGNOSTIC_FLOW_ENABLE_TOPIC = (
    "/model/sailboat/rudder_test_flow/enable"
)

LOADED_TRACKING_GATES = (
    "telemetry",
    "single_owner_command_delivered",
    "steady_tracking_error",
    "overshoot",
    "rise_reached",
    "settled",
    "velocity_limit_fraction",
)


@dataclass(frozen=True)
class ControlledRudderLoadConfig:
    """Acceptance limits for one diagnostic rudder-only water-flow case."""

    water_velocity_world_enu_mps: tuple[float, float, float]
    diagnostic_flow_enable_topic: str = DEFAULT_DIAGNOSTIC_FLOW_ENABLE_TOPIC
    minimum_foil_samples: int = 5
    relative_speed_tolerance_mps: float = 0.5
    minimum_flow_alignment_cosine: float = 0.7
    minimum_force_magnitude_n: float = 0.5
    minimum_abs_yaw_moment_nm: float = 0.1
    command_tolerance_deg: float = 0.1
    minimum_command_match_fraction: float = 0.8
    expected_yaw_moment_sign: int = -1
    minimum_expected_sign_fraction: float = 0.8
    maximum_sign_heading_deviation_deg: float = 15.0
    wrench_sample_period_s: float = 0.1
    maximum_wrench_error_nm: float = 0.05

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any]
    ) -> "ControlledRudderLoadConfig":
        vector = tuple(
            float(value)
            for value in raw.get("water_velocity_world_enu_mps", ())
        )
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            raise ValueError(
                "load.water_velocity_world_enu_mps must contain three finite values"
            )
        config = cls(
            water_velocity_world_enu_mps=vector,
            diagnostic_flow_enable_topic=str(
                raw.get(
                    "diagnostic_flow_enable_topic",
                    cls.diagnostic_flow_enable_topic,
                )
            ).strip(),
            minimum_foil_samples=int(
                raw.get("minimum_foil_samples", cls.minimum_foil_samples)
            ),
            relative_speed_tolerance_mps=float(
                raw.get(
                    "relative_speed_tolerance_mps",
                    cls.relative_speed_tolerance_mps,
                )
            ),
            minimum_flow_alignment_cosine=float(
                raw.get(
                    "minimum_flow_alignment_cosine",
                    cls.minimum_flow_alignment_cosine,
                )
            ),
            minimum_force_magnitude_n=float(
                raw.get(
                    "minimum_force_magnitude_n",
                    cls.minimum_force_magnitude_n,
                )
            ),
            minimum_abs_yaw_moment_nm=float(
                raw.get(
                    "minimum_abs_yaw_moment_nm",
                    cls.minimum_abs_yaw_moment_nm,
                )
            ),
            command_tolerance_deg=float(
                raw.get("command_tolerance_deg", cls.command_tolerance_deg)
            ),
            minimum_command_match_fraction=float(
                raw.get(
                    "minimum_command_match_fraction",
                    cls.minimum_command_match_fraction,
                )
            ),
            expected_yaw_moment_sign=int(
                raw.get(
                    "expected_yaw_moment_sign",
                    cls.expected_yaw_moment_sign,
                )
            ),
            minimum_expected_sign_fraction=float(
                raw.get(
                    "minimum_expected_sign_fraction",
                    cls.minimum_expected_sign_fraction,
                )
            ),
            maximum_sign_heading_deviation_deg=float(
                raw.get(
                    "maximum_sign_heading_deviation_deg",
                    cls.maximum_sign_heading_deviation_deg,
                )
            ),
            wrench_sample_period_s=float(
                raw.get("wrench_sample_period_s", cls.wrench_sample_period_s)
            ),
            maximum_wrench_error_nm=float(
                raw.get("maximum_wrench_error_nm", cls.maximum_wrench_error_nm)
            ),
        )
        if config.target_speed_mps <= 0.0:
            raise ValueError("loaded rudder water velocity must be non-zero")
        if not config.diagnostic_flow_enable_topic.startswith("/"):
            raise ValueError(
                "load.diagnostic_flow_enable_topic must be an absolute topic"
            )
        if config.minimum_foil_samples <= 0:
            raise ValueError("load.minimum_foil_samples must be positive")
        for name in (
            "relative_speed_tolerance_mps",
            "minimum_force_magnitude_n",
            "minimum_abs_yaw_moment_nm",
            "command_tolerance_deg",
            "maximum_sign_heading_deviation_deg",
            "wrench_sample_period_s",
            "maximum_wrench_error_nm",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"load.{name} must be finite and positive")
        for name in (
            "minimum_flow_alignment_cosine",
            "minimum_command_match_fraction",
            "minimum_expected_sign_fraction",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"load.{name} must be within [0, 1]")
        if config.expected_yaw_moment_sign not in (-1, 1):
            raise ValueError("load.expected_yaw_moment_sign must be -1 or 1")
        if config.maximum_sign_heading_deviation_deg > 180.0:
            raise ValueError(
                "load.maximum_sign_heading_deviation_deg must not exceed 180"
            )
        return config

    @property
    def target_speed_mps(self) -> float:
        return math.sqrt(
            sum(value * value for value in self.water_velocity_world_enu_mps)
        )


def render_rudder_loaded_model(
    source_path: Path,
    output_path: Path,
    water_velocity_world_enu_mps: Sequence[float],
    wrench_sample_period_s: float,
    diagnostic_flow_enable_topic: str = (
        DEFAULT_DIAGNOSTIC_FLOW_ENABLE_TOPIC
    ),
) -> dict[str, Any]:
    """Render a trial-only SDF with a disabled, runtime-gated rudder flow."""

    vector = tuple(float(value) for value in water_velocity_world_enu_mps)
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError("rudder water velocity must contain three finite values")
    source = source_path.read_text(encoding="utf-8")
    matches = list(_FOIL_PLUGIN.finditer(source))
    rudder_matches = [
        match
        for match in matches
        if re.search(
            r"<(?:link_name|foil_link|daggerboard_link)>\s*rudder_link\s*</",
            match.group(0),
        )
    ]
    if len(rudder_matches) != 1:
        raise ValueError(
            "expected exactly one rudder FoilLiftDragSystem plugin, found "
            f"{len(rudder_matches)}"
        )
    rudder_match = rudder_matches[0]
    block = rudder_match.group(0)
    if len(_WATER_VELOCITY.findall(block)) != 1:
        raise ValueError("rudder foil must contain exactly one <water_velocity>")
    if len(_DEBUG_PERIOD.findall(block)) != 1:
        raise ValueError("rudder foil must contain exactly one <debug_period>")
    period = float(wrench_sample_period_s)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("rudder wrench sample period must be finite and positive")
    formatted = " ".join(f"{value:.12g}" for value in vector)
    topic = str(diagnostic_flow_enable_topic).strip()
    if not topic.startswith("/"):
        raise ValueError("diagnostic flow enable topic must be absolute")
    if "<diagnostic_water_velocity>" in block:
        raise ValueError("rudder foil already contains diagnostic flow controls")
    loaded_block = _WATER_VELOCITY.sub(
        lambda match: (
            f"{match.group(1)}0 0 0{match.group(2)}"
            f"\n      <diagnostic_water_velocity>{formatted}"
            "</diagnostic_water_velocity>"
            f"\n      <diagnostic_flow_enable_topic>{topic}"
            "</diagnostic_flow_enable_topic>"
            "\n      <diagnostic_flow_enabled>false"
            "</diagnostic_flow_enabled>"
        ),
        block,
        count=1,
    )
    loaded_block = re.sub(
        r"<plugin\b[^>]*>",
        '<plugin filename="libRudderDiagnosticLoadSystem.so" '
        'name="ioes::sim::RudderDiagnosticLoadSystem">',
        loaded_block,
        count=1,
    )
    loaded_block = _DEBUG_PERIOD.sub(
        lambda match: f"{match.group(1)}{period:.12g}{match.group(2)}",
        loaded_block,
        count=1,
    )
    rendered = (
        source[: rudder_match.start()]
        + loaded_block
        + source[rudder_match.end() :]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    rendered_blocks = [
        *_FOIL_PLUGIN.findall(rendered),
        *_RUDDER_DIAGNOSTIC_PLUGIN.findall(rendered),
    ]
    water_by_link: dict[str, list[float]] = {}
    for item in rendered_blocks:
        link = re.search(
            r"<(?:link_name|foil_link|daggerboard_link)>\s*([^<]+?)\s*</",
            item,
        )
        water = _WATER_VELOCITY.search(item)
        if link is None or water is None:
            continue
        water_by_link[link.group(1)] = [
            float(value) for value in water.group(0).split(">", 1)[1].split("<", 1)[0].split()
        ]
    if water_by_link.get("rudder_link") != [0.0, 0.0, 0.0]:
        raise ValueError("rendered rudder base water velocity must remain zero")
    if water_by_link.get("daggerboard_link") != [0.0, 0.0, 0.0]:
        raise ValueError("diagnostic render must not load the daggerboard")
    return {
        "source_model_sdf": str(source_path),
        "rendered_model_sdf": str(output_path),
        "rudder_base_water_velocity_world_enu_mps": [0.0, 0.0, 0.0],
        "rudder_diagnostic_water_velocity_world_enu_mps": list(vector),
        "diagnostic_flow_enable_topic": topic,
        "diagnostic_flow_initially_enabled": False,
        "diagnostic_plugin": "ioes::sim::RudderDiagnosticLoadSystem",
        "rudder_wrench_sample_period_s": period,
        "daggerboard_water_velocity_world_enu_mps": [0.0, 0.0, 0.0],
        "production_model_modified": False,
    }


def parse_diagnostic_flow_activation_sim_time(text: str) -> float | None:
    """Return the first plugin-confirmed disabled-to-enabled transition time."""

    matches = [
        float(match.group(1))
        for match in _DIAGNOSTIC_FLOW_TRANSITION.finditer(text)
    ]
    return min(matches) if matches else None


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _length(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _heading_deviation_deg(
    heading_deg: float, reference_heading_deg: float
) -> float:
    return abs(
        (float(heading_deg) - float(reference_heading_deg) + 180.0) % 360.0
        - 180.0
    )


def _align_heading_deviations(
    samples: Sequence[RudderWrenchSample],
    heading_samples: Sequence[Mapping[str, Any]],
    reference_heading_deg: float,
) -> list[tuple[RudderWrenchSample, float, float]]:
    """Align each wrench record to the nearest finite measured heading."""

    points = sorted(
        (
            float(item["sim_time_s"]),
            float(item["heading_deg"]),
        )
        for item in heading_samples
        if item.get("sim_time_s") is not None
        and item.get("heading_deg") is not None
        and math.isfinite(float(item["sim_time_s"]))
        and math.isfinite(float(item["heading_deg"]))
    )
    if not points:
        return []

    times = [item[0] for item in points]
    aligned = []
    for sample in samples:
        insertion = bisect.bisect_left(times, sample.sim_time_s)
        candidates = []
        if insertion < len(points):
            candidates.append(points[insertion])
        if insertion > 0:
            candidates.append(points[insertion - 1])
        nearest_time, nearest_heading = min(
            candidates,
            key=lambda item: abs(item[0] - sample.sim_time_s),
        )
        aligned.append(
            (
                sample,
                _heading_deviation_deg(nearest_heading, reference_heading_deg),
                abs(nearest_time - sample.sim_time_s),
            )
        )
    return aligned


def summarize_loaded_rudder_wrench(
    samples: Sequence[RudderWrenchSample],
    *,
    heading_samples: Sequence[Mapping[str, Any]],
    sign_reference_heading_deg: float | None,
    sign_reference_sim_time_s: float | None,
    target_rudder_deg: float,
    config: ControlledRudderLoadConfig,
) -> dict[str, Any]:
    """Summarize relative-water and wrench validity for a loaded step."""

    target_flow = config.water_velocity_world_enu_mps
    target_speed = config.target_speed_mps
    force_magnitude = [
        _length(
            item.force_world_x_n,
            item.force_world_y_n,
            item.force_world_z_n,
        )
        for item in samples
    ]
    relative_speed = [item.speed_mps for item in samples]
    alignments = []
    wrench_errors = []
    projection_errors = []
    for item in samples:
        speed = _length(
            item.water_relative_x_mps,
            item.water_relative_y_mps,
            item.water_relative_z_mps,
        )
        if speed > 1.0e-9:
            alignments.append(
                sum(
                    actual * expected
                    for actual, expected in zip(
                        (
                            item.water_relative_x_mps,
                            item.water_relative_y_mps,
                            item.water_relative_z_mps,
                        ),
                        target_flow,
                    )
                )
                / (speed * target_speed)
            )
        cross = (
            item.lever_world_y_m * item.force_world_z_n
            - item.lever_world_z_m * item.force_world_y_n,
            item.lever_world_z_m * item.force_world_x_n
            - item.lever_world_x_m * item.force_world_z_n,
            item.lever_world_x_m * item.force_world_y_n
            - item.lever_world_y_m * item.force_world_x_n,
        )
        wrench_errors.append(
            _length(
                cross[0] - item.moment_world_x_nm,
                cross[1] - item.moment_world_y_nm,
                cross[2] - item.moment_world_z_nm,
            )
        )
        projection = sum(
            moment * axis
            for moment, axis in zip(
                (
                    item.moment_world_x_nm,
                    item.moment_world_y_nm,
                    item.moment_world_z_nm,
                ),
                (
                    item.yaw_axis_world_x,
                    item.yaw_axis_world_y,
                    item.yaw_axis_world_z,
                ),
            )
        )
        projection_errors.append(abs(projection - item.yaw_moment_nm))

    command_match_fraction = (
        sum(
            abs(item.command_deg - target_rudder_deg)
            <= config.command_tolerance_deg
            for item in samples
        )
        / len(samples)
        if samples
        else 0.0
    )
    reference_heading_deg = (
        float(sign_reference_heading_deg)
        if sign_reference_heading_deg is not None
        and math.isfinite(float(sign_reference_heading_deg))
        else None
    )
    reference_sim_time_s = (
        float(sign_reference_sim_time_s)
        if sign_reference_sim_time_s is not None
        and math.isfinite(float(sign_reference_sim_time_s))
        else None
    )
    heading_aligned = (
        _align_heading_deviations(
            samples, heading_samples, reference_heading_deg
        )
        if reference_heading_deg is not None
        else []
    )
    sign_heading_valid = [
        (item, deviation_deg, alignment_delta_s)
        for item, deviation_deg, alignment_delta_s in heading_aligned
        if reference_sim_time_s is not None
        and item.sim_time_s >= reference_sim_time_s
        and deviation_deg <= config.maximum_sign_heading_deviation_deg
    ]
    sign_active = [
        (item, deviation_deg, alignment_delta_s)
        for item, deviation_deg, alignment_delta_s in sign_heading_valid
        if abs(item.yaw_moment_nm) >= config.minimum_abs_yaw_moment_nm
    ]
    sign_sample_count_sufficient = (
        len(sign_active) >= config.minimum_foil_samples
    )
    expected_sign_fraction = (
        sum(
            item.yaw_moment_nm * config.expected_yaw_moment_sign > 0.0
            for item, _, _ in sign_active
        )
        / len(sign_active)
        if sign_active
        else 0.0
    )
    mean_speed = fmean(relative_speed) if relative_speed else None
    mean_force = fmean(force_magnitude) if force_magnitude else None
    mean_abs_yaw = (
        fmean(abs(item.yaw_moment_nm) for item in samples) if samples else None
    )
    maximum_math_error = max(wrench_errors + projection_errors, default=None)
    gates = {
        "minimum_foil_samples": len(samples) >= config.minimum_foil_samples,
        "relative_water_speed": (
            mean_speed is not None
            and abs(mean_speed - target_speed)
            <= config.relative_speed_tolerance_mps
        ),
        "relative_water_direction": (
            bool(alignments)
            and fmean(alignments) >= config.minimum_flow_alignment_cosine
        ),
        "rudder_command_delivered": (
            command_match_fraction >= config.minimum_command_match_fraction
        ),
        "nonzero_hydrodynamic_force": (
            mean_force is not None
            and mean_force >= config.minimum_force_magnitude_n
        ),
        "nonzero_yaw_moment": (
            mean_abs_yaw is not None
            and mean_abs_yaw >= config.minimum_abs_yaw_moment_nm
        ),
        "minimum_sign_evaluation_samples": sign_sample_count_sufficient,
        "expected_yaw_moment_sign": (
            sign_sample_count_sufficient
            and expected_sign_fraction >= config.minimum_expected_sign_fraction
        ),
        "wrench_math_consistent": (
            maximum_math_error is not None
            and maximum_math_error <= config.maximum_wrench_error_nm
        ),
    }
    return {
        "sample_count": len(samples),
        "target_rudder_deg": float(target_rudder_deg),
        "configured_water_velocity_world_enu_mps": list(target_flow),
        "configured_water_speed_mps": target_speed,
        "relative_water_speed_mps": _stats(relative_speed),
        "relative_water_x_mps": _stats(
            [item.water_relative_x_mps for item in samples]
        ),
        "relative_water_y_mps": _stats(
            [item.water_relative_y_mps for item in samples]
        ),
        "relative_water_z_mps": _stats(
            [item.water_relative_z_mps for item in samples]
        ),
        "flow_alignment_cosine": _stats(alignments),
        "alpha_deg": _stats([item.alpha_deg for item in samples]),
        "force_magnitude_n": _stats(force_magnitude),
        "force_world_x_n": _stats([item.force_world_x_n for item in samples]),
        "force_world_y_n": _stats([item.force_world_y_n for item in samples]),
        "force_world_z_n": _stats([item.force_world_z_n for item in samples]),
        "yaw_moment_nm": _stats([item.yaw_moment_nm for item in samples]),
        "command_match_fraction": command_match_fraction,
        "expected_yaw_moment_sign": config.expected_yaw_moment_sign,
        "expected_yaw_moment_sign_fraction": expected_sign_fraction,
        "sign_reference_heading_deg": reference_heading_deg,
        "sign_reference_capture_sim_time_s": reference_sim_time_s,
        "sign_reference_source": "pre_step_odometry",
        "sign_evaluation_maximum_heading_deviation_deg": (
            config.maximum_sign_heading_deviation_deg
        ),
        "sign_evaluation_required_sample_count": config.minimum_foil_samples,
        "sign_evaluation_heading_valid_sample_count": len(sign_heading_valid),
        "sign_evaluation_sample_count": len(sign_active),
        "sign_evaluation_heading_deviation_deg": _stats(
            [deviation for _, deviation, _ in sign_active]
        ),
        "sign_evaluation_all_matched_heading_deviation_deg": _stats(
            [deviation for _, deviation, _ in heading_aligned]
        ),
        "sign_evaluation_heading_alignment_delta_s": _stats(
            [delta for _, _, delta in sign_active]
        ),
        "maximum_wrench_error_nm": maximum_math_error,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "passed": all(gates.values()),
    }


def summarize_loaded_tracking(tracking: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate existing actuator gates without the unloaded-speed contract."""

    steps = list(tracking.get("steps", ()))
    step_results = []
    for step in steps:
        source_gates = dict(step.get("gates", {}))
        gates = {
            name: bool(source_gates.get(name, False))
            for name in LOADED_TRACKING_GATES
        }
        step_results.append(
            {
                "target_rudder_deg": step.get("target_rudder_deg"),
                "gates": gates,
                "failed_gates": [
                    name for name, passed in gates.items() if not passed
                ],
                "passed": all(gates.values()),
            }
        )
    return {
        "excluded_unloaded_only_gate": "unloaded_speed",
        "steps": step_results,
        "passed": bool(step_results)
        and all(step["passed"] for step in step_results),
    }


def wrench_sample_rows(
    samples: Sequence[RudderWrenchSample],
) -> list[dict[str, float]]:
    return [asdict(sample) for sample in samples]
