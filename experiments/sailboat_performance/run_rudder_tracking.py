"""Run fixed-rudder actuator-isolation diagnostics.

This runner reuses the validated launch, pre-arm hold, and command adapter.  It
does not enter AUTO or GUIDED, never enables sail force, and does not create a
second rudder command publisher.  The adapter's existing diagnostic override
is the sole owner of the final rudder setpoint during the measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.sailboat_physics.rudder_response import parse_rudder_wrench

from .guided_heading import (
    parse_ardupilot_parameter_file,
    validate_wind_only_sdf,
    validate_wind_only_servo_functions,
)
from .rudder_tracking import (
    PreMeasurementSettlingState,
    RudderTrackingConfig,
    pre_measurement_settle_timed_out,
    summarize_tracking_run,
    validate_setpoints_deg,
)
from .rudder_loaded import (
    ControlledRudderLoadConfig,
    parse_diagnostic_flow_activation_sim_time,
    render_rudder_loaded_model,
    summarize_loaded_rudder_wrench,
    summarize_loaded_tracking,
    wrench_sample_rows,
)
from .run_polar_trial import (
    arm_vehicle,
    ensure_residual_disabled,
    execute_shell,
    git_head,
    make_bo_compatible_config,
    resolve_path,
)
from .telemetry import PerformanceRosCollector, telemetry_ages_are_fresh


DEFAULT_CONFIG = Path(
    "experiments/sailboat_performance/config/rudder_unloaded.yaml"
)
DEFAULT_PARAMS = Path("experiments/sailboat_BO/optimized_baseline_params.json")
DEFAULT_RESULTS = Path("experiments/sailboat_performance/results")

CALM_WORLD_TEMPLATE = "rudder_calm"
CALM_WAVE_MODEL_URI = "model://waves_low"
ZERO_TOLERANCE = 1.0e-12

RAW_FIELDS = [
    "trial_id",
    "step_index",
    "sample_time_utc",
    "sim_time_s",
    "elapsed_step_s",
    "target_rudder_rad",
    "target_rudder_deg",
    "base_rudder_rad",
    "final_rudder_rad",
    "actual_rudder_angle_rad",
    "actual_rudder_angle_deg",
    "actual_rudder_velocity_rad_s",
    "command_target_error_rad",
    "tracking_error_rad",
    "target_tracking_error_rad",
    "speed_over_ground_mps",
    "x_east_m",
    "y_north_m",
    "z_up_m",
    "roll_frd_deg",
    "pitch_frd_deg",
    "heading_deg",
    "odom_age_s",
    "rudder_age_s",
    "actuator_age_s",
    "telemetry_valid",
]

YAW_ORIGIN_FIELDS = [
    "sample_time_utc",
    "sim_time_s",
    "phase",
    "startup_hold_requested_held",
    "diagnostic_flow_requested_enabled",
    "heading_deg",
    "derived_yaw_rate_deg_s",
    "gazebo_angular_yaw_rate_deg_s",
    "x_east_m",
    "y_north_m",
    "speed_over_ground_mps",
    "actual_rudder_angle_rad",
    "actual_rudder_angle_deg",
    "odom_age_s",
    "rudder_age_s",
]


class PreMeasurementSettleTimeout(TimeoutError):
    """Raised when the unloaded boat never reaches the measurement gate."""


def make_yaw_origin_record(
    snapshot: Any,
    *,
    phase: str,
    startup_hold_requested_held: bool,
    diagnostic_flow_requested_enabled: bool,
) -> dict[str, Any]:
    """Serialize read-only telemetry used to locate pre-measurement yaw onset."""

    rudder_angle = snapshot.actual_rudder_angle_rad
    return {
        "sample_time_utc": datetime.now(timezone.utc).isoformat(),
        "sim_time_s": snapshot.sim_time_s,
        "phase": str(phase),
        "startup_hold_requested_held": bool(startup_hold_requested_held),
        "diagnostic_flow_requested_enabled": bool(
            diagnostic_flow_requested_enabled
        ),
        "heading_deg": snapshot.heading_deg,
        "derived_yaw_rate_deg_s": snapshot.derived_yaw_rate_deg_s,
        "gazebo_angular_yaw_rate_deg_s": (
            snapshot.gazebo_angular_yaw_rate_deg_s
        ),
        "x_east_m": snapshot.x_east_m,
        "y_north_m": snapshot.y_north_m,
        "speed_over_ground_mps": snapshot.speed_over_ground_mps,
        "actual_rudder_angle_rad": rudder_angle,
        "actual_rudder_angle_deg": (
            math.degrees(float(rudder_angle))
            if rudder_angle is not None
            else None
        ),
        "odom_age_s": snapshot.odom_age_s,
        "rudder_age_s": snapshot.rudder_age_s,
    }


class YawOriginRecorder:
    """Continuously sample the existing collector while lifecycle calls block."""

    def __init__(self, collector: Any, sample_hz: float):
        frequency = float(sample_hz)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("yaw-origin sample_hz must be finite and positive")
        self._collector = collector
        self._sample_period_s = 1.0 / frequency
        self._phase = "not_started"
        self._startup_hold_requested_held = True
        self._diagnostic_flow_requested_enabled = False
        self._records: list[dict[str, Any]] = []
        self._last_sim_time_s: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_phase(
        self,
        phase: str,
        *,
        startup_hold_requested_held: bool,
        diagnostic_flow_requested_enabled: bool,
    ) -> None:
        with self._lock:
            self._phase = str(phase)
            self._startup_hold_requested_held = bool(
                startup_hold_requested_held
            )
            self._diagnostic_flow_requested_enabled = bool(
                diagnostic_flow_requested_enabled
            )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("yaw-origin recorder already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            return [dict(record) for record in self._records]

    def _run(self) -> None:
        while not self._stop_event.wait(0.01):
            snapshot = self._collector.snapshot()
            if snapshot.sim_time_s is None:
                continue
            sim_time_s = float(snapshot.sim_time_s)
            if not math.isfinite(sim_time_s):
                continue
            with self._lock:
                if (
                    self._last_sim_time_s is not None
                    and sim_time_s - self._last_sim_time_s
                    < 0.95 * self._sample_period_s
                ):
                    continue
                self._last_sim_time_s = sim_time_s
                self._records.append(
                    make_yaw_origin_record(
                        snapshot,
                        phase=self._phase,
                        startup_hold_requested_held=(
                            self._startup_hold_requested_held
                        ),
                        diagnostic_flow_requested_enabled=(
                            self._diagnostic_flow_requested_enabled
                        ),
                    )
                )


def summarize_yaw_origin_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize coverage only; yaw onset remains an evidence-based review."""

    items = list(records)
    phase_sample_counts: dict[str, int] = {}
    for item in items:
        phase = str(item.get("phase", ""))
        phase_sample_counts[phase] = phase_sample_counts.get(phase, 0) + 1
    sim_times = [
        float(item["sim_time_s"])
        for item in items
        if item.get("sim_time_s") is not None
        and math.isfinite(float(item["sim_time_s"]))
    ]
    return {
        "sample_count": len(items),
        "phase_sample_counts": phase_sample_counts,
        "first_sim_time_s": min(sim_times) if sim_times else None,
        "last_sim_time_s": max(sim_times) if sim_times else None,
        "automatic_yaw_onset_classification": False,
    }


def pre_measurement_settle_timeout_outcome(
    exc: BaseException,
) -> dict[str, str]:
    """Build the explicit runtime result for a settling timeout."""

    return {
        "status": "failure",
        "failure_reason": "pre_measurement_settle_timeout",
        "error": f"{type(exc).__name__}: {exc}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure fixed-rudder joint tracking with optional load"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--params-file", default=str(DEFAULT_PARAMS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--repeat-idx", type=int, default=0)
    parser.add_argument(
        "--execution-backend", choices=["local", "docker-exec"], default=None
    )
    parser.add_argument("--docker-container")
    parser.add_argument("--container-repo-root")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return non-zero when any diagnostic gate fails",
    )
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> RudderTrackingConfig:
    missing = sorted({"study", "world", "spawn", "diagnostic"} - set(config))
    if missing:
        raise ValueError("missing rudder diagnostic sections: " + ", ".join(missing))
    diagnostic = config["diagnostic"]
    validate_setpoints_deg(diagnostic.get("setpoints_deg", ()))
    if config["world"].get("template") != CALM_WORLD_TEMPLATE:
        raise ValueError(
            "strict unloaded rudder tracking requires the diagnostic-only "
            f"{CALM_WORLD_TEMPLATE!r} world template"
        )
    load = config.get("load")
    if load is not None:
        if load.get("type") != (
            "rudder_foil_runtime_gated_world_water_velocity"
        ):
            raise ValueError("unsupported diagnostic rudder load type")
        ControlledRudderLoadConfig.from_mapping(load)
        if tuple(float(value) for value in diagnostic["setpoints_deg"]) != (5.0,):
            raise ValueError("the first loaded diagnostic must use only +5 deg")
    return RudderTrackingConfig.from_mapping(diagnostic)


def _parse_vector(text: str | None, *, field: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in str(text or "").split())
    except ValueError as exc:
        raise ValueError(f"{field} must contain finite numeric values") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{field} must contain finite numeric values")
    return values


def _require_zero(values: Iterable[float], *, field: str) -> None:
    if any(abs(float(value)) > ZERO_TOLERANCE for value in values):
        raise ValueError(f"strict unloaded contract requires zero {field}")


def validate_calm_environment(
    repo_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Verify that the diagnostic world removes wind, waves and current."""

    world_cfg = config["world"]
    template = str(world_cfg.get("template", ""))
    if template != CALM_WORLD_TEMPLATE:
        raise ValueError(
            "strict unloaded rudder tracking cannot use a production world: "
            f"expected {CALM_WORLD_TEMPLATE!r}, got {template!r}"
        )
    configured_forcing = {
        "configured wind magnitude noise": "wind_mag_noise_stddev",
        "configured wind direction noise": "wind_dir_noise_stddev",
        "configured wind magnitude sine amplitude": (
            "wind_mag_sin_amp_percent"
        ),
        "configured wind direction sine amplitude": "wind_dir_sin_amp_deg",
    }
    for label, key in configured_forcing.items():
        _require_zero(
            _parse_vector(str(world_cfg.get(key, "")), field=label),
            field=label,
        )

    world_path = (
        repo_root / "models/dave_worlds/worlds" / f"{template}.world"
    )
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"diagnostic world has no <world>: {world_path}")

    expected_world_name = str(world_cfg.get("internal_world_name", "waves"))
    if world.get("name") != expected_world_name:
        raise ValueError(
            "diagnostic world name does not match internal_world_name: "
            f"{world.get('name')!r} != {expected_world_name!r}"
        )

    wind = world.find("wind/linear_velocity")
    _require_zero(
        _parse_vector(wind.text if wind is not None else None, field="world wind"),
        field="world wind",
    )

    wind_effects = world.find(
        "plugin[@filename='gz-sim-wind-effects-system']"
    )
    if wind_effects is None:
        raise ValueError(
            "diagnostic world must retain the active WindEffects structure "
            "required by render_trial_world"
        )
    stochastic_paths = {
        "wind magnitude sine amplitude": (
            "horizontal/magnitude/sin/amplitude_percent"
        ),
        "wind magnitude noise mean": "horizontal/magnitude/noise/mean",
        "wind magnitude noise standard deviation": (
            "horizontal/magnitude/noise/stddev"
        ),
        "wind direction sine amplitude": "horizontal/direction/sin/amplitude",
        "wind direction noise mean": "horizontal/direction/noise/mean",
        "wind direction noise standard deviation": (
            "horizontal/direction/noise/stddev"
        ),
        "vertical wind noise mean": "vertical/noise/mean",
        "vertical wind noise standard deviation": "vertical/noise/stddev",
    }
    for label, path in stochastic_paths.items():
        node = wind_effects.find(path)
        _require_zero(
            _parse_vector(node.text if node is not None else None, field=label),
            field=label,
        )

    includes = [
        str(node.text or "").strip() for node in world.findall("include/uri")
    ]
    if CALM_WAVE_MODEL_URI not in includes:
        raise ValueError(
            f"diagnostic world must include {CALM_WAVE_MODEL_URI!r}"
        )

    current_plugins = []
    for plugin in world.findall("plugin"):
        identity = " ".join(
            (plugin.get("filename", ""), plugin.get("name", ""))
        ).lower()
        if "ocean-current" in identity or "oceancurrent" in identity:
            current_plugins.append(identity)
    if current_plugins:
        raise ValueError(
            "strict unloaded world must not load an ocean-current plugin"
        )

    wave_model_path = (
        repo_root / "models/dave_world_models/waves_low/model.sdf"
    )
    wave_root = ET.parse(wave_model_path).getroot()
    wave_nodes = list(wave_root.iter("wave"))
    if not wave_nodes:
        raise ValueError(f"calm wave model has no <wave>: {wave_model_path}")
    for index, wave in enumerate(wave_nodes):
        if str(wave.findtext("algorithm", "")).strip().lower() != "fft":
            raise ValueError(f"calm wave[{index}] must use the FFT model")
        _require_zero(
            _parse_vector(wave.findtext("wind_speed"), field="wave wind_speed"),
            field="wave wind_speed",
        )
        _require_zero(
            _parse_vector(wave.findtext("steepness"), field="wave steepness"),
            field="wave steepness",
        )

    sailboat_sdf = (
        repo_root / "models/dave_robot_models/description/sailboat/model.sdf"
    )
    # The file contains Gazebo extension tags without a namespace declaration,
    # so inspect only the explicit FoilLiftDrag water-velocity elements here.
    water_velocity_text = [
        match.group(1)
        for match in re.finditer(
            r"<water_velocity>([^<]+)</water_velocity>",
            sailboat_sdf.read_text(encoding="utf-8"),
        )
    ]
    if not water_velocity_text:
        raise ValueError("sailboat FoilLiftDrag water_velocity was not found")
    for index, text in enumerate(water_velocity_text):
        _require_zero(
            _parse_vector(text, field=f"foil water_velocity[{index}]"),
            field=f"foil water_velocity[{index}]",
        )

    return {
        "world_template": template,
        "world_path": str(world_path),
        "wave_model_uri": CALM_WAVE_MODEL_URI,
        "wave_model_path": str(wave_model_path),
        "wind_world_enu_mps": [0.0, 0.0, 0.0],
        "wave_forcing_enabled": False,
        "ocean_current_enu_mps": [0.0, 0.0, 0.0],
        "foil_water_velocity_mps": [0.0, 0.0, 0.0],
    }


def make_scenario(config: dict[str, Any]) -> dict[str, Any]:
    world = dict(config["world"])
    # The dedicated calm template removes wave excitation; the scenario also
    # enforces zero air wind. Sail force stays disabled for the entire run.
    world["wind_world_xyz_mps"] = [0.0, 0.0, 0.0]
    load = config.get("load")
    scenario_id = "rudder_unloaded_tracking"
    if load is not None:
        target_speed = ControlledRudderLoadConfig.from_mapping(load).target_speed_mps
        speed_tag = f"{target_speed:g}".replace(".", "p")
        scenario_id = f"rudder_loaded_tracking_{speed_tag}mps"
    return {
        "id": scenario_id,
        "split": "diagnostic",
        "world": world,
        "spawn": dict(config["spawn"]),
    }


def write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_yaw_origin_csv(
    path: Path, records: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=YAW_ORIGIN_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def append_local_robot_description_file(
    launch_plan: Any, model_path: Path
) -> Any:
    """Append a trial-only model path without changing default launch behavior."""

    if launch_plan.backend != "local":
        raise ValueError(
            "loaded rudder diagnostic currently requires --execution-backend local"
        )
    argument = f"robot_description_file:={model_path}"
    process_cmd = list(launch_plan.process_cmd)
    if len(process_cmd) < 3 or process_cmd[-2] != "-lc":
        raise ValueError("unexpected local launch command structure")
    process_cmd[-1] = f"{process_cmd[-1]} {shlex.quote(argument)}"
    return replace(
        launch_plan,
        launch_cmd=[*launch_plan.launch_cmd, argument],
        process_cmd=process_cmd,
        display_cmd=f"{launch_plan.display_cmd} {shlex.quote(argument)}",
    )


def serialize_ros_parameter_value(value: float | bool) -> str:
    """Serialize scalar values without losing their ROS parameter type."""

    # bool is an int subclass in Python, so it must be handled before numeric
    # conversion.  repr(float(...)) retains a decimal point for integer-valued
    # floats (for example, 0.0 and -5.0), preventing ros2 param from inferring
    # an INTEGER for a parameter declared as DOUBLE.
    if isinstance(value, bool):
        return "true" if value else "false"
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("ROS double parameter value must be finite")
    return repr(numeric)


def _parameter_command(
    execution: Any,
    node_name: str,
    parameter_name: str,
    value: float | bool,
) -> None:
    value_text = serialize_ros_parameter_value(value)
    result = execute_shell(
        execution,
        "ros2 param set "
        + " ".join(
            shlex.quote(part)
            for part in (node_name, parameter_name, value_text)
        ),
    )
    if result.returncode != 0 or "successful" not in result.stdout.lower():
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"failed to set {parameter_name}={value} on {node_name}: {detail}"
        )


def set_rudder_override(
    execution: Any,
    node_name: str,
    *,
    command_rad: float,
    enabled: bool | None = None,
) -> None:
    """Update the existing adapter override without adding a topic publisher."""

    _parameter_command(
        execution,
        node_name,
        "diagnostic_rudder_command_rad",
        float(command_rad),
    )
    if enabled is not None:
        _parameter_command(
            execution,
            node_name,
            "diagnostic_rudder_override_enabled",
            enabled,
        )


def publish_rudder_test_flow_enabled(
    execution: Any,
    topic: str,
    enabled: bool,
    *,
    repeats: int = 1,
    interval_s: float = 0.0,
) -> None:
    """Toggle only the trial-local rudder diagnostic flow."""

    if not str(topic).startswith("/"):
        raise ValueError("diagnostic flow enable topic must be absolute")
    command = " ".join(
        shlex.quote(part)
        for part in (
            "gz",
            "topic",
            "-t",
            str(topic),
            "-m",
            "gz.msgs.Boolean",
            "-p",
            f"data: {'true' if enabled else 'false'}",
        )
    )
    count = max(1, int(repeats))
    delay_s = max(0.0, float(interval_s))
    for index in range(count):
        result = execute_shell(execution, command)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                "failed to publish rudder diagnostic flow command: "
                f"enabled={enabled}, topic={topic}, error={detail}"
            )
        if index < count - 1 and delay_s > 0.0:
            time.sleep(delay_s)
    print(
        "[rudder_tracking] diagnostic flow command: "
        f"enabled={str(enabled).lower()}, topic={topic}, repeats={count}"
    )


def make_record(
    *,
    trial_id: str,
    step_index: int,
    target_rad: float,
    start_sim_time_s: float,
    snapshot: Any,
    maximum_source_age_s: float,
) -> dict[str, Any]:
    required = (
        snapshot.sim_time_s,
        snapshot.actual_rudder_angle_rad,
        snapshot.actual_rudder_velocity_rad_s,
        snapshot.base_rudder_rad,
        snapshot.final_rudder_rad,
        snapshot.speed_over_ground_mps,
    )
    telemetry_valid = (
        all(value is not None and math.isfinite(float(value)) for value in required)
        and telemetry_ages_are_fresh(
            (
                snapshot.odom_age_s,
                snapshot.rudder_age_s,
                snapshot.actuator_age_s,
            ),
            maximum_source_age_s,
        )
    )
    actual = snapshot.actual_rudder_angle_rad
    final = snapshot.final_rudder_rad
    return {
        "trial_id": trial_id,
        "step_index": int(step_index),
        "sample_time_utc": datetime.now(timezone.utc).isoformat(),
        "sim_time_s": snapshot.sim_time_s,
        "elapsed_step_s": (
            float(snapshot.sim_time_s) - float(start_sim_time_s)
            if snapshot.sim_time_s is not None
            else None
        ),
        "target_rudder_rad": target_rad,
        "target_rudder_deg": math.degrees(target_rad),
        "base_rudder_rad": snapshot.base_rudder_rad,
        "final_rudder_rad": final,
        "actual_rudder_angle_rad": actual,
        "actual_rudder_angle_deg": (
            math.degrees(float(actual)) if actual is not None else None
        ),
        "actual_rudder_velocity_rad_s": snapshot.actual_rudder_velocity_rad_s,
        "command_target_error_rad": (
            float(final) - target_rad if final is not None else None
        ),
        "tracking_error_rad": (
            float(actual) - float(final)
            if actual is not None and final is not None
            else None
        ),
        "target_tracking_error_rad": (
            float(actual) - target_rad if actual is not None else None
        ),
        "speed_over_ground_mps": snapshot.speed_over_ground_mps,
        "x_east_m": snapshot.x_east_m,
        "y_north_m": snapshot.y_north_m,
        "z_up_m": snapshot.z_up_m,
        "roll_frd_deg": snapshot.roll_frd_deg,
        "pitch_frd_deg": snapshot.pitch_frd_deg,
        "heading_deg": snapshot.heading_deg,
        "odom_age_s": snapshot.odom_age_s,
        "rudder_age_s": snapshot.rudder_age_s,
        "actuator_age_s": snapshot.actuator_age_s,
        "telemetry_valid": telemetry_valid,
    }


def main() -> int:
    args = parse_args()

    # Runtime-only imports keep unit tests independent of ROS, Gazebo and
    # pymavlink availability.
    from experiments.sailboat_BO.common import load_yaml
    from experiments.sailboat_BO.run_trial import (
        ExecutionConfig,
        attempt_arm_until_armed,
        build_context,
        build_launch_plan,
        connect_mavlink,
        get_repo_root,
        get_workspace_root,
        is_tcp_endpoint_open,
        load_param_overrides_from_file,
        publish_sail_force_enabled,
        publish_startup_state_held,
        render_trial_world,
        start_launch_process,
        stop_launch_process,
        validate_execution_backend,
        wait_for_endpoint_closed,
        write_trial_manifest,
        write_trial_param_file,
    )

    repo_root = get_repo_root()
    workspace_root = get_workspace_root(repo_root)
    config_path = resolve_path(repo_root, args.config)
    params_path = resolve_path(repo_root, args.params_file)
    config = load_yaml(config_path)
    diagnostic_config = validate_config(config)
    loaded_config = (
        ControlledRudderLoadConfig.from_mapping(config["load"])
        if config.get("load") is not None
        else None
    )
    environment_contract = validate_calm_environment(repo_root, config)
    setpoints_deg = validate_setpoints_deg(config["diagnostic"]["setpoints_deg"])
    results_dir = resolve_path(repo_root, args.results_dir)

    missing_commands = [
        name for name in ("gz", "ros2", "ardurover") if shutil.which(name) is None
    ]
    if missing_commands:
        raise SystemExit(
            "rudder tracking must run inside the simulation container; missing: "
            + ", ".join(missing_commands)
        )

    execution = ExecutionConfig(
        backend=args.execution_backend
        or str(config["study"].get("execution_backend", "local")),
        docker_container=args.docker_container
        or config["study"].get("docker_container"),
        container_repo_root=args.container_repo_root
        or config["study"].get("container_repo_root"),
    )
    validate_execution_backend(execution)
    if loaded_config is not None and execution.backend != "local":
        raise SystemExit(
            "loaded rudder diagnostic requires --execution-backend local"
        )
    params = load_param_overrides_from_file(params_path)
    base_param_path = resolve_path(repo_root, config["study"]["base_param_file"])
    base_values = parse_ardupilot_parameter_file(base_param_path)
    base_values.update(params)
    validate_wind_only_servo_functions(base_values)
    validate_wind_only_sdf(
        repo_root / "models/dave_robot_models/description/sailboat/model.sdf"
    )

    scenario = make_scenario(config)
    context = build_context(
        make_bo_compatible_config(config),
        scenario,
        params,
        repo_root,
        workspace_root,
        results_dir,
        int(args.repeat_idx),
    )
    write_trial_manifest(context)
    write_trial_param_file(base_param_path, params, context.files.param_file)
    trial_world_name = render_trial_world(context)
    loaded_model_contract: dict[str, Any] | None = None
    loaded_model_path: Path | None = None
    if loaded_config is not None:
        loaded_model_path = context.files.trial_dir / "rudder_loaded_model.sdf"
        loaded_model_contract = render_rudder_loaded_model(
            repo_root
            / "models/dave_robot_models/description/sailboat/model.sdf",
            loaded_model_path,
            loaded_config.water_velocity_world_enu_mps,
            loaded_config.wrench_sample_period_s,
            loaded_config.diagnostic_flow_enable_topic,
        )
    launch_plan = build_launch_plan(
        context, execution, trial_world_name, prepare_remote_inputs=True
    )
    if loaded_model_path is not None:
        launch_plan = append_local_robot_description_file(
            launch_plan, loaded_model_path
        )
    manifest = json.loads(context.files.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": (
                "loaded_fixed_rudder_tracking"
                if loaded_config is not None
                else "unloaded_fixed_rudder_tracking"
            ),
            "diagnostic_config": str(config_path),
            "rudder_command_owner": "sailboat_command_adapter diagnostic override",
            "waypoint_mission": False,
            "guided_control": False,
            "residual_rl": False,
            "sail_force_enabled": False,
            "configured_wind_world_enu_mps": [0.0, 0.0, 0.0],
            "strict_unloaded_environment": environment_contract,
            "diagnostic_rudder_load": loaded_model_contract,
            "git_commit": git_head(repo_root),
        }
    )
    context.files.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    endpoint = str(context.study_cfg["mavlink_endpoint"])
    if is_tcp_endpoint_open(endpoint):
        raise SystemExit(f"MAVLink endpoint already in use: {endpoint}")

    process = None
    master = None
    collector: PerformanceRosCollector | None = None
    yaw_origin_recorder: YawOriginRecorder | None = None
    yaw_origin_records: list[dict[str, Any]] = []
    yaw_origin_samples_path = (
        context.files.samples_csv.parent / "yaw_origin_samples.csv"
    )
    override_enabled = False
    diagnostic_flow_enabled = False
    diagnostic_flow_enable_requested_sim_time_s: float | None = None
    diagnostic_flow_enable_requested_at_utc: str | None = None
    diagnostic_flow_enabled_sim_time_s: float | None = None
    records: list[dict[str, Any]] = []
    starts_by_step: dict[int, float] = {}
    targets_by_step: dict[int, float] = {}
    sign_references_by_step: dict[int, dict[str, float]] = {}
    outcome: dict[str, Any] = {
        "status": "failure",
        "failure_reason": "not_started",
    }
    endpoint_closed = False
    settling_state = PreMeasurementSettlingState(
        required_speed_mps=diagnostic_config.maximum_unloaded_speed_mps,
        required_consecutive_s=(
            diagnostic_config.pre_measurement_settle_required_s
        ),
    )
    settling_started_wall_s: float | None = None
    settling_finished_wall_s: float | None = None
    loaded_wrench_samples_path = (
        context.files.summary_dir / "loaded_rudder_wrench_samples.csv"
    )

    try:
        load_description = (
            "rudder_only_water_velocity=zero_until_measurement, target="
            + str(list(loaded_config.water_velocity_world_enu_mps))
            if loaded_config is not None
            else "rudder_load=none"
        )
        print(
            "[rudder_tracking] calm isolation environment: "
            f"world={environment_contract['world_template']}, "
            f"wave={environment_contract['wave_model_uri']}, "
            f"wind=zero, ocean_current=zero, {load_description}"
        )
        print(f"[rudder_tracking] trial_id: {context.trial_id}")
        print(f"[rudder_tracking] launch: {launch_plan.display_cmd}")
        process = start_launch_process(context, launch_plan)
        publish_startup_state_held(
            context, execution, True, reason="rudder_tracking_prearm"
        )
        publish_sail_force_enabled(
            context, execution, False, reason="rudder_tracking_isolation"
        )
        collector = PerformanceRosCollector(
            str(context.study_cfg["namespace"]),
            str(scenario["world"].get("internal_world_name", "waves")),
        )
        collector.start()
        yaw_origin_recorder = YawOriginRecorder(
            collector, diagnostic_config.sample_hz
        )
        yaw_origin_recorder.set_phase(
            "startup_hold_prearm",
            startup_hold_requested_held=True,
            diagnostic_flow_requested_enabled=False,
        )
        yaw_origin_recorder.start()
        master = connect_mavlink(
            endpoint,
            timeout_s=float(
                context.study_cfg.get("mavlink_connect_timeout_s", 120.0)
            ),
            launch_process=process,
            stdout_log=context.files.logs_dir / "launch.stdout.log",
            stderr_log=context.files.logs_dir / "launch.stderr.log",
        )
        time.sleep(3.0)
        yaw_origin_recorder.set_phase(
            "arming_hold",
            startup_hold_requested_held=True,
            diagnostic_flow_requested_enabled=False,
        )
        arm_vehicle(master, context, attempt_arm_until_armed)
        yaw_origin_recorder.set_phase(
            "armed_hold",
            startup_hold_requested_held=True,
            diagnostic_flow_requested_enabled=False,
        )

        adapter_node = str(
            config["diagnostic"].get(
                "command_adapter_node", "/sailboat_command_adapter"
            )
        )
        ensure_residual_disabled(execution, adapter_node)
        set_rudder_override(
            execution, adapter_node, command_rad=0.0, enabled=True
        )
        override_enabled = True
        collector.wait_ready(
            float(config["diagnostic"]["telemetry_startup_timeout_s"]),
            diagnostic_config.maximum_source_age_s,
        )

        # Holding all model links at zero velocity also holds the rudder joint.
        # Release it before any response measurement; sail force stays off.
        yaw_origin_recorder.set_phase(
            "startup_hold_release_command",
            startup_hold_requested_held=False,
            diagnostic_flow_requested_enabled=False,
        )
        publish_startup_state_held(
            context, execution, False, reason="rudder_tracking_measurement"
        )
        release_settle_s = float(
            context.study_cfg.get("startup_state_release_settle_s", 0.2)
        )
        if release_settle_s > 0.0:
            time.sleep(release_settle_s)

        sample_period_s = 1.0 / diagnostic_config.sample_hz
        poll_period_s = min(0.02, 0.5 * sample_period_s)
        settling_started_wall_s = time.monotonic()
        next_settle_poll_wall_s = settling_started_wall_s
        last_settle_sim_time_s: float | None = None
        neutral_command_tolerance_rad = math.radians(0.1)
        yaw_origin_recorder.set_phase(
            "pre_measurement_settling",
            startup_hold_requested_held=False,
            diagnostic_flow_requested_enabled=False,
        )
        print(
            "[rudder_tracking] pre-measurement settling: "
            f"speed<={diagnostic_config.maximum_unloaded_speed_mps:.3f} m/s "
            "for "
            f"{diagnostic_config.pre_measurement_settle_required_s:.1f} sim s, "
            "neutral diagnostic override remains active"
        )
        while not settling_state.reached:
            now = time.monotonic()
            if process.poll() is not None:
                raise RuntimeError(
                    "simulation launch process exited during pre-measurement "
                    "settling"
                )
            if pre_measurement_settle_timed_out(
                wall_elapsed_s=now - settling_started_wall_s,
                timeout_s=diagnostic_config.pre_measurement_settle_timeout_s,
                reached=settling_state.reached,
            ):
                raise PreMeasurementSettleTimeout(
                    "boat did not remain below the unloaded speed threshold "
                    "for the required interval"
                )
            if now + 1e-9 < next_settle_poll_wall_s:
                time.sleep(min(0.01, next_settle_poll_wall_s - now))
                continue
            next_settle_poll_wall_s = now + poll_period_s
            snapshot = collector.snapshot()
            if snapshot.sim_time_s is None or not math.isfinite(
                float(snapshot.sim_time_s)
            ):
                continue
            sim_time_s = float(snapshot.sim_time_s)
            if (
                last_settle_sim_time_s is not None
                and sim_time_s - last_settle_sim_time_s
                < 0.95 * sample_period_s
            ):
                continue
            last_settle_sim_time_s = sim_time_s
            speed_valid = (
                snapshot.speed_over_ground_mps is not None
                and math.isfinite(float(snapshot.speed_over_ground_mps))
                and float(snapshot.speed_over_ground_mps) >= 0.0
            )
            final_command_valid = (
                snapshot.final_rudder_rad is not None
                and math.isfinite(float(snapshot.final_rudder_rad))
            )
            telemetry_valid = telemetry_ages_are_fresh(
                (
                    snapshot.odom_age_s,
                    snapshot.rudder_age_s,
                    snapshot.actuator_age_s,
                ),
                diagnostic_config.maximum_source_age_s,
            ) and speed_valid and final_command_valid
            neutral_rudder_owned = (
                final_command_valid
                and abs(float(snapshot.final_rudder_rad))
                <= neutral_command_tolerance_rad
            )
            settling_state.observe(
                sim_time_s=sim_time_s,
                speed_mps=(
                    float(snapshot.speed_over_ground_mps)
                    if speed_valid
                    else None
                ),
                telemetry_valid=telemetry_valid,
                neutral_rudder_owned=neutral_rudder_owned,
                x_east_m=snapshot.x_east_m,
                y_north_m=snapshot.y_north_m,
                z_up_m=snapshot.z_up_m,
                roll_frd_deg=snapshot.roll_frd_deg,
                pitch_frd_deg=snapshot.pitch_frd_deg,
                heading_deg=snapshot.heading_deg,
            )
        settling_finished_wall_s = time.monotonic()
        print(
            "[rudder_tracking] pre-measurement settling reached: "
            f"duration_sim_s={settling_state.last_sim_time_s - settling_state.first_sim_time_s:.2f}, "
            f"final_speed_mps={settling_state.final_speed_mps:.3f}"
        )
        yaw_origin_recorder.set_phase(
            "pre_flow_reference",
            startup_hold_requested_held=False,
            diagnostic_flow_requested_enabled=False,
        )

        # The wall-time limit below applies only to the fixed-setpoint
        # measurement.  Settling has its own explicit timeout above.
        run_started_wall_s = time.monotonic()
        maximum_wall_duration_s = float(
            config["diagnostic"].get(
                "maximum_wall_duration_s",
                3.0 * diagnostic_config.setpoint_hold_s * len(setpoints_deg)
                + 120.0,
            )
        )
        last_sample_sim_time: float | None = None

        for step_index, target_deg in enumerate(setpoints_deg):
            before = collector.wait_ready(
                float(config["diagnostic"]["telemetry_startup_timeout_s"]),
                diagnostic_config.maximum_source_age_s,
            )
            if before.sim_time_s is None or before.actual_rudder_angle_rad is None:
                raise RuntimeError("pre-step rudder telemetry is incomplete")
            if loaded_config is not None and before.heading_deg is None:
                raise RuntimeError("pre-step heading telemetry is incomplete")
            target_rad = math.radians(target_deg)
            starts_by_step[step_index] = float(before.actual_rudder_angle_rad)
            targets_by_step[step_index] = target_rad
            start_sim_time_s = float(before.sim_time_s)
            sign_reference_message = ""
            if loaded_config is not None:
                sign_references_by_step[step_index] = {
                    "heading_deg": float(before.heading_deg),
                    "sim_time_s": start_sim_time_s,
                }
                if step_index == 0:
                    diagnostic_flow_enable_requested_sim_time_s = (
                        start_sim_time_s
                    )
                    diagnostic_flow_enable_requested_at_utc = datetime.now(
                        timezone.utc
                    ).isoformat()
                    yaw_origin_recorder.set_phase(
                        "diagnostic_flow_enable_command",
                        startup_hold_requested_held=False,
                        diagnostic_flow_requested_enabled=True,
                    )
                    publish_rudder_test_flow_enabled(
                        execution,
                        loaded_config.diagnostic_flow_enable_topic,
                        True,
                    )
                    diagnostic_flow_enabled = True
                sign_reference_message = (
                    ", sign_reference_heading_deg="
                    f"{float(before.heading_deg):.3f}, "
                    f"sign_reference_sim_time_s={start_sim_time_s:.3f}"
                )
            set_rudder_override(
                execution, adapter_node, command_rad=target_rad
            )
            if loaded_config is not None:
                yaw_origin_recorder.set_phase(
                    "rudder_measurement",
                    startup_hold_requested_held=False,
                    diagnostic_flow_requested_enabled=True,
                )
            print(
                "[rudder_tracking] setpoint: "
                f"step={step_index}, target_deg={target_deg:.1f}, "
                f"start_actual_deg={math.degrees(starts_by_step[step_index]):.3f}"
                f"{sign_reference_message}"
            )

            next_poll_wall_s = time.monotonic()
            while True:
                now = time.monotonic()
                if process.poll() is not None:
                    raise RuntimeError(
                        "simulation launch process exited during rudder tracking"
                    )
                if now - run_started_wall_s >= maximum_wall_duration_s:
                    raise TimeoutError("rudder tracking exceeded maximum wall time")
                if now + 1e-9 < next_poll_wall_s:
                    time.sleep(min(0.01, next_poll_wall_s - now))
                    continue
                next_poll_wall_s = now + poll_period_s
                snapshot = collector.snapshot()
                if snapshot.sim_time_s is None or (
                    last_sample_sim_time is not None
                    and float(snapshot.sim_time_s) - last_sample_sim_time
                    < 0.95 * sample_period_s
                ):
                    continue
                last_sample_sim_time = float(snapshot.sim_time_s)
                record = make_record(
                    trial_id=context.trial_id,
                    step_index=step_index,
                    target_rad=target_rad,
                    start_sim_time_s=start_sim_time_s,
                    snapshot=snapshot,
                    maximum_source_age_s=diagnostic_config.maximum_source_age_s,
                )
                records.append(record)
                if (
                    record["elapsed_step_s"] is not None
                    and float(record["elapsed_step_s"])
                    >= diagnostic_config.setpoint_hold_s
                ):
                    break

        outcome = {"status": "complete", "failure_reason": ""}
    except PreMeasurementSettleTimeout as exc:
        settling_finished_wall_s = time.monotonic()
        outcome = pre_measurement_settle_timeout_outcome(exc)
        print(f"[rudder_tracking] ERROR: {outcome['error']}")
    except Exception as exc:
        outcome = {
            "status": "failure",
            "failure_reason": "runtime_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[rudder_tracking] ERROR: {outcome['error']}")
    finally:
        if settling_started_wall_s is not None and settling_finished_wall_s is None:
            settling_finished_wall_s = time.monotonic()
        if override_enabled:
            try:
                set_rudder_override(
                    execution, adapter_node, command_rad=0.0, enabled=False
                )
            except Exception as exc:
                outcome["override_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                outcome["status"] = "failure"
                outcome["failure_reason"] = "cleanup_error"
        if diagnostic_flow_enabled and loaded_config is not None:
            try:
                publish_rudder_test_flow_enabled(
                    execution,
                    loaded_config.diagnostic_flow_enable_topic,
                    False,
                )
            except Exception as exc:
                outcome["diagnostic_flow_cleanup_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                outcome["status"] = "failure"
                outcome["failure_reason"] = "cleanup_error"
        if yaw_origin_recorder is not None:
            yaw_origin_recorder.set_phase(
                "cleanup",
                startup_hold_requested_held=False,
                diagnostic_flow_requested_enabled=False,
            )
            yaw_origin_records = yaw_origin_recorder.stop()
        write_yaw_origin_csv(yaw_origin_samples_path, yaw_origin_records)
        write_csv(context.files.samples_csv, records)
        if collector is not None:
            collector.close()
        if master is not None:
            try:
                master.close()
            except Exception:
                pass
        stop_launch_process(process, execution, launch_plan)
        endpoint_closed = wait_for_endpoint_closed(
            endpoint,
            float(context.study_cfg.get("mavlink_release_timeout_s", 15.0)),
        )
        if not endpoint_closed:
            outcome["status"] = "failure"
            outcome["failure_reason"] = "cleanup_error"

    diagnostic = summarize_tracking_run(
        records,
        starts_by_step=starts_by_step,
        targets_by_step=targets_by_step,
        config=diagnostic_config,
    )
    loaded_tracking = None
    loaded_wrench_diagnostic = None
    if loaded_config is not None:
        log_text = ""
        for log_path in (
            context.files.logs_dir / "launch.stdout.log",
            context.files.logs_dir / "launch.stderr.log",
        ):
            if log_path.exists():
                log_text += log_path.read_text(encoding="utf-8", errors="replace")
        wrench_samples = parse_rudder_wrench(log_text)
        diagnostic_flow_enabled_sim_time_s = (
            parse_diagnostic_flow_activation_sim_time(log_text)
        )
        measurement_times = [
            float(record["sim_time_s"])
            for record in records
            if record.get("sim_time_s") is not None
        ]
        if measurement_times:
            first_measurement_s = min(measurement_times)
            last_measurement_s = max(measurement_times)
            if diagnostic_flow_enabled_sim_time_s is not None:
                first_measurement_s = max(
                    first_measurement_s,
                    diagnostic_flow_enabled_sim_time_s,
                )
            wrench_samples = [
                sample
                for sample in wrench_samples
                if first_measurement_s <= sample.sim_time_s <= last_measurement_s
            ]
        wrench_rows = wrench_sample_rows(wrench_samples)
        loaded_wrench_samples_path.parent.mkdir(parents=True, exist_ok=True)
        if wrench_rows:
            with loaded_wrench_samples_path.open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(wrench_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(wrench_rows)
        else:
            loaded_wrench_samples_path.write_text("", encoding="utf-8")
        loaded_tracking = summarize_loaded_tracking(diagnostic)
        sign_reference = sign_references_by_step.get(0, {})
        loaded_wrench_diagnostic = summarize_loaded_rudder_wrench(
            wrench_samples,
            heading_samples=records,
            sign_reference_heading_deg=sign_reference.get("heading_deg"),
            sign_reference_sim_time_s=sign_reference.get("sim_time_s"),
            target_rudder_deg=setpoints_deg[0],
            config=loaded_config,
        )
        loaded_wrench_diagnostic.update(
            {
                "rudder_test_flow_enable_topic": (
                    loaded_config.diagnostic_flow_enable_topic
                ),
                "rudder_test_flow_enable_requested_sim_time_s": (
                    diagnostic_flow_enable_requested_sim_time_s
                ),
                "rudder_test_flow_enable_requested_at_utc": (
                    diagnostic_flow_enable_requested_at_utc
                ),
                "rudder_test_flow_enabled_sim_time_s": (
                    diagnostic_flow_enabled_sim_time_s
                ),
            }
        )
    settling_wall_duration_s = (
        max(0.0, settling_finished_wall_s - settling_started_wall_s)
        if settling_started_wall_s is not None
        and settling_finished_wall_s is not None
        else 0.0
    )
    pre_measurement_settling = settling_state.to_summary(
        timeout_s=diagnostic_config.pre_measurement_settle_timeout_s,
        wall_duration_s=settling_wall_duration_s,
    )
    yaw_origin = summarize_yaw_origin_records(yaw_origin_records)
    yaw_origin.update(
        {
            "samples_csv": str(yaw_origin_samples_path),
            "rudder_test_flow_enable_requested_sim_time_s": (
                diagnostic_flow_enable_requested_sim_time_s
            ),
            "rudder_test_flow_enabled_sim_time_s": (
                diagnostic_flow_enabled_sim_time_s
            ),
        }
    )
    summary = {
        "metadata": {
            "trial_id": context.trial_id,
            "started_at_utc": context.started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_head(repo_root),
            "config_path": str(config_path),
            "params_file": str(params_path),
            "params": context.params,
        },
        "contract": {
            "load": (
                "rudder_foil_runtime_gated_world_water_velocity"
                if loaded_config is not None
                else "unloaded"
            ),
            "wind_world_enu_mps": [0.0, 0.0, 0.0],
            "wave_forcing_enabled": False,
            "ocean_current_enu_mps": [0.0, 0.0, 0.0],
            "foil_water_velocity_mps": (
                [0.0, 0.0, 0.0]
                if loaded_config is not None
                else [0.0, 0.0, 0.0]
            ),
            "rudder_test_flow_target_world_enu_mps": (
                list(loaded_config.water_velocity_world_enu_mps)
                if loaded_config is not None
                else None
            ),
            "rudder_test_flow_enable_topic": (
                loaded_config.diagnostic_flow_enable_topic
                if loaded_config is not None
                else None
            ),
            "loaded_foil": "rudder_link" if loaded_config is not None else None,
            "daggerboard_water_velocity_mps": [0.0, 0.0, 0.0],
            "motor_propulsion_enabled": False,
            "sail_force_enabled": False,
            "guided_or_auto_control": False,
            "residual_rl_enabled": False,
            "final_rudder_owner": (
                "sailboat_command_adapter diagnostic_rudder_override"
            ),
            "startup_hold_released_before_measurement": True,
        },
        "configuration": {
            **asdict(diagnostic_config),
            "setpoints_deg": list(setpoints_deg),
            "load": asdict(loaded_config) if loaded_config is not None else None,
        },
        "outcome": {
            **outcome,
            "cleanup_endpoint_closed": endpoint_closed,
        },
        "pre_measurement_settling": pre_measurement_settling,
        "yaw_origin": yaw_origin,
        "diagnostic": diagnostic,
        "loaded_tracking": loaded_tracking,
        "hydrodynamic_load": loaded_wrench_diagnostic,
        "loaded_wrench_samples_csv": (
            str(loaded_wrench_samples_path)
            if loaded_config is not None
            else None
        ),
        "passed": (
            outcome["status"] == "complete"
            and pre_measurement_settling["reached"]
            and (
                diagnostic["passed"]
                if loaded_config is None
                else bool(loaded_tracking and loaded_tracking["passed"])
                and bool(
                    loaded_wrench_diagnostic
                    and loaded_wrench_diagnostic["passed"]
                )
            )
        ),
    }
    context.files.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"samples_csv: {context.files.samples_csv}")
    print(f"summary_json: {context.files.summary_json}")
    print(f"runtime_status: {outcome['status']}")
    print(f"rudder_tracking_passed: {summary['passed']}")
    if loaded_config is not None:
        print(f"loaded_wrench_samples_csv: {loaded_wrench_samples_path}")

    if outcome["status"] != "complete":
        return 1
    if args.require_pass and not summary["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
