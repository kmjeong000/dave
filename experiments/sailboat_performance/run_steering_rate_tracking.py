"""Run one direct ArduPilot GUIDED steering-rate characterization trial."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import signed_twa_deg, wind_world_enu_from_condition
from .guided_heading import (
    GuidedTurnRateTarget,
    parse_ardupilot_parameter_file,
    validate_wind_only_sdf,
    validate_wind_only_servo_functions,
)
from .run_polar_trial import (
    arm_vehicle,
    ensure_residual_disabled,
    git_head,
    make_bo_compatible_config,
    make_scenario,
    resolve_path,
    validate_performance_config,
)
from .steering_rate import summarize_turn_rate_steps, symmetric_turn_rate_sequence
from .telemetry import (
    MavlinkPerformanceState,
    PerformanceRosCollector,
    drain_mavlink,
    telemetry_ages_are_fresh,
)


DEFAULT_CONFIG = Path("experiments/sailboat_performance/config/polar_smoke.yaml")
DEFAULT_PARAMS = Path("experiments/sailboat_BO/baseline_params.json")

RAW_FIELDS = [
    "trial_id",
    "phase",
    "step_index",
    "step_elapsed_s",
    "target_turn_rate_deg_s",
    "sample_time_utc",
    "sim_time_s",
    "elapsed_since_release_s",
    "heading_deg",
    "mavlink_yaw_deg",
    "derived_yaw_rate_deg_s",
    "gazebo_angular_yaw_rate_deg_s",
    "mavlink_yaw_rate_deg_s",
    "mavlink_gazebo_yaw_rate_error_deg_s",
    "speed_over_ground_mps",
    "roll_frd_deg",
    "pitch_frd_deg",
    "base_rudder_rad",
    "final_rudder_rad",
    "actual_rudder_angle_rad",
    "actual_rudder_velocity_rad_s",
    "base_sheet_allowance_rad",
    "final_sheet_allowance_rad",
    "actual_boom_angle_rad",
    "actual_boom_velocity_rad_s",
    "measured_wind_from_deg",
    "measured_wind_speed_mps",
    "measured_signed_twa_deg",
    "guided_mode_active",
    "telemetry_valid",
    "odom_age_s",
    "rudder_age_s",
    "boom_age_s",
    "actuator_age_s",
    "wind_age_s",
    "mavlink_attitude_age_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize the direct Rover GUIDED turn-rate loop"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--params-file", default=str(DEFAULT_PARAMS))
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repeat-idx", type=int, default=0)
    parser.add_argument("--rate-deg-s", type=float, default=12.0)
    parser.add_argument("--active-hold-s", type=float, default=3.0)
    parser.add_argument("--final-zero-hold-s", type=float, default=5.0)
    parser.add_argument("--warmup-min-speed-mps", type=float, default=0.5)
    parser.add_argument("--warmup-max-yaw-rate-deg-s", type=float, default=2.0)
    parser.add_argument("--warmup-hold-s", type=float, default=3.0)
    parser.add_argument("--warmup-timeout-s", type=float, default=60.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--target-refresh-hz", type=float, default=5.0)
    parser.add_argument(
        "--execution-backend", choices=["local", "docker-exec"], default=None
    )
    parser.add_argument("--docker-container")
    parser.add_argument("--container-repo-root")
    return parser.parse_args()


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    if args.sample_hz <= 0.0 or args.target_refresh_hz <= 0.0:
        raise SystemExit("sample and target refresh rates must be positive")
    if (
        args.warmup_min_speed_mps <= 0.0
        or args.warmup_max_yaw_rate_deg_s <= 0.0
        or args.warmup_hold_s <= 0.0
    ):
        raise SystemExit("warmup speed, yaw-rate limit, and hold must be positive")
    sequence = symmetric_turn_rate_sequence(
        args.rate_deg_s, args.active_hold_s, args.final_zero_hold_s
    )

    from experiments.sailboat_BO.common import load_yaml
    from experiments.sailboat_BO.run_trial import (
        ExecutionConfig,
        attempt_arm_until_armed,
        build_context,
        build_launch_plan,
        connect_mavlink,
        get_mavlink_module,
        get_repo_root,
        get_workspace_root,
        is_tcp_endpoint_open,
        load_param_overrides_from_file,
        publish_sail_force_enabled,
        publish_startup_state_held,
        render_trial_world,
        request_message_intervals,
        set_vehicle_mode,
        start_launch_process,
        stop_launch_process,
        validate_execution_backend,
        wait_for_endpoint_closed,
        write_trial_manifest,
        write_trial_param_file,
    )
    from experiments.sailboat_physics.diagnose_wind import wait_for_wind_settle
    from experiments.sailboat_physics.wind import Vector3

    repo_root = get_repo_root()
    workspace_root = get_workspace_root(repo_root)
    config_path = resolve_path(repo_root, args.config)
    params_path = resolve_path(repo_root, args.params_file)
    results_dir = resolve_path(repo_root, args.results_dir)
    config = validate_performance_config(load_yaml(config_path))
    condition = config["condition"]
    performance = config["performance"]
    missing = [name for name in ("gz", "ros2", "ardurover") if shutil.which(name) is None]
    if missing:
        raise SystemExit(
            "steering-rate trial must run inside the simulation container; missing: "
            + ", ".join(missing)
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
    params = load_param_overrides_from_file(params_path)
    base_param_path = resolve_path(repo_root, config["study"]["base_param_file"])
    base_values = parse_ardupilot_parameter_file(base_param_path)
    base_values.update(params)
    validate_wind_only_servo_functions(base_values)
    validate_wind_only_sdf(
        repo_root / "models/dave_robot_models/description/sailboat/model.sdf"
    )

    wind = wind_world_enu_from_condition(
        float(condition["tws_mps"]),
        float(condition["target_heading_deg"]),
        float(condition["signed_twa_deg"]),
    )
    scenario = make_scenario(config, wind)
    scenario["id"] = "steering_rate_tracking"
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
    launch_plan = build_launch_plan(
        context, execution, trial_world_name, prepare_remote_inputs=True
    )
    manifest = json.loads(context.files.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "steering_rate_tracking",
            "performance_config": str(config_path),
            "params_file": str(params_path),
            "control_contract": "GUIDED BODY_NED vx + yaw_rate",
            "rate_sequence_deg_s_and_duration_s": [list(item) for item in sequence],
            "waypoint_mission": False,
            "residual_rl": False,
            "wind_only": True,
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
    records: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {"status": "failure", "failure_reason": "not_started"}
    exit_code = 1
    try:
        print(f"[steering_rate] trial_id: {context.trial_id}")
        print(f"[steering_rate] launch: {launch_plan.display_cmd}")
        process = start_launch_process(context, launch_plan)
        publish_startup_state_held(context, execution, True, reason="rate_tracking_prearm")
        publish_sail_force_enabled(context, execution, False, reason="rate_tracking_prearm")
        collector = PerformanceRosCollector(
            str(context.study_cfg["namespace"]),
            str(scenario["world"].get("internal_world_name", "waves")),
        )
        collector.start()
        master = connect_mavlink(
            endpoint,
            timeout_s=float(context.study_cfg.get("mavlink_connect_timeout_s", 120.0)),
            launch_process=process,
            stdout_log=context.files.logs_dir / "launch.stdout.log",
            stderr_log=context.files.logs_dir / "launch.stderr.log",
        )
        time.sleep(3.0)
        arm_vehicle(master, context, attempt_arm_until_armed)
        set_vehicle_mode(master, "GUIDED", timeout_s=10.0)
        mavlink = get_mavlink_module(master)
        request_message_intervals(master)
        guided_mode = int(master.mode_mapping()["GUIDED"])
        target = GuidedTurnRateTarget(
            master,
            mavlink,
            0.0,
            float(performance["desired_speed_mps"]),
            float(args.target_refresh_hz),
        )
        target.send_if_due(time.monotonic(), force=True)
        ensure_residual_disabled(
            execution,
            str(performance.get("command_adapter_node", "/sailboat_command_adapter")),
        )
        collector.wait_ready(
            float(performance["telemetry_startup_timeout_s"]),
            float(performance["maximum_source_age_s"]),
        )
        settled_at = wait_for_wind_settle(
            [
                context.files.logs_dir / "launch.stdout.log",
                context.files.logs_dir / "launch.stderr.log",
            ],
            expected_world=Vector3(wind.east, wind.north, 0.0),
            tolerance_mps=float(performance["wind_settle_tolerance_mps"]),
            hold_sim_time_s=float(performance["wind_settle_hold_s"]),
            timeout_s=float(performance["wind_settle_timeout_s"]),
            poll_interval_s=float(performance["wind_settle_poll_s"]),
        )
        print(f"[steering_rate] wind settled at sim_time_s={settled_at:.3f}")
        publish_startup_state_held(context, execution, False, reason="rate_tracking")
        time.sleep(float(context.study_cfg.get("startup_state_release_settle_s", 0.2)))
        publish_sail_force_enabled(context, execution, True, reason="rate_tracking")
        time.sleep(float(context.study_cfg.get("sail_force_enable_settle_s", 0.5)))
        released = collector.wait_ready(
            float(performance["telemetry_startup_timeout_s"]),
            float(performance["maximum_source_age_s"]),
        )
        if released.sim_time_s is None:
            raise RuntimeError("release odometry has no simulation timestamp")
        release_sim_time = float(released.sim_time_s)

        mav_state = MavlinkPerformanceState()
        sample_period_s = 1.0 / float(args.sample_hz)
        poll_period_s = min(0.025, 0.5 * sample_period_s)
        last_sample_sim_time: float | None = None
        warmup_ready_since: float | None = None
        measurement_start: float | None = None
        step_index = -1
        step_start: float | None = None
        loop_start_wall = time.monotonic()
        max_wall_s = 3.0 * (
            float(args.warmup_timeout_s) + sum(duration for _, duration in sequence)
        ) + 120.0

        while True:
            now = time.monotonic()
            if process.poll() is not None:
                raise RuntimeError("simulation launch process exited")
            if now - loop_start_wall >= max_wall_s:
                outcome["failure_reason"] = "wall_timeout"
                break
            target.send_if_due(now)
            drain_mavlink(master, mavlink, mav_state)
            ros = collector.snapshot()
            if ros.sim_time_s is None or (
                last_sample_sim_time is not None
                and ros.sim_time_s - last_sample_sim_time < 0.95 * sample_period_s
            ):
                time.sleep(poll_period_s)
                continue
            last_sample_sim_time = float(ros.sim_time_s)
            elapsed = last_sample_sim_time - release_sim_time
            reference_now = time.monotonic()
            wind_age = mav_state.wind_age_s(reference_now)
            attitude_age = mav_state.attitude_age_s(reference_now)
            maximum_age = float(performance["maximum_source_age_s"])
            required = (
                ros.heading_deg,
                ros.gazebo_angular_yaw_rate_deg_s,
                ros.speed_over_ground_mps,
                ros.roll_frd_deg,
                ros.actual_rudder_angle_rad,
                ros.final_rudder_rad,
                ros.actual_boom_angle_rad,
                mav_state.wind_from_deg,
                mav_state.wind_speed_mps,
                mav_state.yaw_deg,
                mav_state.yaw_rate_deg_s,
            )
            telemetry_valid = all(
                value is not None and math.isfinite(float(value)) for value in required
            ) and telemetry_ages_are_fresh(
                (
                    ros.odom_age_s,
                    ros.rudder_age_s,
                    ros.boom_age_s,
                    ros.actuator_age_s,
                    wind_age,
                    attitude_age,
                ),
                maximum_age,
            )
            guided_active = mav_state.heartbeat_custom_mode == guided_mode

            if measurement_start is None:
                ready = (
                    telemetry_valid
                    and guided_active
                    and elapsed >= float(performance["sail_force_ramp_s"])
                    and float(ros.speed_over_ground_mps or 0.0)
                    >= float(args.warmup_min_speed_mps)
                    and abs(float(mav_state.yaw_rate_deg_s or 0.0))
                    <= float(args.warmup_max_yaw_rate_deg_s)
                )
                if ready:
                    if warmup_ready_since is None:
                        warmup_ready_since = last_sample_sim_time
                    elif last_sample_sim_time - warmup_ready_since >= float(args.warmup_hold_s):
                        measurement_start = last_sample_sim_time
                        step_index = 0
                        step_start = last_sample_sim_time
                        target.set_turn_rate(sequence[0][0])
                        target.send_if_due(now, force=True)
                        print(
                            "[steering_rate] measurement started: "
                            f"speed={ros.speed_over_ground_mps:.3f} m/s, "
                            f"heading={ros.heading_deg:.2f} deg"
                        )
                else:
                    warmup_ready_since = None
                if elapsed >= float(args.warmup_timeout_s):
                    outcome["failure_reason"] = "warmup_not_ready"
                    break
            else:
                assert step_start is not None
                if last_sample_sim_time - step_start >= sequence[step_index][1]:
                    step_index += 1
                    if step_index >= len(sequence):
                        outcome["status"] = "success"
                        outcome["failure_reason"] = ""
                        exit_code = 0
                        break
                    step_start = last_sample_sim_time
                    target.set_turn_rate(sequence[step_index][0])
                    target.send_if_due(now, force=True)
                    print(
                        f"[steering_rate] step={step_index}, "
                        f"target={sequence[step_index][0]:+.1f} deg/s"
                    )

            phase = "warmup" if measurement_start is None else "measurement"
            current_target = 0.0 if step_index < 0 else float(sequence[step_index][0])
            current_step_elapsed = (
                0.0 if step_start is None else last_sample_sim_time - step_start
            )
            measured_twa = (
                signed_twa_deg(mav_state.wind_from_deg, ros.heading_deg)
                if mav_state.wind_from_deg is not None and ros.heading_deg is not None
                else None
            )
            record = {
                "trial_id": context.trial_id,
                "phase": phase,
                "step_index": step_index,
                "step_elapsed_s": current_step_elapsed,
                "target_turn_rate_deg_s": current_target,
                "sample_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_since_release_s": elapsed,
                **asdict(ros),
                "mavlink_yaw_deg": mav_state.yaw_deg,
                "mavlink_yaw_rate_deg_s": mav_state.yaw_rate_deg_s,
                "mavlink_gazebo_yaw_rate_error_deg_s": (
                    float(mav_state.yaw_rate_deg_s)
                    - float(ros.gazebo_angular_yaw_rate_deg_s)
                    if mav_state.yaw_rate_deg_s is not None
                    and ros.gazebo_angular_yaw_rate_deg_s is not None
                    else None
                ),
                "measured_wind_from_deg": mav_state.wind_from_deg,
                "measured_wind_speed_mps": mav_state.wind_speed_mps,
                "measured_signed_twa_deg": measured_twa,
                "guided_mode_active": guided_active,
                "telemetry_valid": telemetry_valid,
                "wind_age_s": wind_age,
                "mavlink_attitude_age_s": attitude_age,
            }
            records.append(record)

            if ros.roll_frd_deg is not None and abs(ros.roll_frd_deg) > float(
                performance["maximum_roll_deg"]
            ):
                outcome["failure_reason"] = "excessive_roll"
                break
            if mav_state.heartbeat_custom_mode is not None and not guided_active:
                outcome["failure_reason"] = "guided_mode_lost"
                break
    except Exception as exc:
        outcome["failure_reason"] = "runtime_exception"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[steering_rate] ERROR: {outcome['error']}")
    finally:
        _write_csv(context.files.samples_csv, records)
        if collector is not None:
            collector.close()
        if master is not None:
            try:
                master.close()
            except Exception:
                pass
        stop_launch_process(process, execution, launch_plan)
        endpoint_closed = wait_for_endpoint_closed(
            endpoint, float(context.study_cfg.get("mavlink_release_timeout_s", 15.0))
        )
        outcome["cleanup_endpoint_closed"] = endpoint_closed
        if not endpoint_closed:
            outcome["status"] = "failure"
            outcome["failure_reason"] = "cleanup_error"
            exit_code = 1

    measurement = [record for record in records if record["phase"] == "measurement"]
    summary = {
        "trial_id": context.trial_id,
        "status": outcome["status"],
        "failure_reason": outcome["failure_reason"],
        "error": outcome.get("error"),
        "config": str(config_path),
        "params_file": str(params_path),
        "parameters": {
            name: base_values.get(name)
            for name in (
                "ATC_STR_RAT_P",
                "ATC_STR_RAT_I",
                "ATC_STR_RAT_D",
                "ATC_STR_RAT_FF",
                "ATC_STR_RAT_MAX",
                "ATC_STR_ACC_MAX",
                "ATC_STR_DEC_MAX",
            )
        },
        "command_sequence": [list(item) for item in sequence],
        "metrics": summarize_turn_rate_steps(measurement),
        "sample_count": len(records),
        "measurement_sample_count": len(measurement),
        "cleanup_endpoint_closed": outcome.get("cleanup_endpoint_closed"),
        "git_commit": git_head(repo_root),
    }
    context.files.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"samples_csv: {context.files.samples_csv}")
    print(f"summary_json: {context.files.summary_json}")
    print(f"status: {outcome['status']}")
    print(f"failure_reason: {outcome['failure_reason']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
