from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .core import RawState


class AttachBackend(Protocol):
    def reset(self, *, seed: int | None, options: dict[str, Any]) -> RawState:
        ...

    def step(
        self,
        rudder_residual_rad: float,
        sail_residual_rad: float,
        control_period_s: float,
    ) -> RawState:
        ...

    def close(self) -> None:
        ...


ProcessFactory = Callable[[Sequence[str], Path, Path, Path], Any]


@dataclass(frozen=True)
class LifecycleConfig:
    """Process-level configuration for one automatically managed BO trial."""

    repo_root: Path
    scenario_path: Path
    scenario_id: str
    params_file: Path
    results_dir: Path
    execution_backend: str = "local"
    docker_container: str | None = None
    container_repo_root: str | None = None
    startup_timeout_s: float = 180.0
    shutdown_timeout_s: float = 30.0
    repeat_start: int = 0
    python_executable: str = sys.executable

    def __post_init__(self) -> None:
        if self.execution_backend not in {"local", "docker-exec"}:
            raise ValueError("execution_backend must be 'local' or 'docker-exec'")
        if self.startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be positive")
        if self.shutdown_timeout_s <= 0.0:
            raise ValueError("shutdown_timeout_s must be positive")


@dataclass(frozen=True)
class EpisodeControlFiles:
    directory: Path
    ready: Path
    stop: Path
    status: Path
    stdout: Path
    stderr: Path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _tail(path: Path, line_count: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def _default_process_factory(
    command: Sequence[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()


class EpisodeLifecycleBackend:
    """Start, attach to, stop, and recreate one BO trial per Gym episode.

    The BO runner remains responsible for Gazebo/SITL launch, mission upload,
    arm/AUTO, telemetry logging, result writing, and process cleanup. This
    backend coordinates it through atomic ready/stop/status marker files and
    delegates real-time residual I/O to a fresh ROS attach backend per episode.
    """

    def __init__(
        self,
        config: LifecycleConfig,
        attach_factory: Callable[[], AttachBackend],
        *,
        process_factory: ProcessFactory | None = None,
    ):
        self.config = config
        self.attach_factory = attach_factory
        self._process_factory = process_factory or _default_process_factory
        self._episode_count = 0
        self._process: Any | None = None
        self._attach: AttachBackend | None = None
        self._control: EpisodeControlFiles | None = None
        self._last_state: RawState | None = None
        self._ready_payload: dict[str, Any] = {}
        self._last_cleanup_error = ""
        session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._session_dir = (
            self.config.results_dir
            / "_lifecycle"
            / f"{session_stamp}_{os.getpid()}"
        )

    @property
    def current_trial_id(self) -> str:
        return str(self._ready_payload.get("trial_id", ""))

    @property
    def current_summary_json(self) -> str:
        return str(self._ready_payload.get("summary_json", ""))

    @property
    def last_cleanup_error(self) -> str:
        return self._last_cleanup_error

    def _control_files(self, episode_number: int) -> EpisodeControlFiles:
        directory = self._session_dir / f"episode_{episode_number:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return EpisodeControlFiles(
            directory=directory,
            ready=directory / "ready.json",
            stop=directory / "stop.json",
            status=directory / "status.json",
            stdout=directory / "run_trial.stdout.log",
            stderr=directory / "run_trial.stderr.log",
        )

    def _build_command(
        self,
        repeat_idx: int,
        control: EpisodeControlFiles,
    ) -> list[str]:
        command = [
            self.config.python_executable,
            "-m",
            "experiments.sailboat_BO.run_trial",
            "--scenario",
            str(self.config.scenario_path),
            "--scenario-id",
            self.config.scenario_id,
            "--params-file",
            str(self.config.params_file),
            "--repeat-idx",
            str(repeat_idx),
            "--results-dir",
            str(self.config.results_dir),
            "--execution-backend",
            self.config.execution_backend,
            "--lifecycle-ready-file",
            str(control.ready),
            "--lifecycle-stop-file",
            str(control.stop),
            "--lifecycle-status-file",
            str(control.status),
        ]
        if self.config.docker_container:
            command.extend(["--docker-container", self.config.docker_container])
        if self.config.container_repo_root:
            command.extend(
                ["--container-repo-root", self.config.container_repo_root]
            )
        return command

    def _startup_error(self, message: str) -> RuntimeError:
        control = self._control
        details = []
        if control is not None:
            stdout_tail = _tail(control.stdout)
            stderr_tail = _tail(control.stderr)
            if stdout_tail:
                details.append(f"run_trial stdout tail:\n{stdout_tail}")
            if stderr_tail:
                details.append(f"run_trial stderr tail:\n{stderr_tail}")
        suffix = "\n" + "\n".join(details) if details else ""
        return RuntimeError(message + suffix)

    def _wait_until_ready(self) -> dict[str, Any]:
        if self._process is None or self._control is None:
            raise RuntimeError("episode process has not been started")
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            ready = _read_json(self._control.ready)
            if ready is not None and bool(ready.get("ready")):
                return ready
            status = _read_json(self._control.status)
            return_code = self._process.poll()
            if status is not None or return_code is not None:
                outcome = (status or {}).get("outcome", {})
                raise self._startup_error(
                    "BO trial exited before lifecycle readiness: "
                    f"return_code={return_code}, outcome={outcome}"
                )
            time.sleep(0.1)
        raise self._startup_error(
            "timed out waiting for BO mission upload, arm, and AUTO readiness"
        )

    def _status_payload(self) -> dict[str, Any] | None:
        if self._control is None:
            return None
        return _read_json(self._control.status)

    def _wait_for_terminal_status(self, state: RawState) -> RawState:
        """Reconcile local mission capture with the BO runner outcome.

        Ros2AttachBackend tracks waypoints for observations and reward shaping,
        so it can see the final capture slightly before run_trial.py.  The BO
        runner owns the canonical mission and safety rules; wait for its
        finalized status instead of exposing the local capture as a terminal
        Gym transition and then stopping the runner from close().
        """

        if self._process is None:
            return replace(
                state,
                mission_complete=False,
                termination_reason="runner_process_exit",
                termination_truncated=True,
            )

        deadline = time.monotonic() + self.config.shutdown_timeout_s
        while time.monotonic() < deadline:
            status = self._status_payload()
            if status is not None:
                return self._apply_status(state, status)

            return_code = self._process.poll()
            if return_code is not None:
                # The status marker and process exit can become visible in
                # either order. Re-read once before declaring infrastructure
                # failure.
                status = self._status_payload()
                if status is not None:
                    return self._apply_status(state, status)
                return replace(
                    state,
                    mission_complete=False,
                    termination_reason="runner_process_exit",
                    termination_truncated=True,
                )
            time.sleep(0.1)

        status = self._status_payload()
        if status is not None:
            return self._apply_status(state, status)
        return replace(
            state,
            mission_complete=False,
            termination_reason="runner_terminal_sync_timeout",
            termination_truncated=True,
        )

    @staticmethod
    def _apply_status(state: RawState, payload: dict[str, Any]) -> RawState:
        outcome = payload.get("outcome", {})
        if not isinstance(outcome, dict):
            outcome = {}
        if bool(outcome.get("mission_complete")) or outcome.get("status") == "success":
            return replace(
                state,
                mission_complete=True,
                termination_reason="mission_complete",
                termination_truncated=False,
            )

        reason = str(outcome.get("failure_reason", "")).strip()
        if not reason:
            reason = "runner_stopped"
        terminated_reasons = {"excessive_roll", "stuck_low_speed", "no_progress"}
        return replace(
            state,
            mission_complete=False,
            termination_reason=reason,
            termination_truncated=reason not in terminated_reasons,
        )

    def _request_stop(self, reason: str) -> None:
        if self._control is None or self._process is None:
            return
        if self._process.poll() is None and not self._control.stop.exists():
            _write_json_atomic(
                self._control.stop,
                {
                    "reason": reason,
                    "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    def _wait_for_process_exit(self, timeout_s: float) -> bool:
        if self._process is None:
            return True
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return True
            time.sleep(0.1)
        return self._process.poll() is not None

    def _stop_current_episode(self, reason: str) -> None:
        cleanup_errors: list[str] = []
        if self._attach is not None:
            try:
                self._attach.close()
            except Exception as exc:  # cleanup must continue to the process layer
                cleanup_errors.append(f"attach close: {type(exc).__name__}: {exc}")
            finally:
                self._attach = None

        if self._process is not None and self._process.poll() is None:
            self._request_stop(reason)
            if not self._wait_for_process_exit(self.config.shutdown_timeout_s):
                try:
                    self._process.send_signal(signal.SIGINT)
                except Exception as exc:
                    cleanup_errors.append(
                        f"runner SIGINT: {type(exc).__name__}: {exc}"
                    )
                if not self._wait_for_process_exit(10.0):
                    try:
                        self._process.kill()
                    except Exception as exc:
                        cleanup_errors.append(
                            f"runner kill: {type(exc).__name__}: {exc}"
                        )
                    self._wait_for_process_exit(2.0)
                if self._process.poll() is None:
                    cleanup_errors.append("runner process did not exit after kill")

        self._process = None
        self._control = None
        self._last_state = None
        self._ready_payload = {}
        self._last_cleanup_error = "; ".join(cleanup_errors)

    def reset(self, *, seed: int | None, options: dict[str, Any]) -> RawState:
        self._stop_current_episode("environment_reset")
        if self._last_cleanup_error:
            raise RuntimeError(
                "refusing to start a new episode after incomplete cleanup: "
                f"{self._last_cleanup_error}"
            )
        episode_number = self._episode_count
        self._episode_count += 1
        repeat_idx = int(
            options.get("repeat_idx", self.config.repeat_start + episode_number)
        )
        control = self._control_files(episode_number)
        for marker in (control.ready, control.stop, control.status):
            marker.unlink(missing_ok=True)
        command = self._build_command(repeat_idx, control)
        self._control = control
        self._process = self._process_factory(
            command,
            self.config.repo_root,
            control.stdout,
            control.stderr,
        )

        try:
            self._ready_payload = self._wait_until_ready()
            self._attach = self.attach_factory()
            state = self._attach.reset(seed=seed, options={})
        except Exception:
            self._stop_current_episode("startup_failure")
            raise

        self._last_state = state
        return state

    def step(
        self,
        rudder_residual_rad: float,
        sail_residual_rad: float,
        control_period_s: float,
    ) -> RawState:
        if self._attach is None or self._process is None or self._last_state is None:
            raise RuntimeError("reset() must start an episode before step()")

        status = self._status_payload()
        if status is not None:
            self._last_state = self._apply_status(self._last_state, status)
            return self._last_state

        state = self._attach.step(
            rudder_residual_rad,
            sail_residual_rad,
            control_period_s,
        )
        status = self._status_payload()
        if status is None and self._process.poll() is not None:
            # A final status and process exit can become visible in either
            # order. Re-read before treating the exit as a runner failure.
            status = self._status_payload()
            if status is None:
                state = replace(
                    state,
                    mission_complete=False,
                    termination_reason="runner_process_exit",
                    termination_truncated=True,
                )
        if status is not None:
            state = self._apply_status(state, status)
        elif state.mission_complete:
            state = self._wait_for_terminal_status(state)
        self._last_state = state
        return state

    def close(self) -> None:
        self._stop_current_episode("environment_close")
