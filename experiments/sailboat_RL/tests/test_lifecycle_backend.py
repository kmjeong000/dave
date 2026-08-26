from __future__ import annotations

import json
from pathlib import Path

from experiments.sailboat_RL.core import RawState
from experiments.sailboat_RL.lifecycle_backend import (
    EpisodeLifecycleBackend,
    LifecycleConfig,
)


def make_state(**overrides) -> RawState:
    values = {
        "sim_time_s": 10.0,
        "x_m": 0.0,
        "y_m": 0.0,
        "yaw_rad": 0.0,
        "speed_mps": 1.0,
        "roll_deg": 0.0,
        "target_x_m": 0.0,
        "target_y_m": 100.0,
        "wind_x_mps": 0.0,
        "wind_y_mps": 8.0,
        "base_rudder_rad": 0.1,
        "base_sail_rad": 0.2,
        "residual_rudder_rad": 0.0,
        "residual_sail_rad": 0.0,
        "cross_track_error_m": 0.0,
    }
    values.update(overrides)
    return RawState(**values)


class FakeAttachBackend:
    def __init__(self, step_state=None):
        self.closed = False
        self.reset_calls = []
        self.step_calls = []
        self.step_state = step_state

    def reset(self, *, seed, options):
        self.reset_calls.append((seed, options))
        return make_state()

    def step(self, rudder_residual_rad, sail_residual_rad, control_period_s):
        self.step_calls.append(
            (rudder_residual_rad, sail_residual_rad, control_period_s)
        )
        return self.step_state or make_state(sim_time_s=10.5)

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(
        self,
        stop_path: Path,
        status_path: Path,
        natural_outcome=None,
    ):
        self.stop_path = stop_path
        self.status_path = status_path
        self.natural_outcome = natural_outcome
        self.return_code = None
        self.signals = []
        self.killed = False
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        if self.return_code is None and self.natural_outcome is not None:
            self.status_path.write_text(
                json.dumps({"outcome": self.natural_outcome}),
                encoding="utf-8",
            )
            self.return_code = 0
        if self.return_code is None and self.stop_path.exists():
            request = json.loads(self.stop_path.read_text(encoding="utf-8"))
            self.status_path.write_text(
                json.dumps(
                    {
                        "outcome": {
                            "status": "failure",
                            "failure_reason": request["reason"],
                            "mission_complete": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.return_code = 0
        return self.return_code

    def send_signal(self, value):
        self.signals.append(value)
        self.return_code = -1

    def kill(self):
        self.killed = True
        self.return_code = -9


class FakeProcessFactory:
    def __init__(self, natural_outcome=None):
        self.commands = []
        self.processes = []
        self.natural_outcome = natural_outcome

    def __call__(self, command, cwd, stdout_path, stderr_path):
        del cwd, stdout_path, stderr_path
        command = list(command)
        self.commands.append(command)
        ready_path = Path(command[command.index("--lifecycle-ready-file") + 1])
        stop_path = Path(command[command.index("--lifecycle-stop-file") + 1])
        status_path = Path(command[command.index("--lifecycle-status-file") + 1])
        ready_path.write_text(
            json.dumps(
                {
                    "ready": True,
                    "trial_id": f"fake_trial_{len(self.commands)}",
                    "summary_json": str(ready_path.parent / "summary.json"),
                }
            ),
            encoding="utf-8",
        )
        process = FakeProcess(
            stop_path,
            status_path,
            natural_outcome=self.natural_outcome,
        )
        self.processes.append(process)
        return process


def make_config(tmp_path: Path) -> LifecycleConfig:
    return LifecycleConfig(
        repo_root=tmp_path,
        scenario_path=tmp_path / "scenario.yaml",
        scenario_id="eval_long_oblique",
        params_file=tmp_path / "params.json",
        results_dir=tmp_path / "results",
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
        python_executable="python-test",
    )


def test_reset_starts_ready_runner_and_close_requests_cooperative_stop(tmp_path):
    process_factory = FakeProcessFactory()
    attach = FakeAttachBackend()
    backend = EpisodeLifecycleBackend(
        make_config(tmp_path),
        lambda: attach,
        process_factory=process_factory,
    )

    state = backend.reset(seed=7, options={})

    assert state.sim_time_s == 10.0
    assert attach.reset_calls == [(7, {})]
    assert backend.current_trial_id == "fake_trial_1"
    command = process_factory.commands[0]
    assert command[:3] == ["python-test", "-m", "experiments.sailboat_BO.run_trial"]
    assert command[command.index("--execution-backend") + 1] == "local"

    backend.close()

    assert attach.closed
    assert process_factory.processes[0].return_code == 0
    assert not process_factory.processes[0].killed
    assert backend.last_cleanup_error == ""


def test_second_reset_cleans_previous_episode_before_starting_next(tmp_path):
    process_factory = FakeProcessFactory()
    attaches = []

    def attach_factory():
        attach = FakeAttachBackend()
        attaches.append(attach)
        return attach

    backend = EpisodeLifecycleBackend(
        make_config(tmp_path),
        attach_factory,
        process_factory=process_factory,
    )

    backend.reset(seed=1, options={})
    backend.reset(seed=2, options={})

    assert len(process_factory.processes) == 2
    assert process_factory.processes[0].return_code == 0
    assert attaches[0].closed
    assert not attaches[1].closed
    assert backend.current_trial_id == "fake_trial_2"
    backend.close()


def test_second_reset_starts_after_naturally_completed_runner(tmp_path):
    process_factory = FakeProcessFactory()
    attaches = []

    def attach_factory():
        attach = FakeAttachBackend()
        attaches.append(attach)
        return attach

    backend = EpisodeLifecycleBackend(
        make_config(tmp_path),
        attach_factory,
        process_factory=process_factory,
    )
    backend.reset(seed=1, options={})
    assert backend._control is not None
    backend._control.status.write_text(
        json.dumps(
            {
                "outcome": {
                    "status": "success",
                    "failure_reason": "",
                    "mission_complete": True,
                }
            }
        ),
        encoding="utf-8",
    )
    process_factory.processes[0].return_code = 0

    terminal = backend.step(0.0, 0.0, 0.5)
    backend.reset(seed=2, options={})

    assert terminal.mission_complete
    assert terminal.termination_reason == "mission_complete"
    assert attaches[0].closed
    assert len(process_factory.processes) == 2
    assert backend.current_trial_id == "fake_trial_2"
    assert backend.last_cleanup_error == ""
    backend.close()


def test_runner_failure_status_is_returned_as_terminal_state(tmp_path):
    process_factory = FakeProcessFactory()
    attach = FakeAttachBackend()
    backend = EpisodeLifecycleBackend(
        make_config(tmp_path),
        lambda: attach,
        process_factory=process_factory,
    )
    backend.reset(seed=None, options={})
    assert backend._control is not None
    backend._control.status.write_text(
        json.dumps(
            {
                "outcome": {
                    "status": "failure",
                    "failure_reason": "no_progress",
                    "mission_complete": False,
                }
            }
        ),
        encoding="utf-8",
    )

    state = backend.step(0.0, 0.0, 0.5)

    assert state.termination_reason == "no_progress"
    assert not state.termination_truncated
    assert attach.step_calls == []
    backend.close()


def test_local_mission_completion_waits_for_runner_success(tmp_path):
    process_factory = FakeProcessFactory(
        natural_outcome={
            "status": "success",
            "failure_reason": "",
            "mission_complete": True,
        }
    )
    attach = FakeAttachBackend(
        step_state=make_state(sim_time_s=10.5, mission_complete=True)
    )
    backend = EpisodeLifecycleBackend(
        make_config(tmp_path),
        lambda: attach,
        process_factory=process_factory,
    )
    backend.reset(seed=None, options={})

    state = backend.step(0.0, 0.0, 0.5)

    assert state.mission_complete
    assert state.termination_reason == "mission_complete"
    assert not state.termination_truncated
    assert not process_factory.processes[0].stop_path.exists()
    backend.close()


def test_runner_failure_overrides_local_mission_completion(tmp_path):
    process_factory = FakeProcessFactory(
        natural_outcome={
            "status": "failure",
            "failure_reason": "no_progress",
            "mission_complete": False,
        }
    )
    attach = FakeAttachBackend(
        step_state=make_state(sim_time_s=10.5, mission_complete=True)
    )
    backend = EpisodeLifecycleBackend(
        make_config(tmp_path),
        lambda: attach,
        process_factory=process_factory,
    )
    backend.reset(seed=None, options={})

    state = backend.step(0.0, 0.0, 0.5)

    assert not state.mission_complete
    assert state.termination_reason == "no_progress"
    assert not state.termination_truncated
    backend.close()
