from __future__ import annotations

import json
import re
import shlex
from unittest.mock import patch

from experiments.sailboat_BO.run_trial import (
    ExecutionConfig,
    read_lifecycle_stop_reason,
    run_cleanup_token,
    write_json_marker,
)


def test_write_json_marker_creates_atomic_payload(tmp_path):
    marker = tmp_path / "control" / "ready.json"

    write_json_marker(marker, {"ready": True, "trial_id": "trial-1"})

    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "ready": True,
        "trial_id": "trial-1",
    }
    assert list(marker.parent.glob("*.tmp")) == []


def test_read_lifecycle_stop_reason_uses_requested_reason(tmp_path):
    marker = tmp_path / "stop.json"
    marker.write_text(
        json.dumps({"reason": "environment_reset"}),
        encoding="utf-8",
    )

    assert read_lifecycle_stop_reason(marker) == "environment_reset"


def test_read_lifecycle_stop_reason_defaults_for_invalid_marker(tmp_path):
    marker = tmp_path / "stop.json"
    marker.write_text("not-json", encoding="utf-8")

    assert read_lifecycle_stop_reason(marker) == "external_stop"


def test_cleanup_token_terminates_then_kills_without_matching_its_shell():
    execution = ExecutionConfig(backend="local")

    with patch("experiments.sailboat_BO.run_trial.subprocess.run") as run:
        run_cleanup_token(execution, "20260826T010203Z__train_crosswind__r00")

    command = run.call_args.args[0]
    assert command[:2] == ["bash", "-lc"]
    script = command[2]
    assert "pkill -TERM -f" in script
    assert "pkill -KILL -f" in script
    assert "sleep 0.5" in script
    pattern = shlex.split(script.split("pkill -TERM -f -- ", 1)[1])[0]
    assert re.search(pattern, "ardurover 20260826T010203Z__train_crosswind__r00")
    assert not re.search(pattern, script)
