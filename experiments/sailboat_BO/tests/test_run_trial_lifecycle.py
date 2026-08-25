from __future__ import annotations

import json

from experiments.sailboat_BO.run_trial import (
    read_lifecycle_stop_reason,
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

