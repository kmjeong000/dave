from __future__ import annotations

from datetime import datetime, timezone

import pytest

from experiments.sailboat_RL.train_sac import (
    default_run_dir,
    parse_args,
    prepare_run_dir,
    validate_training_args,
)


def test_training_arguments_accept_short_smoke_configuration():
    args = parse_args(
        [
            "--total-timesteps",
            "20",
            "--learning-starts",
            "5",
            "--buffer-size",
            "100",
            "--batch-size",
            "4",
        ]
    )

    validate_training_args(args)


def test_training_arguments_reject_run_without_learning():
    args = parse_args(
        [
            "--total-timesteps",
            "10",
            "--learning-starts",
            "10",
        ]
    )

    with pytest.raises(ValueError, match="learning-starts"):
        validate_training_args(args)


def test_check_only_cannot_skip_the_check():
    args = parse_args(["--check-env-only", "--skip-env-check"])

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_training_args(args)


def test_default_run_dir_is_timestamped_below_results(tmp_path):
    now = datetime(2026, 8, 25, 1, 2, 3, tzinfo=timezone.utc)

    result = default_run_dir(tmp_path, now=now)

    assert result.parent == tmp_path / "experiments/sailboat_RL/results/sac"
    assert result.name.startswith("20260825T010203Z_")


def test_prepare_run_dir_creates_artifact_layout_and_rejects_reuse(tmp_path):
    run_dir = tmp_path / "run"

    prepare_run_dir(run_dir)

    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "models").is_dir()
    assert (run_dir / "tensorboard").is_dir()
    assert (run_dir / "trials").is_dir()
    (run_dir / "training_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        prepare_run_dir(run_dir)
