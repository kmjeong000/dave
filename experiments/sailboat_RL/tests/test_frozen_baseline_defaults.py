from __future__ import annotations

from experiments.sailboat_RL.run_lifecycle_smoke import parse_args


def test_lifecycle_smoke_defaults_to_frozen_optimized_baseline():
    args = parse_args([])

    assert args.params_file.endswith("optimized_baseline_params.json")
