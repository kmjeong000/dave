from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_SCENARIO_PATH = Path("experiments/sailboat_BO/scenario.yaml")
DEFAULT_PARAMS_PATH = Path("experiments/sailboat_BO/baseline_params.json")
DEFAULT_RESULTS_DIR = Path("experiments/sailboat_BO/results")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
