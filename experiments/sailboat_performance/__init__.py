"""Sailboat sailing-performance characterization tools.

This package is intentionally separate from the BO objective and residual-RL
environment.  It measures the frozen baseline controller without changing
either of those research paths.
"""

from .core import (
    course_over_ground_deg,
    signed_twa_deg,
    vmg_toward_bearing_mps,
    wind_world_enu_from_condition,
)
from .steady_state import SteadyStateConfig, SteadyStateDetector

__all__ = [
    "SteadyStateConfig",
    "SteadyStateDetector",
    "course_over_ground_deg",
    "signed_twa_deg",
    "vmg_toward_bearing_mps",
    "wind_world_enu_from_condition",
]
