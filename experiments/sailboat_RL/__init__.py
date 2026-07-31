"""Residual reinforcement-learning environment for the DAVE sailboat."""

from .core import (
    EnvironmentConfig,
    RawState,
    RewardConfig,
    TransitionResult,
    build_observation,
    evaluate_transition,
    scale_action,
)

__all__ = [
    "EnvironmentConfig",
    "RawState",
    "RewardConfig",
    "TransitionResult",
    "build_observation",
    "evaluate_transition",
    "scale_action",
]
