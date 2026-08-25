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
from .lifecycle_backend import EpisodeLifecycleBackend, LifecycleConfig

__all__ = [
    "EnvironmentConfig",
    "RawState",
    "RewardConfig",
    "TransitionResult",
    "build_observation",
    "evaluate_transition",
    "scale_action",
    "EpisodeLifecycleBackend",
    "LifecycleConfig",
]
