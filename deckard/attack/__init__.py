"""Public attack configuration exports.

This subpackage exposes the attack-side configuration objects used by
``ExperimentConfig`` and direct Python callers.
"""

from .base import AttackConfig, SensitiveFeaturesWrapper
from .evasion import EvasionAttackConfig
from .inference import InferenceAttackConfig
from .poisoning import PoisoningAttackConfig
from .extraction import ExtractionAttackConfig
from .reconstruction import ReconstructionAttackConfig


def __getattr__(name):
    if name == "PytorchAttackConfig":
        from ..pytorch.attack import PytorchAttackConfig

        return PytorchAttackConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Import declarations to register example configs with ConfigStore
from . import declarations  # noqa: F401

__all__ = [
    "AttackConfig",
    "PytorchAttackConfig",
    "SensitiveFeaturesWrapper",
    "EvasionAttackConfig",
    "InferenceAttackConfig",
    "PoisoningAttackConfig",
    "ExtractionAttackConfig",
    "ReconstructionAttackConfig",
]
