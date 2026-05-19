"""Public attack configuration exports.

This subpackage exposes the attack-side configuration objects used by
``ExperimentConfig`` and direct Python callers.
"""

from .base import AttackConfig, SensitiveFeaturesWrapper
from .evasion import EvasionAttackConfig
from .extraction import ExtractionAttackConfig
from .inference import InferenceAttackConfig
from .poisoning import PoisoningAttackConfig
from .reconstruction import ReconstructionAttackConfig


def __getattr__(name):
    if name == "PytorchAttackConfig":
        from ..frameworks.pytorch.attack import PytorchAttackConfig

        return PytorchAttackConfig
    raise ModuleNotFoundError(f"Can't import {name}. Is the dependency installed?")


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
