"""Public attack configuration exports.

This subpackage exposes the attack-side configuration objects used by
``ExperimentConfig`` and direct Python callers.
"""

from .base import AttackConfig, SensitiveFeaturesWrapper


__all__ = [
    "AttackConfig",
    "SensitiveFeaturesWrapper",
]
