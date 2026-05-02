"""Public attack configuration exports.

This subpackage exposes the attack-side configuration objects used by
``ExperimentConfig`` and direct Python callers.
"""

from .base import AttackConfig, SensitiveFeaturesWrapper

# Import declarations to register example configs with ConfigStore
from . import declarations  # noqa: F401

__all__ = ["AttackConfig", "SensitiveFeaturesWrapper"]
