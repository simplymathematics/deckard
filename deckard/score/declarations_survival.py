"""Survival score-profile declarations and ConfigStore registrations."""

from .base import safe_store
from .survival import DefaultLifelinesConfig


class DefaultLifelinesDict:
    scorers = DefaultLifelinesConfig()


safe_store(group="score", name="lifelines", node=DefaultLifelinesConfig)
