"""Survival score-profile declarations and ConfigStore registrations."""

from .base import safe_store
from .survival import DefaultLifelinesConfig


safe_store(group="score", name="lifelines", node=DefaultLifelinesConfig)
