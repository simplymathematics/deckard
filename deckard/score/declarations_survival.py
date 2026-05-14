"""Survival score-profile declarations and ConfigStore registrations."""

from .base import safe_store


safe_store(
	group="score",
	name="lifelines",
	node={"_target_": "deckard.plugins.lifelines.score.DefaultLifelinesConfig"},
)
