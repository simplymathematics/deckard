"""Fairness score-profile declarations and ConfigStore registrations."""

from .base import safe_store


safe_store(
    group="score",
    name="fairlearn-classification",
    node={"_target_": "deckard.plugins.fairlearn.score.DefaultFairlearnScorerConfig"},
)
safe_store(
    group="score",
    name="fairlearn-regression",
    node={"_target_": "deckard.plugins.fairlearn.score.DefaultFairlearnScorerConfig"},
)
