"""Fairness score-profile declarations and ConfigStore registrations."""

from .base import safe_store
from .fairness import DefaultFairlearnScorerConfig


safe_store(
    group="score",
    name="fairlearn-classification",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScorerConfig"},
)
safe_store(
    group="score",
    name="fairlearn-regression",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScorerConfig"},
)
