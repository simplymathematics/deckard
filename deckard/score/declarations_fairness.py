"""Fairness score-profile declarations and ConfigStore registrations."""

from .base import safe_store
from .fairness import (
    DefaultFairlearnConfig,
    DefaultFairlearnScoreConfig,
)


class DefaultFairlearnDict:
    scorers = DefaultFairlearnConfig()


class DefaultFairlearnClassificationDict:
    scorers = DefaultFairlearnScoreConfig(classifier=True)


class DefaultFairlearnRegressionDict:
    scorers = DefaultFairlearnScoreConfig(classifier=False)


safe_store(
    group="score",
    name="fairlearn-classification",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScoreConfig", "classifier": True},
)
safe_store(
    group="score",
    name="fairlearn-regression",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScoreConfig", "classifier": False},
)
