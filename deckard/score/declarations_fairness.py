"""Fairness score-profile declarations and ConfigStore registrations."""

from .base import safe_store
from .fairness import (
    DefaultFairlearnScoreDictConfig,
    DefaultFairlearnScoreDictConfig,
)


class DefaultFairlearnScoreDict:
    scorers = DefaultFairlearnScoreDictConfig()


class DefaultFairlearnClassificationDict:
    scorers = DefaultFairlearnScoreDictConfig(classifier=True)


class DefaultFairlearnRegressionDict:
    scorers = DefaultFairlearnScoreDictConfig(classifier=False)


safe_store(
    group="score",
    name="fairlearn-classification",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScoreDictConfig", "classifier": True},
)
safe_store(
    group="score",
    name="fairlearn-regression",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScoreDictConfig", "classifier": False},
)
