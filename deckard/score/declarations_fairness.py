"""Fairness score-profile declarations and ConfigStore registrations."""

from .base import safe_store
from .fairness import (
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnConfig,
    DefaultFairlearnRegressionConfig,
)


class DefaultFairlearnDict:
    scorers = DefaultFairlearnConfig()


class DefaultFairlearnClassificationDict:
    scorers = DefaultFairlearnClassificationConfig()


class DefaultFairlearnRegressionDict:
    scorers = DefaultFairlearnRegressionConfig()


safe_store(
    group="score",
    name="fairlearn-classification",
    node=DefaultFairlearnClassificationConfig,
)
safe_store(
    group="score",
    name="fairlearn-regression",
    node=DefaultFairlearnRegressionConfig,
)
