"""Named score-profile declarations and ConfigStore registrations."""

from .base import DefaultClassifierConfig, DefaultRegressorConfig, safe_store
from .data import DefaultDataClassificationConfig, DefaultDataRegressionConfig
from .fairness import (
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnConfig,
    DefaultFairlearnRegressionConfig,
)
from .survival import DefaultLifelinesConfig


class DefaultClassifierDict:
    scorers = DefaultClassifierConfig()


class DefaultRegressorDict:
    scorers = DefaultRegressorConfig()


class DefaultFairlearnDict:
    scorers = DefaultFairlearnConfig()


class DefaultFairlearnClassificationDict:
    scorers = DefaultFairlearnClassificationConfig()


class DefaultFairlearnRegressionDict:
    scorers = DefaultFairlearnRegressionConfig()


class DefaultLifelinesDict:
    scorers = DefaultLifelinesConfig()


class DefaultDataClassificationDict:
    scorers = DefaultDataClassificationConfig()


class DefaultDataRegressionDict:
    scorers = DefaultDataRegressionConfig()


safe_store(group="score", name="fairlearn-classification", node=DefaultFairlearnClassificationConfig)
safe_store(group="score", name="fairlearn-regression", node=DefaultFairlearnRegressionConfig)
safe_store(group="score", name="lifelines", node=DefaultLifelinesConfig)
