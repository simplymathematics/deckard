import numpy as np

from deckard.model.base import ModelConfig
from deckard.model._mixins import ModelPrunerMixin, PretrainedModelMixin


class _DummyTrial:
    def __init__(self, should_prune=False):
        self._should_prune = should_prune
        self.reports = []

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._should_prune


def test_model_config_exposes_prune_and_cache_mixins():
    class PrunedCachedModelConfig(ModelConfig, ModelPrunerMixin, PretrainedModelMixin):
        pass

    cfg = PrunedCachedModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )

    assert callable(getattr(cfg, "check_prune"))
    assert callable(getattr(cfg, "load_cached"))


def test_model_pruner_mixin_reports_and_decides():
    class PrunedModelConfig(ModelConfig, ModelPrunerMixin):
        pass

    cfg = PrunedModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )
    trial = _DummyTrial(should_prune=True)

    should_stop = cfg.check_prune(trial, value=0.42, step=3)

    assert should_stop is True
    assert trial.reports == [(0.42, 3)]


def test_model_training_mixin_called_by_train_sets_runtime_metrics():
    cfg = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([0, 1, 1, 0])

    cfg.train(X, y)

    assert cfg.training_time is not None
    assert cfg.training_n == 4
