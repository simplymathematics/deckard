import numpy as np
import pytest

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
        name="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )

    assert callable(getattr(cfg, "check_prune"))
    assert callable(getattr(cfg, "load_cached"))


def test_model_pruner_mixin_reports_and_decides():
    class PrunedModelConfig(ModelConfig, ModelPrunerMixin):
        pass

    cfg = PrunedModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )
    trial = _DummyTrial(should_prune=True)

    should_stop = cfg.check_prune(trial, value=0.42, step=3)

    assert should_stop is True
    assert trial.reports == [(0.42, 3)]


def test_model_pruner_mixin_defaults_step_to_zero():
    class PrunedModelConfig(ModelConfig, ModelPrunerMixin):
        pass

    cfg = PrunedModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )
    trial = _DummyTrial(should_prune=False)

    should_stop = cfg.check_prune(trial, value=0.9)

    assert should_stop is False
    assert trial.reports == [(0.9, 0)]


def test_model_pruner_mixin_handles_none_trial():
    class PrunedModelConfig(ModelConfig, ModelPrunerMixin):
        pass

    cfg = PrunedModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )

    assert cfg.check_prune(None, value=0.3, step=2) is False


class _DummyLoadedModel:
    def __init__(self):
        self.loaded_paths = []

    def load(self, path):
        self.loaded_paths.append(path)
        return {"source": "model", "path": path}


class _ConfigWithLoad(PretrainedModelMixin):
    def __init__(self):
        self._model = _DummyLoadedModel()

    def load(self, path):
        return {"source": "config", "path": path}


class _ConfigWithoutLoad(PretrainedModelMixin):
    def __init__(self):
        self._model = _DummyLoadedModel()


def test_pretrained_mixin_prefers_config_loader_over_model_loader():
    cfg = _ConfigWithLoad()

    loaded = cfg.load_cached("artifact.pkl")

    assert loaded == {"source": "config", "path": "artifact.pkl"}
    assert cfg._model.loaded_paths == []


def test_pretrained_mixin_falls_back_to_model_loader():
    cfg = _ConfigWithoutLoad()

    loaded = cfg.load_cached("artifact.pkl")

    assert loaded == {"source": "model", "path": "artifact.pkl"}
    assert cfg._model.loaded_paths == ["artifact.pkl"]


def test_pretrained_mixin_raises_without_any_loader():
    class _ConfigNoLoaders(PretrainedModelMixin):
        pass

    cfg = _ConfigNoLoaders()

    with pytest.raises(NotImplementedError, match="requires a load\(path\) method"):
        cfg.load_cached("missing.pkl")


def test_model_training_mixin_called_by_train_sets_runtime_metrics():
    cfg = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        model_params={"max_iter": 10},
        classifier=True,
    )
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([0, 1, 1, 0])

    cfg.train(X, y)

    assert cfg.training_time is not None
    assert cfg.training_n == 4
