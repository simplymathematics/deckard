from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from deckard.model.trainers import (
    BaseTrainer,
    PartialFitPruningTrainer,
    PartialFitTrainer,
    PretrainedTrainer,
    PruningTrainer,
    PytorchTrainer,
    SklearnTrainer,
)


class _DummyPartialFitModel:
    def __init__(self):
        self.calls = []

    def partial_fit(self, X, y, **kwargs):
        self.calls.append((X, y, kwargs))
        return self


class _DummyConfig:
    def __init__(self):
        self.trainer = "sklearn"
        self.trainer_params = {}
        self._trainer_obj = None
        self.classifier = True
        self.fit_params = {}
        self.training_time = None
        self.training_n = None
        self._model = _DummyPartialFitModel()

    def train(self, X, y):
        self.training_time = 0.1
        self.training_n = len(y)

    def save_object(self, obj, filepath):
        return None

    def load(self, filepath):
        return self

    def _is_model_fitted(self, model, X_sample=None):
        return False

    def check_prune(self, trial, value=None, step=None):
        if hasattr(trial, "report"):
            trial.report(value, step)
        if hasattr(trial, "should_prune"):
            return bool(trial.should_prune())
        return False


def test_base_trainer_resolve_and_compose_alias():
    cfg = _DummyConfig()
    cfg.trainer = "partial_fit"
    resolved = BaseTrainer.resolve(cfg)
    assert isinstance(resolved, PartialFitTrainer)

    composed = BaseTrainer.compose(cfg)
    assert isinstance(composed, PartialFitTrainer)


def test_partial_fit_trainer_executes_incremental_fit():
    cfg = _DummyConfig()
    cfg.trainer = "partial_fit"
    cfg._model = _DummyPartialFitModel()
    data = SimpleNamespace(X_train=np.array([[0.0], [1.0]]), y_train=np.array([0, 1]))

    times = BaseTrainer.execute(cfg, data, times={})

    assert "training_n" in times
    assert cfg.training_n == 2
    assert len(cfg._model.calls) == 1


def test_pretrained_trainer_raises_without_artifact_by_default(tmp_path):
    cfg = _DummyConfig()
    trainer = PretrainedTrainer(allow_fallback_training=False)
    data = SimpleNamespace(X_train=np.array([[0.0], [1.0]]), y_train=np.array([0, 1]))

    with pytest.raises(NotFittedError):
        trainer(cfg, data, model_file=str(tmp_path / "missing.pkl"), times={})


def test_pretrained_trainer_allows_fallback_training(tmp_path):
    cfg = _DummyConfig()
    trainer = PretrainedTrainer(allow_fallback_training=True)
    data = SimpleNamespace(X_train=np.array([[0.0], [1.0]]), y_train=np.array([0, 1]))

    times = trainer(cfg, data, model_file=str(tmp_path / "missing.pkl"), times={})
    assert times["training_n"] == 2


def test_pruning_trainers_set_pruned_flag():
    class _Trial:
        def report(self, value, step):
            return None

        def should_prune(self):
            return True

    cfg = _DummyConfig()
    data = SimpleNamespace(X_train=np.array([[0.0], [1.0]]), y_train=np.array([0, 1]))

    pruning_times = PruningTrainer(trial=_Trial())(cfg, data, times={})
    partial_pruning_times = PartialFitPruningTrainer(trial=_Trial())(
        cfg,
        data,
        times={},
    )

    assert pruning_times["pruned"] is True
    assert partial_pruning_times["pruned"] is True


def test_trainer_classes_available_for_runtime_selection():
    assert callable(SklearnTrainer())
    assert callable(PytorchTrainer())
