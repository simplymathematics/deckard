from types import SimpleNamespace

import pytest

from deckard.model.trainer import TrainerDefenseConfig


class _TrainerDefenseKwarg:
    def __init__(self, classifier=None, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs

    def get_classifier(self):
        return {"wrapped": self.classifier, "kwargs": self.kwargs}


class _TrainerDefensePositional:
    def __init__(self, classifier, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs


def _make_config():
    cfg = TrainerDefenseConfig()
    cfg.defense_params = {"eps": 0.2}
    cfg._model = None
    return cfg


def test_trainer_defense_rejects_non_torch_estimators():
    cfg = _make_config()
    with pytest.raises(ValueError, match="only support neural-network models"):
        cfg(
            data=None,
            defense_type="trainer",
            defense_subtype="retraining",
            defense_class=_TrainerDefenseKwarg,
            art_class=object,
            init_params={},
            base_estimator=object(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_trainer_defense_builds_with_kwarg_ctor_and_get_classifier(monkeypatch):
    cfg = _make_config()
    wrapped = SimpleNamespace(name="wrapped")
    cfg._build_art_wrapper = lambda **kwargs: wrapped
    monkeypatch.setattr(
        "deckard.model.trainer._is_torch_model_instance", lambda _m: True
    )

    defense, defended_estimator = cfg(
        data=None,
        defense_type="trainer",
        defense_subtype="retraining",
        defense_class=_TrainerDefenseKwarg,
        art_class=object,
        init_params={"epochs": 1},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert defense.classifier is wrapped
    assert defended_estimator["wrapped"] is wrapped


def test_trainer_defense_positional_ctor_fallback_returns_wrapper(monkeypatch):
    cfg = _make_config()
    wrapped = SimpleNamespace(name="wrapped")
    cfg._build_art_wrapper = lambda **kwargs: wrapped
    monkeypatch.setattr(
        "deckard.model.trainer._is_torch_model_instance", lambda _m: True
    )

    defense, defended_estimator = cfg(
        data=None,
        defense_type="trainer",
        defense_subtype="retraining",
        defense_class=_TrainerDefensePositional,
        art_class=object,
        init_params={},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert defense.classifier is wrapped
    assert defended_estimator is wrapped
