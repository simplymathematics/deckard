from types import SimpleNamespace

import pytest

from deckard.model.defense.detector import DetectorDefenseConfig
from test.test_model.defense_support import (
    make_defense_config,
    patch_torch_model_check,
)


class _DefenseCtorKwarg:
    def __init__(self, detector=None, **kwargs):
        self.detector = detector
        self.kwargs = kwargs


class _DefenseCtorPositional:
    def __init__(self, detector_classifier, **kwargs):
        self.detector_classifier = detector_classifier
        self.kwargs = kwargs


class _PoisonDefense:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, model, **init_params):
        return {"model": model, "init_params": init_params, "kwargs": self.kwargs}


def test_detector_defense_evasion_rejects_non_torch_estimators():
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})

    with pytest.raises(ValueError, match="only support neural-network models"):
        cfg(
            data=None,
            defense_type="detector",
            defense_subtype="evasion",
            defense_class=_DefenseCtorKwarg,
            art_class=object,
            init_params={},
            base_estimator=object(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_detector_defense_evasion_builds_detector_with_kwarg_ctor(monkeypatch):
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})
    wrapped = SimpleNamespace(name="wrapped")

    cfg._build_art_wrapper = lambda **kwargs: wrapped
    patch_torch_model_check(
        monkeypatch,
        "deckard.model.defense.detector._is_torch_model_instance",
    )

    defense, returned_wrapper = cfg(
        data=None,
        defense_type="detector",
        defense_subtype="evasion",
        defense_class=_DefenseCtorKwarg,
        art_class=object,
        init_params={"x": 1},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=["p"],
        existing_postprocessors=["q"],
    )

    assert returned_wrapper is wrapped
    assert defense.detector is wrapped
    assert getattr(wrapped, "_deckard_evasion_detector") is defense


def test_detector_defense_evasion_supports_positional_ctor_fallback(monkeypatch):
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})
    wrapped = SimpleNamespace(name="wrapped")

    cfg._build_art_wrapper = lambda **kwargs: wrapped
    patch_torch_model_check(
        monkeypatch,
        "deckard.model.defense.detector._is_torch_model_instance",
    )

    defense, returned_wrapper = cfg(
        data=None,
        defense_type="detector",
        defense_subtype="evasion",
        defense_class=_DefenseCtorPositional,
        art_class=object,
        init_params={"x": 1},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert returned_wrapper is wrapped
    assert defense.detector_classifier is wrapped


def test_detector_defense_poison_path_uses_model_and_init_params(monkeypatch):
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})
    cfg._model = "trained-model"
    cfg.get_model = lambda: "trained-model"

    defense, defended_estimator = cfg(
        data=None,
        defense_type="detector",
        defense_subtype="poison",
        defense_class=_PoisonDefense,
        art_class=None,
        init_params={"epochs": 2},
        base_estimator=None,
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert isinstance(defense, _PoisonDefense)
    assert defended_estimator["model"] == "trained-model"
    assert defended_estimator["init_params"] == {"epochs": 2}


def test_detector_defense_unknown_subtype_not_implemented():
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})

    with pytest.raises(NotImplementedError, match="not implemented"):
        cfg(
            data=None,
            defense_type="detector",
            defense_subtype="unknown",
            defense_class=_DefenseCtorKwarg,
            art_class=None,
            init_params={},
            base_estimator=None,
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_detector_defense_detect_evasion_public_method(monkeypatch):
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})
    wrapped = SimpleNamespace(name="wrapped")

    cfg._build_art_wrapper = lambda **kwargs: wrapped
    patch_torch_model_check(
        monkeypatch,
        "deckard.model.defense.detector._is_torch_model_instance",
    )

    defense, returned_wrapper = cfg.detect_evasion(
        defense_class=_DefenseCtorKwarg,
        art_class=object,
        init_params={"x": 1},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=["p"],
        existing_postprocessors=["q"],
    )

    assert returned_wrapper is wrapped
    assert defense.detector is wrapped


def test_detector_defense_detect_poison_public_method():
    cfg = make_defense_config(DetectorDefenseConfig, defense_params={"alpha": 0.5})
    cfg._model = "trained-model"
    cfg.get_model = lambda: "trained-model"

    defense, defended_estimator = cfg.detect_poison(
        defense_class=_PoisonDefense,
        init_params={"epochs": 2},
    )

    assert isinstance(defense, _PoisonDefense)
    assert defended_estimator["model"] == "trained-model"
    assert defended_estimator["init_params"] == {"epochs": 2}
