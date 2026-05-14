from types import SimpleNamespace

import pytest

from deckard.model.transformer import TransformerDefenseConfig


class _TransformerDefenseKwarg:
    def __init__(self, classifier=None, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs

    def get_classifier(self):
        return {"wrapped": self.classifier, "kwargs": self.kwargs}


class _TransformerDefensePositional:
    def __init__(self, classifier, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs


class _TransformerDefenseNotImplKwarg:
    def __init__(self, classifier=None, **kwargs):
        raise NotImplementedError("unsupported")


class _TransformerDefenseNotImplPositional:
    def __init__(self, classifier, **kwargs):
        raise NotImplementedError("unsupported")


class _TransformerDefenseTypeThenNotImpl:
    def __init__(self, *args, **kwargs):
        if "classifier" in kwargs:
            raise TypeError("keyword classifier unsupported")
        raise NotImplementedError("unsupported positional backend")


class _TransformerDefenseNoGetClassifier:
    def __init__(self, classifier=None, **kwargs):
        self.classifier = classifier


def _make_config():
    cfg = TransformerDefenseConfig()
    cfg.defense_params = {"beta": 0.1}
    cfg._model = None
    return cfg


def test_transformer_defense_unknown_subtype_raises():
    cfg = _make_config()
    with pytest.raises(ValueError, match="Unknown transformer subtype"):
        cfg(
            data=None,
            defense_type="transformer",
            defense_subtype="bad",
            defense_class=_TransformerDefenseKwarg,
            art_class=object,
            init_params={},
            base_estimator=object(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_transformer_defense_rejects_non_torch_estimators():
    cfg = _make_config()
    with pytest.raises(ValueError, match="only support neural-network models"):
        cfg(
            data=None,
            defense_type="transformer",
            defense_subtype="evasion",
            defense_class=_TransformerDefenseKwarg,
            art_class=object,
            init_params={},
            base_estimator=object(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_transformer_defense_kwarg_ctor_and_get_classifier(monkeypatch):
    cfg = _make_config()
    wrapped = SimpleNamespace(name="wrapped")
    cfg._build_art_wrapper = lambda **kwargs: wrapped
    monkeypatch.setattr("deckard.model.transformer._is_torch_model_instance", lambda _m: True)

    defense, defended_estimator = cfg(
        data=None,
        defense_type="transformer",
        defense_subtype="poisoning",
        defense_class=_TransformerDefenseKwarg,
        art_class=object,
        init_params={"epochs": 1},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert defense.classifier is wrapped
    assert defended_estimator["wrapped"] is wrapped


def test_transformer_defense_positional_ctor_fallback(monkeypatch):
    cfg = _make_config()
    wrapped = SimpleNamespace(name="wrapped")
    cfg._build_art_wrapper = lambda **kwargs: wrapped
    monkeypatch.setattr("deckard.model.transformer._is_torch_model_instance", lambda _m: True)

    defense, defended_estimator = cfg(
        data=None,
        defense_type="transformer",
        defense_subtype="evasion",
        defense_class=_TransformerDefensePositional,
        art_class=object,
        init_params={},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert defense.classifier is wrapped
    assert defended_estimator is wrapped


def test_transformer_defense_not_implemented_mapped_to_value_error_kwarg(monkeypatch):
    cfg = _make_config()
    cfg._build_art_wrapper = lambda **kwargs: SimpleNamespace(name="wrapped")
    monkeypatch.setattr("deckard.model.transformer._is_torch_model_instance", lambda _m: True)

    with pytest.raises(ValueError, match="initialization failed"):
        cfg(
            data=None,
            defense_type="transformer",
            defense_subtype="evasion",
            defense_class=_TransformerDefenseNotImplKwarg,
            art_class=object,
            init_params={},
            base_estimator=SimpleNamespace(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_transformer_defense_not_implemented_mapped_to_value_error_positional(monkeypatch):
    cfg = _make_config()
    cfg._build_art_wrapper = lambda **kwargs: SimpleNamespace(name="wrapped")
    monkeypatch.setattr("deckard.model.transformer._is_torch_model_instance", lambda _m: True)

    with pytest.raises(ValueError, match="initialization failed"):
        cfg(
            data=None,
            defense_type="transformer",
            defense_subtype="evasion",
            defense_class=_TransformerDefenseNotImplPositional,
            art_class=object,
            init_params={},
            base_estimator=SimpleNamespace(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_transformer_defense_typeerror_then_notimplemented_maps_to_value_error(
    monkeypatch,
):
    cfg = _make_config()
    cfg._build_art_wrapper = lambda **kwargs: SimpleNamespace(name="wrapped")
    monkeypatch.setattr("deckard.model.transformer._is_torch_model_instance", lambda _m: True)

    with pytest.raises(ValueError, match="initialization failed"):
        cfg(
            data=None,
            defense_type="transformer",
            defense_subtype="evasion",
            defense_class=_TransformerDefenseTypeThenNotImpl,
            art_class=object,
            init_params={},
            base_estimator=SimpleNamespace(),
            existing_preprocessors=[],
            existing_postprocessors=[],
        )


def test_transformer_defense_without_get_classifier_returns_wrapper(monkeypatch):
    cfg = _make_config()
    wrapped = SimpleNamespace(name="wrapped")
    cfg._build_art_wrapper = lambda **kwargs: wrapped
    monkeypatch.setattr("deckard.model.transformer._is_torch_model_instance", lambda _m: True)

    defense, defended_estimator = cfg(
        data=None,
        defense_type="transformer",
        defense_subtype="evasion",
        defense_class=_TransformerDefenseNoGetClassifier,
        art_class=object,
        init_params={},
        base_estimator=SimpleNamespace(),
        existing_preprocessors=[],
        existing_postprocessors=[],
    )

    assert defense.classifier is wrapped
    assert defended_estimator is wrapped
