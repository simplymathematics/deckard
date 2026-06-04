import pytest

from deckard.model.defense.transformer import TransformerDefenseConfig
from test.test_model.defense_support import (
    KwargClassifierDefense,
    PositionalClassifierDefense,
    assert_rejects_non_torch_estimator,
    make_defense_call_kwargs,
    make_defense_config,
    run_wrapped_defense,
    setup_wrapped_art_builder,
)


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


def test_transformer_defense_unknown_subtype_raises():
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    with pytest.raises(ValueError, match="Unknown transformer subtype"):
        cfg(
            **make_defense_call_kwargs(
                defense_type="transformer",
                defense_subtype="bad",
                defense_class=KwargClassifierDefense,
                base_estimator=object(),
            ),
        )


def test_transformer_defense_rejects_non_torch_estimators():
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    assert_rejects_non_torch_estimator(
        cfg,
        defense_type="transformer",
        defense_subtype="evasion",
        defense_class=KwargClassifierDefense,
    )


def test_transformer_defense_kwarg_ctor_and_get_classifier(monkeypatch):
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    wrapped, defense, defended_estimator = run_wrapped_defense(
        cfg,
        monkeypatch,
        torch_model_check_target="deckard.model.defense.transformer._is_torch_model_instance",
        defense_type="transformer",
        defense_subtype="poisoning",
        defense_class=KwargClassifierDefense,
        init_params={"epochs": 1},
    )

    assert defense.classifier is wrapped
    assert defended_estimator["wrapped"] is wrapped


def test_transformer_defense_positional_ctor_fallback(monkeypatch):
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    wrapped, defense, defended_estimator = run_wrapped_defense(
        cfg,
        monkeypatch,
        torch_model_check_target="deckard.model.defense.transformer._is_torch_model_instance",
        defense_type="transformer",
        defense_subtype="evasion",
        defense_class=PositionalClassifierDefense,
    )

    assert defense.classifier is wrapped
    assert defended_estimator is wrapped


def test_transformer_defense_not_implemented_mapped_to_value_error_kwarg(monkeypatch):
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    setup_wrapped_art_builder(
        cfg,
        monkeypatch,
        target="deckard.model.defense.transformer._is_torch_model_instance",
    )

    with pytest.raises(ValueError, match="initialization failed"):
        cfg(
            **make_defense_call_kwargs(
                defense_type="transformer",
                defense_subtype="evasion",
                defense_class=_TransformerDefenseNotImplKwarg,
            ),
        )


def test_transformer_defense_not_implemented_mapped_to_value_error_positional(
    monkeypatch,
):
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    setup_wrapped_art_builder(
        cfg,
        monkeypatch,
        target="deckard.model.defense.transformer._is_torch_model_instance",
    )

    with pytest.raises(ValueError, match="initialization failed"):
        cfg(
            **make_defense_call_kwargs(
                defense_type="transformer",
                defense_subtype="evasion",
                defense_class=_TransformerDefenseNotImplPositional,
            ),
        )


def test_transformer_defense_typeerror_then_notimplemented_maps_to_value_error(
    monkeypatch,
):
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    setup_wrapped_art_builder(
        cfg,
        monkeypatch,
        target="deckard.model.defense.transformer._is_torch_model_instance",
    )

    with pytest.raises(ValueError, match="initialization failed"):
        cfg(
            **make_defense_call_kwargs(
                defense_type="transformer",
                defense_subtype="evasion",
                defense_class=_TransformerDefenseTypeThenNotImpl,
            ),
        )


def test_transformer_defense_without_get_classifier_returns_wrapper(monkeypatch):
    cfg = make_defense_config(TransformerDefenseConfig, defense_params={"beta": 0.1})
    wrapped = setup_wrapped_art_builder(
        cfg,
        monkeypatch,
        target="deckard.model.defense.transformer._is_torch_model_instance",
    )

    defense, defended_estimator = cfg(
        **make_defense_call_kwargs(
            defense_type="transformer",
            defense_subtype="evasion",
            defense_class=_TransformerDefenseNoGetClassifier,
        ),
    )

    assert defense.classifier is wrapped
    assert defended_estimator is wrapped
