from deckard.model.defense.trainer import TrainerDefenseConfig
from test.test_model.defense_support import (
    KwargClassifierDefense,
    PositionalClassifierDefense,
    assert_rejects_non_torch_estimator,
    make_defense_config,
    run_wrapped_defense,
)


def test_trainer_defense_rejects_non_torch_estimators():
    cfg = make_defense_config(TrainerDefenseConfig, defense_params={"eps": 0.2})
    assert_rejects_non_torch_estimator(
        cfg,
        defense_type="trainer",
        defense_subtype="retraining",
        defense_class=KwargClassifierDefense,
    )


def test_trainer_defense_builds_with_kwarg_ctor_and_get_classifier(monkeypatch):
    cfg = make_defense_config(TrainerDefenseConfig, defense_params={"eps": 0.2})
    wrapped, defense, defended_estimator = run_wrapped_defense(
        cfg,
        monkeypatch,
        torch_model_check_target="deckard.model.defense.trainer._is_torch_model_instance",
        defense_type="trainer",
        defense_subtype="retraining",
        defense_class=KwargClassifierDefense,
        init_params={"epochs": 1},
    )

    assert defense.classifier is wrapped
    assert defended_estimator["wrapped"] is wrapped


def test_trainer_defense_positional_ctor_fallback_returns_wrapper(monkeypatch):
    cfg = make_defense_config(TrainerDefenseConfig, defense_params={"eps": 0.2})
    wrapped, defense, defended_estimator = run_wrapped_defense(
        cfg,
        monkeypatch,
        torch_model_check_target="deckard.model.defense.trainer._is_torch_model_instance",
        defense_type="trainer",
        defense_subtype="retraining",
        defense_class=PositionalClassifierDefense,
    )

    assert defense.classifier is wrapped
    assert defended_estimator is wrapped
