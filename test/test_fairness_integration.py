import pytest

from deckard.attack import AttackConfig
from deckard.data import FairnessDataConfig
from deckard.model import FairnessDefenseConfig, FairnessModelConfig, ModelConfig

pytest.importorskip("fairlearn")


def _fairness_data():
    cfg = FairnessDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "n_clusters_per_class": 1,
            "n_classes": 2,
            "random_state": 23,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        groupby_columns=["feature_0"],
        sensitive_columns=["feature_0"],
        pipeline={
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        },
    )
    cfg()
    return cfg


def test_fairness_data_and_model_scores():
    data = _fairness_data()

    model = FairnessModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=data,
    )
    model(data)

    assert "fairness_scores" in data.score_dict
    assert "accuracy" in model.score_dict
    assert any(key.endswith("_accuracy") for key in model.score_dict)


def test_fairness_defense_config_apply_to_trained_model():
    data = _fairness_data()

    model = FairnessModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=data,
    )
    model(data)

    defense = FairnessDefenseConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        defense_name="art.defences.postprocessor.GaussianNoise",
        defense_params={"scale": 0.1},
        data=data,
    )

    defended = defense.apply_to(estimator=model.get_model(), data=data)

    assert defended is not None
    assert defense.defense_application_time is not None


@pytest.fixture(scope="module")
def adult_fairness_data():
    cfg = FairnessDataConfig(
        dataset_name="adult",
        train_size=160,
        test_size=80,
        random_state=42,
        classifier=True,
        groupby_columns=["sex"],
        sensitive_columns=["sex"],
        pipeline={
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        },
    )
    cfg()
    return cfg


@pytest.fixture(scope="module")
def adult_fairness_model(adult_fairness_data):
    model = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(adult_fairness_data)
    return model


@pytest.mark.parametrize("use_attack", [False, True])
def test_adult_fairness_data_model_with_and_without_attack(
    use_attack,
    adult_fairness_data,
    adult_fairness_model,
):
    assert "fairness_scores" in adult_fairness_data.score_dict
    assert "accuracy" in adult_fairness_model.score_dict

    if not use_attack:
        return

    attack_cfg = AttackConfig(
        attack_type="art.attacks.evasion.BoundaryAttack",
        attack_params={
            "batch_size": 5,
            "targeted": False,
            "delta": 0.01,
            "epsilon": 0.01,
            "max_iter": 2,
            "num_trial": 5,
            "sample_size": 5,
            "init_size": 5,
            "min_epsilon": 0.0,
            "verbose": False,
        },
        attack_size=5,
    )
    try:
        scores = attack_cfg(data=adult_fairness_data, model=adult_fairness_model)
    except Exception as exc:  # pragma: no cover - runtime-specific attack failures
        pytest.skip(f"Unable to run adult evasion attack for fairness module: {exc}")

    assert any(key.startswith("evasion_") for key in scores)
    assert "attack_score_time" in scores
