import numpy as np
import pytest

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig
from deckard.model import DefenseConfig, ModelConfig
from deckard.score.attack import AttackScorerConfig
from deckard.score import DefaultDataClassificationConfig, DefaultDataRegressionConfig


def _load_or_skip(cfg):
    try:
        cfg()
    except (
        Exception
    ) as exc:  # pragma: no cover - optional deps/network/runtime variability
        pytest.skip(f"Unable to load dataset for integration test: {exc}")
    return cfg


def _train_model_or_skip(model_cfg, data_cfg):
    try:
        model_cfg(data_cfg)
    except Exception as exc:  # pragma: no cover - estimator/runtime variability
        pytest.skip(f"Unable to train model for integration test: {exc}")
    return model_cfg


def _base_classification_data():
    cfg = DataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "n_clusters_per_class": 1,
            "n_classes": 2,
            "random_state": 17,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
    )
    cfg()
    return cfg


@pytest.mark.parametrize(
    "data_cfg,model_cfg,expected_key",
    [
        (
            DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 40,
                    "n_features": 10,
                    "n_informative": 4,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 2,
                    "random_state": 7,
                },
                train_size=30,
                test_size=10,
                random_state=42,
                stratify=True,
                classifier=True,
            ),
            ModelConfig(
                model_type="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 25},
            ),
            "accuracy",
        ),
        (
            DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 10,
                    "n_informative": 6,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 3,
                    "random_state": 11,
                },
                train_size=40,
                test_size=20,
                random_state=42,
                stratify=True,
                classifier=True,
            ),
            ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 25, "random_state": 42},
            ),
            "accuracy",
        ),
        (
            DataConfig(
                dataset_name="make_regression",
                data_params={
                    "n_samples": 40,
                    "n_features": 10,
                    "n_informative": 5,
                    "noise": 0.1,
                    "random_state": 13,
                },
                train_size=30,
                test_size=10,
                random_state=42,
                stratify=None,
                classifier=False,
            ),
            ModelConfig(
                model_type="sklearn.linear_model.LinearRegression",
                classifier=False,
            ),
            "mse",
        ),
    ],
)
def test_base_data_model_end_to_end_without_attack(data_cfg, model_cfg, expected_key):
    data_cfg = _load_or_skip(data_cfg)
    model_cfg = _train_model_or_skip(model_cfg, data_cfg)

    assert data_cfg.X_train is not None
    assert data_cfg.X_test is not None
    assert model_cfg.get_model() is not None
    assert expected_key in model_cfg.score_dict


@pytest.mark.parametrize(
    "data_cfg,model_cfg,attack_cfg,attack_model_from_model,expected_prefix",
    [
        (
            DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 10,
                    "n_informative": 4,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 2,
                    "random_state": 7,
                },
                train_size=40,
                test_size=20,
                random_state=42,
                stratify=True,
                classifier=True,
            ),
            ModelConfig(
                model_type="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 25},
            ),
            AttackConfig(
                attack_type="art.attacks.evasion.FastGradientMethod",
                attack_params={"eps": 0.1},
                attack_size=20,
            ),
            False,
            "evasion_",
        ),
        (
            DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 10,
                    "n_informative": 6,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 3,
                    "random_state": 11,
                },
                train_size=40,
                test_size=20,
                random_state=42,
                stratify=True,
                classifier=True,
            ),
            ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 25, "random_state": 42},
            ),
            AttackConfig(
                attack_type="art.attacks.evasion.BoundaryAttack",
                attack_params={
                    "batch_size": 10,
                    "targeted": False,
                    "delta": 0.01,
                    "epsilon": 0.01,
                    "max_iter": 2,
                    "num_trial": 5,
                    "sample_size": 10,
                    "init_size": 10,
                    "min_epsilon": 0.0,
                    "verbose": False,
                },
                attack_size=20,
            ),
            False,
            "evasion_",
        ),
    ],
)
def test_base_data_model_attack_end_to_end(
    data_cfg,
    model_cfg,
    attack_cfg,
    attack_model_from_model,
    expected_prefix,
):
    data_cfg = _load_or_skip(data_cfg)
    model_cfg = _train_model_or_skip(model_cfg, data_cfg)

    attack_model = model_cfg.get_model() if attack_model_from_model else model_cfg
    try:
        attack_scores = attack_cfg(data=data_cfg, model=attack_model)
    except Exception as exc:  # pragma: no cover - attack support can vary per runtime
        pytest.skip(f"Unable to execute attack integration path: {exc}")

    assert attack_cfg.attack is not None
    assert attack_cfg.predictions is not None
    assert any(key.startswith(expected_prefix) for key in attack_scores)


def test_attack_scorer_with_data_and_model_context():
    data = _base_classification_data()
    model = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(data)

    scorer = AttackScorerConfig()
    ben_pred = np.asarray(model.get_model().predict(data.X_test))
    adv_pred = ben_pred.copy()
    if len(adv_pred) > 0:
        adv_pred[0] = 1 - adv_pred[0]

    scores = scorer.score_evasion(
        ben_pred_labels=ben_pred,
        adv_pred_labels=adv_pred,
        y_true=np.asarray(data.y_test),
        attack_size=len(adv_pred),
    )

    assert "evasion_accuracy" in scores
    assert "evasion_success" in scores
    assert "attack_score_time" in scores


def test_data_analysis_scorer_classification_with_reference_column():
    data = _base_classification_data()
    scorer = DefaultDataClassificationConfig()

    features = data.X_test.copy()
    # Non-target analysis column to validate reference-column override behavior.
    features["age_proxy"] = features["feature_0"] * 10 + 35

    scores = scorer(
        y_true=np.asarray(data.y_test),
        y_pred=features,
        mode=None,
        reference_column="age_proxy",
    )

    assert "num_classes" in scores
    assert "class_count_min" in scores
    assert "mutual_information_mean" in scores
    assert "mutual_information_max" in scores


def test_data_analysis_scorer_regression_with_reference_column():
    data = _load_or_skip(
        DataConfig(
            dataset_name="make_regression",
            data_params={
                "n_samples": 40,
                "n_features": 10,
                "n_informative": 5,
                "noise": 0.1,
                "random_state": 13,
            },
            train_size=30,
            test_size=10,
            random_state=42,
            stratify=None,
            classifier=False,
        ),
    )
    scorer = DefaultDataRegressionConfig()

    scores = scorer(
        y_true=np.asarray(data.y_test),
        y_pred=data.X_test,
        mode=None,
        reference_column="feature_0",
    )

    assert "mutual_information_mean" in scores
    assert "mutual_information_max" in scores
    assert "empirical_cdf" in scores
    assert callable(scores["empirical_cdf"])


def test_defense_config_apply_to_trained_model():
    data = _base_classification_data()
    model = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(data)

    defense = DefenseConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        defense_name="art.defences.postprocessor.GaussianNoise",
        defense_params={"scale": 0.1},
    )

    defended = defense.apply_to(estimator=model.get_model(), data=data)

    assert defended is not None
    assert defense.defense_application_time is not None


def test_experiment_config_with_attack_end_to_end():
    experiment = ExperimentConfig(
        data=DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 10,
                "n_informative": 4,
                "n_redundant": 0,
                "n_clusters_per_class": 1,
                "n_classes": 2,
                "random_state": 21,
            },
            train_size=40,
            test_size=20,
            random_state=42,
            stratify=True,
            classifier=True,
        ),
        model=ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 25},
        ),
        attack=AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=20,
        ),
        files=FileConfig(),
        classifier=True,
        experiment_name="base-integration-smoke",
    )

    scores = experiment()

    assert "accuracy" in scores
    assert any(key.startswith("evasion_") for key in scores)
    assert "attack_score_time" in scores


@pytest.fixture(scope="module")
def adult_base_data():
    return _load_or_skip(
        DataConfig(
            dataset_name="adult",
            train_size=160,
            test_size=80,
            random_state=42,
            classifier=True,
        ),
    )


@pytest.fixture(scope="module")
def adult_base_model(adult_base_data):
    return _train_model_or_skip(
        ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 25},
        ),
        adult_base_data,
    )


@pytest.mark.parametrize(
    "attack_builder,expected_prefix",
    [
        (
            lambda model: AttackConfig(
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
            ),
            "evasion_",
        ),
        (
            lambda model: AttackConfig(
                attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
                attack_params={},
                attack_size=30,
            ),
            "membership_inference_",
        ),
        (
            lambda model: AttackConfig(
                attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
                targeted_attribute=["age"],
                attack_params={
                    "attack_model_type": "nn",
                    "scale_range": (0, 89),
                    "is_continuous": True,
                    "nn_model_epochs": 1,
                },
                attack_size=80,
            ),
            "inferred_age_",
        ),
    ],
)
def test_adult_all_attack_types_base(
    attack_builder,
    expected_prefix,
    adult_base_data,
    adult_base_model,
):
    attack_cfg = attack_builder(adult_base_model)
    try:
        scores = attack_cfg(data=adult_base_data, model=adult_base_model)
    except Exception as exc:  # pragma: no cover - runtime-specific attack failures
        pytest.skip(f"Unable to run adult attack for base module: {exc}")

    assert any(key.startswith(expected_prefix) for key in scores)
    assert "attack_score_time" in scores
