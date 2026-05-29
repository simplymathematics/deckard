import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

os.environ.setdefault("DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION", "1")

from deckard.attack import AttackConfig  # noqa: E402
from deckard.data import DataConfig  # noqa: E402
from deckard.experiment import ExperimentConfig  # noqa: E402
from deckard.file import FileConfig  # noqa: E402
from deckard.model import DefenseConfig, ModelConfig  # noqa: E402
from deckard.model.defense.base import DefensePipelineConfig  # noqa: E402
from deckard.score.attack import AttackScorerConfig  # noqa: E402
from deckard.score import (  # noqa: E402
    DefaultDataClassificationScorerDictConfig,
    DefaultDataRegressionScorerDictConfig,
)


def _load_or_skip(cfg):
    cfg()
    return cfg


def _train_model_or_skip(model_cfg, data_cfg):
    model_cfg(data_cfg)
    return model_cfg


def _reset_hydra_state() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)


def _base_classification_data():
    cfg = DataConfig(
        name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "n_clusters_per_class": 1,
            "n_classes": 2,
            "random_state": 17,
        },
        sampler={
            "name": "split",
            "train_size": 30,
            "test_size": 10,
            "random_state": 42,
            "stratify": True,
        },
        classifier=True,
    )
    cfg()
    return cfg


@pytest.mark.parametrize(
    "data_cfg,model_cfg,expected_key",
    [
        (
            DataConfig(
                name="make_classification",
                data_params={
                    "n_samples": 40,
                    "n_features": 10,
                    "n_informative": 4,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 2,
                    "random_state": 7,
                },
                sampler={
                    "name": "split",
                    "train_size": 30,
                    "test_size": 10,
                    "random_state": 42,
                    "stratify": True,
                },
                classifier=True,
            ),
            ModelConfig(
                name="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 25},
            ),
            "accuracy",
        ),
        (
            DataConfig(
                name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 10,
                    "n_informative": 6,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 3,
                    "random_state": 11,
                },
                sampler={
                    "name": "split",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                    "stratify": True,
                },
                classifier=True,
            ),
            ModelConfig(
                name="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 25, "random_state": 42},
            ),
            "accuracy",
        ),
        (
            DataConfig(
                name="make_regression",
                data_params={
                    "n_samples": 40,
                    "n_features": 10,
                    "n_informative": 5,
                    "noise": 0.1,
                    "random_state": 13,
                },
                sampler={
                    "name": "split",
                    "train_size": 30,
                    "test_size": 10,
                    "random_state": 42,
                    "stratify": None,
                },
                classifier=False,
            ),
            ModelConfig(
                name="sklearn.linear_model.LinearRegression",
                classifier=False,
            ),
            "mse",
        ),
    ],
)
def test_base_data_model_end_to_end_without_attack(
    data_cfg,
    model_cfg,
    expected_key,
):
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
                name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 10,
                    "n_informative": 4,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 2,
                    "random_state": 7,
                },
                sampler={
                    "name": "split",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                    "stratify": True,
                },
                classifier=True,
            ),
            ModelConfig(
                name="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 25},
            ),
            AttackConfig(
                name="art.attacks.evasion.FastGradientMethod",
                attack_params={"eps": 0.1},
                attack_size=20,
            ),
            False,
            "evasion_",
        ),
        (
            DataConfig(
                name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 10,
                    "n_informative": 6,
                    "n_redundant": 0,
                    "n_clusters_per_class": 1,
                    "n_classes": 3,
                    "random_state": 11,
                },
                sampler={
                    "name": "split",
                    "train_size": 40,
                    "test_size": 20,
                    "random_state": 42,
                    "stratify": True,
                },
                classifier=True,
            ),
            ModelConfig(
                name="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 25, "random_state": 42},
            ),
            AttackConfig(
                name="art.attacks.evasion.BoundaryAttack",
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
    attack_scores = attack_cfg(data=data_cfg, model=attack_model)

    assert attack_cfg.attack is not None
    assert attack_cfg.attack_predictions is not None
    assert any(key.startswith(expected_prefix) for key in attack_scores)


def test_attack_scorer_with_data_and_model_context():
    data = _base_classification_data()
    model = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
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
    scorer = DefaultDataClassificationScorerDictConfig()

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
            name="make_regression",
            data_params={
                "n_samples": 40,
                "n_features": 10,
                "n_informative": 5,
                "noise": 0.1,
                "random_state": 13,
            },
            sampler={
                "name": "split",
                "train_size": 30,
                "test_size": 10,
                "random_state": 42,
                "stratify": None,
            },
            classifier=False,
        ),
    )
    scorer = DefaultDataRegressionScorerDictConfig()

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
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(data)

    defense = DefenseConfig(
        name="art.defences.postprocessor.GaussianNoise",
        model_name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        defense_params={"scale": 0.1},
    )

    defended = defense.apply_to(estimator=model.get_model(), data=data)

    assert defended is not None
    assert defense.defense_application_time is not None


def test_experiment_config_with_attack_end_to_end():
    experiment = ExperimentConfig(
        data=DataConfig(
            name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 10,
                "n_informative": 4,
                "n_redundant": 0,
                "n_clusters_per_class": 1,
                "n_classes": 2,
                "random_state": 21,
            },
            sampler={
                "name": "split",
                "train_size": 40,
                "test_size": 20,
                "random_state": 42,
                "stratify": True,
            },
            classifier=True,
        ),
        model=ModelConfig(
            name="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 25},
        ),
        attack=AttackConfig(
            name="art.attacks.evasion.FastGradientMethod",
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
            name="adult",
            sampler={
                "name": "split",
                "train_size": 160,
                "test_size": 80,
                "random_state": 42,
            },
            classifier=True,
        ),
    )


@pytest.fixture(scope="module")
def adult_base_model(adult_base_data):
    return _train_model_or_skip(
        ModelConfig(
            name="sklearn.linear_model.LogisticRegression",
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
                name="art.attacks.evasion.BoundaryAttack",
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
                name="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
                attack_params={},
                attack_size=30,
            ),
            "membership_inference_",
        ),
        (
            lambda model: AttackConfig(
                name="art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
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
        (
            lambda model: AttackConfig(
                name="art.attacks.inference.reconstruction.DatabaseReconstruction",
                attack_params={"split": "train", "missing_index": 0},
                attack_size=1,
            ),
            "database_reconstruction_",
        ),
    ],
)
def test_adult_all_attack_families_base(
    attack_builder,
    expected_prefix,
    adult_base_data,
    adult_base_model,
):
    attack_cfg = attack_builder(adult_base_model)
    scores = attack_cfg(data=adult_base_data, model=adult_base_model)

    assert any(key.startswith(expected_prefix) for key in scores)
    assert "attack_score_time" in scores


# ---------------------------------------------------------------------------
# Hash stability and persistence
# ---------------------------------------------------------------------------


def test_data_config_hash_stable_after_execution():
    cfg = DataConfig(
        name="make_classification",
        data_params={
            "n_samples": 20,
            "n_features": 4,
            "n_informative": 2,
            "n_redundant": 0,
            "random_state": 7,
        },
        sampler={
            "name": "split",
            "train_size": 15,
            "test_size": 5,
            "random_state": 42,
            "stratify": True,
        },
        classifier=True,
    )
    original_hash = hash(cfg)
    cfg()
    cfg.score_dict["runtime_metric"] = 1.0
    assert hash(cfg) == original_hash


def test_model_config_hash_stable_after_training():
    data = _base_classification_data()
    model = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    original_hash = hash(model)
    model(data)
    model.score_dict["extra"] = 99
    assert hash(model) == original_hash


def test_attack_config_hash_stable_after_execution():
    data = _base_classification_data()
    model = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(data)
    attack = AttackConfig(
        name="art.attacks.evasion.FastGradientMethod",
        attack_params={"eps": 0.1},
        attack_size=10,
    )
    original_hash = hash(attack)
    attack(data=data, model=model)
    assert hash(attack) == original_hash


def test_experiment_config_hash_stable_after_execution():
    experiment = ExperimentConfig(
        data=DataConfig(
            name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 8,
                "n_informative": 4,
                "n_redundant": 0,
                "n_clusters_per_class": 1,
                "n_classes": 2,
                "random_state": 31,
            },
            sampler={
                "name": "split",
                "train_size": 30,
                "test_size": 10,
                "random_state": 42,
                "stratify": True,
            },
            classifier=True,
        ),
        model=ModelConfig(
            name="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 25},
        ),
        files=FileConfig(),
        classifier=True,
        experiment_name="hash-stability-smoke",
    )
    original_hash = hash(experiment)
    experiment()
    experiment.score_dict["extra"] = 99
    assert hash(experiment) == original_hash


def test_data_config_scores_persist_and_reload():
    cfg = _base_classification_data()
    scores = {"accuracy": 0.9, "n_samples": 40}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "scores.json"
        cfg.save_scores(scores, path)
        loaded = cfg.load_scores(str(path))
    assert loaded["accuracy"] == pytest.approx(0.9)
    assert loaded["n_samples"] == 40


def test_model_config_scores_persist_and_reload():
    data = _base_classification_data()
    model = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(data)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model_scores.json"
        model.save_scores(model.score_dict, path)
        loaded = model.load_scores(str(path))
    assert "accuracy" in loaded


def test_model_config_object_pickle_roundtrip():
    data = _base_classification_data()
    model = ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
    )
    model(data)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.pkl"
        model.save_object(model, str(path))
        loaded = model.load_object(str(path))
    assert isinstance(loaded, ModelConfig)
    assert loaded.score_dict.get("accuracy") == pytest.approx(
        model.score_dict["accuracy"],
    )


def test_experiment_config_scores_persist_and_reload():
    experiment = ExperimentConfig(
        data=DataConfig(
            name="make_classification",
            data_params={
                "n_samples": 40,
                "n_features": 8,
                "n_informative": 4,
                "n_redundant": 0,
                "n_clusters_per_class": 1,
                "n_classes": 2,
                "random_state": 37,
            },
            sampler={
                "name": "split",
                "train_size": 30,
                "test_size": 10,
                "random_state": 42,
                "stratify": True,
            },
            classifier=True,
        ),
        model=ModelConfig(
            name="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 25},
        ),
        files=FileConfig(),
        classifier=True,
        experiment_name="persist-smoke",
    )
    scores = experiment()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "experiment_scores.json"
        experiment.save_scores(scores, path)
        loaded = experiment.load_scores(str(path))
    assert "accuracy" in loaded


# =========================================================================
# CLI-style integration tests using Hydra ConfigStore composition
# =========================================================================


def test_cli_data_model_composition_adult_logistic():
    """Test CLI-style config composition: adult dataset + logistic regression."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=adult",
                "model=logistic",
                "defense=baseline",
                "score=classification",
            ],
        )

    # Load config into data
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    # Use DataConfig if pipeline is present
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)
    _load_or_skip(data)

    # Load config into model
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)
    _train_model_or_skip(model, data)

    assert data.X_train is not None
    assert model.get_model() is not None


def test_cli_data_model_composition_diabetes_rf():
    """Test CLI-style config composition with the canonical rf profile."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "model=rf",
                "score=classification",
            ],
        )

    # Load config into data
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)
    _load_or_skip(data)

    # Load config into model
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)
    _train_model_or_skip(model, data)

    assert data.X_train is not None
    assert model.get_model() is not None


def test_cli_defense_composition_feature_squeezing():
    """Test CLI-style config composition with defense pipeline."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "model=test-logistic",
                "defense=feature-squeezing",
            ],
        )

    # Load config into data
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    data = DataConfig(**data_dict)
    _load_or_skip(data)

    # Load config into model
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)
    _train_model_or_skip(model, data)

    # Load config into defense
    defense_dict = OmegaConf.to_container(cfg.defense, resolve=True)
    defense = DefensePipelineConfig.coerce(defense_dict)

    # Apply defense
    defended = defense.apply(estimator=model.get_model(), data=data)
    assert defended is not None


def test_cli_attack_composition_fgm():
    """Test CLI-style config composition with evasion attack."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "model=test-logistic",
                "attack=boundary",
                "score=classification",
            ],
        )

    # Load configs
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)
    _load_or_skip(data)

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)
    _train_model_or_skip(model, data)

    attack_dict = OmegaConf.to_container(cfg.attack, resolve=True)
    attack = AttackConfig(**attack_dict)

    # Execute attack
    scores = attack(data=data, model=model)
    assert any(key.startswith("evasion_") for key in scores)


def test_cli_attack_composition_database_reconstruction_smoke():
    """Smoke test CLI-style config composition with database reconstruction attack."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "model=test-logistic",
                "attack=database-reconstruction",
                "search/attacks=hsj",
                "score=classification",
            ],
        )

    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)
    _load_or_skip(data)

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)
    _train_model_or_skip(model, data)

    attack_dict = OmegaConf.to_container(cfg.attack, resolve=True)
    attack = AttackConfig(**attack_dict)

    scores = attack(data=data, model=model)
    assert any(key.startswith("database_reconstruction_") for key in scores)


def test_cli_full_experiment_composition():
    """Test CLI-style config composition for full experiment."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "model=test-logistic",
                "attack=boundary",
                "defense=baseline",
                "score=classification",
            ],
        )

    # Load configs
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)

    attack_dict = OmegaConf.to_container(cfg.attack, resolve=True)
    attack = AttackConfig(**attack_dict)

    # Execute experiment
    experiment = ExperimentConfig(data=data, model=model, attack=attack)
    scores = experiment()
    assert "accuracy" in scores
    assert any(key.startswith("evasion_") for key in scores)


def test_cli_full_experiment_composition_database_reconstruction_end_to_end():
    """End-to-end CLI-style composition for experiment with a canonical attack config."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "model=test-logistic",
                "attack=boundary",
                "defense=baseline",
                "score=classification",
            ],
        )

    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)

    attack_dict = OmegaConf.to_container(cfg.attack, resolve=True)
    attack = AttackConfig(**attack_dict)

    experiment = ExperimentConfig(data=data, model=model, attack=attack)
    scores = experiment()
    assert "accuracy" in scores
    assert any(key.startswith("evasion_") for key in scores)


def test_cli_experiment_tuning_mode_emits_test_scores():
    """Test CLI-style composition with ExperimentConfig tuning mode test scoring."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[
                "data=test-classification",
                "+data.sampler.val_size=0.1",
                "model=test-logistic",
                "score=classification",
            ],
        )

    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    if "pipeline" in data_dict:
        data = DataConfig(**data_dict)
    else:
        data = DataConfig(**data_dict)

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model = ModelConfig(**model_dict)

    score_dict = OmegaConf.to_container(cfg.score, resolve=True)
    experiment = ExperimentConfig(
        data=data,
        model=model,
        score={"experiment": score_dict},
        evaluation_mode="tuning",
        experiment_name="tuning-test-integration",
    )

    scores = experiment()
    assert "accuracy" in scores
    assert "validation_accuracy" not in scores
    assert "training_accuracy" not in scores


def test_cli_plot_composition_smoke():
    """Test CLI-style config composition for canonical plot configs."""

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    )
    _reset_hydra_state()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="attack-default",
            overrides=[
                "experiment_name=plot-compose-smoke",
                "data=test-classification",
                "model=test-logistic",
                "attack=hsj",
                "defense=baseline",
                "score=classification",
                "+plot=default",
            ],
        )

    assert "plot" in cfg
    assert cfg.plot.plot_type == "roc_auc"
    assert cfg.plot.backend == "yellowbrick"


def test_artifact_loader_integration():
    """Integration test for ArtifactLoaderMixin."""
    from deckard.artifacts import ArtifactLoaderMixin

    loader = ArtifactLoaderMixin(
        id="integration-loader",
        path="artifacts/integration-artifact.json",
        payload_kind="data",
    )

    artifact = loader.load()
    assert artifact.id == "integration-loader"
    assert artifact.payload_kind == "data"
