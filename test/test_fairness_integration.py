import tempfile
from pathlib import Path

import pytest

from deckard.attack import AttackConfig
from deckard.data import FairlearnDataConfig
from deckard.model import (
    DefenseConfig,
    DefensePipelineConfig,
    FairlearnDefenseConfig,
    FairlearnModelConfig,
    ModelConfig,
)
from deckard.score import FairlearnScoreDictConfig, ScorerConfig
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from fairlearn.reductions import ExponentiatedGradient

pytest.importorskip("fairlearn")
pytest.importorskip("art")


def _fairness_data():
    cfg = FairlearnDataConfig(
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
        sensitive_columns=["feature_0"],
    )
    cfg()
    return cfg


def test_fairness_data_and_model_scores():
    data = _fairness_data()

    model = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=data,
    )
    model(data)

    assert "fairness_scores" in data.score_dict
    assert "accuracy" in model.score_dict
    assert any(key.endswith("_accuracy") for key in model.score_dict)


def test_fairness_regression_data_and_metric_frame_scores():
    data = FairlearnDataConfig(
        dataset_name="make_regression",
        data_params={
            "n_samples": 40,
            "n_features": 8,
            "n_informative": 5,
            "noise": 0.1,
            "random_state": 21,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        classifier=False,
        sensitive_columns=["feature_0"],
    )
    data()

    scorer = FairlearnScoreDictConfig(
        group_scorers={
            "mse": ScorerConfig(
                score_name="mse",
                score_function="sklearn.metrics.mean_squared_error",
            ),
        },
        group_reduction="difference",
        include_group_by_group=True,
        include_group_overall=False,
    )

    model = FairlearnModelConfig(
        model_type="sklearn.linear_model.LinearRegression",
        classifier=False,
        scorer=scorer,
        data=data,
    )
    model(data)

    assert "mse_difference" in model.score_dict
    assert any(key.endswith("_mse") for key in model.score_dict)


def test_fairlearn_group_scorers_list_merges_dicts():
    scorer = FairlearnScoreDictConfig(
        group_scorers=[
            {"mse": {"score_function": "sklearn.metrics.mean_squared_error"}},
            {
                "group_scorers": {
                    "mae": {
                        "score_function": "sklearn.metrics.mean_absolute_error"
                    },
                },
            },
        ],
    )

    assert "mse" in scorer.group_scorers
    assert "mae" in scorer.group_scorers


def test_fairlearn_group_scorers_list_later_wins_on_conflict():
    scorer = FairlearnScoreDictConfig(
        group_scorers=[
            {"mse": {"score_function": "sklearn.metrics.mean_squared_error"}},
            {"mse": {"score_function": "sklearn.metrics.mean_absolute_error"}},
        ],
    )

    assert (
        scorer.group_scorers["mse"].score_function.__name__
        == "mean_absolute_error"
    )


def test_fairness_defense_config_apply_to_trained_model():
    data = _fairness_data()

    model = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=data,
    )
    model(data)

    defense = FairlearnDefenseConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        defense_name="art.defences.postprocessor.GaussianNoise",
        defense_params={"scale": 0.1},
        data=data,
    )

    defended = defense.apply_to(estimator=model.get_model(), data=data)

    assert defended is not None
    assert defense.defense_application_time is not None


def test_mixed_fairlearn_and_art_defenses_apply_with_type_checks():
    class DefenseHookProbe:
        def __init__(self):
            self.before_called = False
            self.after_called = False
            self.after_types = []

        def before_apply_defense(self, model, **kwargs):
            self.before_called = True
            chain = kwargs.get("defense_chain", [])
            assert len(chain) == 2

        def after_apply_defense(self, model, **kwargs):
            self.after_called = True
            self.after_types = kwargs.get("applied_defense_types", [])

    data = _fairness_data()
    probe = DefenseHookProbe()

    model = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 50},
        data=data,
    )
    model._train(data.X_train, data.y_train)

    fair_defense = FairlearnDefenseConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        defense_name="fairlearn.reductions.ExponentiatedGradient",
        defense_params={
            "constraints": "fairlearn.reductions.DemographicParity",
            "eps": 0.05,
        },
        data=data,
    )
    art_defense = DefenseConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        defense_name=None,
        defense_params={},
    )
    # Force generic ART wrapper so a fairlearn wrapped estimator can be nested.
    art_defense.model_type = "sklearn-classifier"

    model.defense = DefensePipelineConfig(
        defenses=[fair_defense, art_defense],
        plugins=[probe],
    )
    defended = model._apply_defense(data)

    assert probe.before_called
    assert probe.after_called
    assert "FairlearnDefenseConfig" in probe.after_types
    assert "DefenseConfig" in probe.after_types
    assert fair_defense.defense_application_time is not None
    assert art_defense.defense_application_time is not None

    # Ensure both layers are present: ART wrapper around a fairlearn estimator.
    assert isinstance(defended, ScikitlearnClassifier)
    assert isinstance(defended.model, ExponentiatedGradient)


@pytest.fixture(scope="module")
def adult_fairness_data():
    cfg = FairlearnDataConfig(
        dataset_name="adult",
        train_size=160,
        test_size=80,
        random_state=42,
        classifier=True,
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
    scores = attack_cfg(data=adult_fairness_data, model=adult_fairness_model)

    assert any(key.startswith("evasion_") for key in scores)
    assert "attack_score_time" in scores


# ---------------------------------------------------------------------------
# Hash stability and persistence
# ---------------------------------------------------------------------------


def test_fairness_data_config_hash_stable_after_execution():
    cfg = FairlearnDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 7,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        sensitive_columns=["feature_0"],
    )
    original_hash = hash(cfg)
    cfg()
    cfg.score_dict["runtime_metric"] = 1.0
    assert hash(cfg) == original_hash


def test_fairness_model_config_hash_stable_after_training():
    data = FairlearnDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 11,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        sensitive_columns=["feature_0"],
        pipeline={
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        },
    )
    data()
    model = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=data,
    )
    original_hash = hash(model)
    model(data)
    model.score_dict["extra"] = 99
    assert hash(model) == original_hash


def test_fairness_data_config_scores_persist_and_reload():
    cfg = FairlearnDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 13,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        sensitive_columns=["feature_0"],
    )
    cfg()
    scores = dict(cfg.score_dict)
    scores["n_samples"] = 40
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fairness_scores.json"
        cfg.save_scores(scores, path)
        loaded = cfg.load_scores(str(path))
    assert "n_samples" in loaded
    assert loaded["n_samples"] == 40


def test_fairness_model_config_object_pickle_roundtrip():
    data = FairlearnDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 10,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 17,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        sensitive_columns=["feature_0"],
        pipeline={
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        },
    )
    data()
    model = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=data,
    )
    model(data)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fairness_model.pkl"
        model.save_object(model, str(path))
        loaded = model.load_object(str(path))
    assert isinstance(loaded, FairlearnModelConfig)
    assert "accuracy" in loaded.score_dict
