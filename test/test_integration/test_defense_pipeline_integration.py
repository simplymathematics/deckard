import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig
from deckard.model import DefenseConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str):
    return OmegaConf.to_container(OmegaConf.load(ROOT / path), resolve=True)


def _base_classification_data():
    cfg = DataConfig(
        name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 8,
            "n_informative": 4,
            "n_redundant": 0,
            "n_clusters_per_class": 1,
            "n_classes": 2,
            "random_state": 17,
        },
        classifier=True,
        sampler={
            "name": "deckard.data.sample.SplitSampler",
            "train_size": 30,
            "test_size": 10,
            "random_state": 42,
            "stratify": True,
        },
    )
    cfg()
    return cfg


def _base_model(defense=None):
    return ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        defense=defense,
    )


def _apply_defense(defense):
    data = _base_classification_data()
    model = _base_model(defense=defense)
    model.train(data.X_train, data.y_train)
    wrapped = model.apply_defense(data)
    return data, model, wrapped


def _wrapper_defense_names(wrapper, attr_name: str):
    defenses = getattr(wrapper, attr_name, None) or []
    return [type(defense).__name__ for defense in defenses]


def _assert_wrapper_defenses(
    wrapper,
    expected_preprocessors,
    expected_postprocessors,
):
    assert (
        _wrapper_defense_names(wrapper, "preprocessing_defences")
        == expected_preprocessors
    )
    assert (
        _wrapper_defense_names(wrapper, "postprocessing_defences")
        == expected_postprocessors
    )


def _example_defense(path: str, *, overrides=None):
    defense = _load_yaml(path)
    if overrides:
        defense_params = dict(defense.get("defense_params", {}))
        defense_params.update(overrides)
        defense["defense_params"] = defense_params
    return defense


@pytest.mark.parametrize(
    "defense_path,expected_preprocessors,expected_postprocessors,expected_name",
    [
        (
            "examples/sklearn/config/defense/feature-squeezing.yaml",
            ["FeatureSqueezing"],
            [],
            "art.defences.preprocessor.FeatureSqueezing",
        ),
        (
            "examples/sklearn/config/defense/gaussian-noise.yaml",
            [],
            ["GaussianNoise"],
            "art.defences.postprocessor.GaussianNoise",
        ),
    ],
)
def test_model_config_accepts_legacy_single_defense_yaml(
    defense_path,
    expected_preprocessors,
    expected_postprocessors,
    expected_name,
):
    _, model, wrapped = _apply_defense(str(ROOT / defense_path))

    assert len(model.defense.defenses) == 1
    assert model.defense.defenses[0].defense_name == expected_name
    assert model.defense_application_time is not None
    _assert_wrapper_defenses(
        wrapped,
        expected_preprocessors,
        expected_postprocessors,
    )


def test_experiment_config_accepts_legacy_single_defense_yaml():
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
                "random_state": 23,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": 30,
                "test_size": 10,
                "random_state": 42,
                "stratify": True,
            },
        ),
        model=ModelConfig(
            name="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 25},
        ),
        defense=str(
            ROOT / "examples/sklearn/config/defense/feature-squeezing.yaml",
        ),
        files=FileConfig(),
        classifier=True,
        experiment_name="legacy-defense-smoke",
    )

    assert len(experiment.defense.defenses) == 1
    assert experiment.model.defense is experiment.defense
    assert experiment.defense.defenses[0].defense_name == (
        "art.defences.preprocessor.FeatureSqueezing"
    )


@pytest.mark.parametrize(
    "defenses,expected_preprocessors,expected_postprocessors",
    [
        (
            [
                _example_defense(
                    "examples/sklearn/config/defense/feature-squeezing.yaml",
                ),
                _example_defense(
                    "examples/sklearn/config/defense/gaussian-augmentation.yaml",
                    overrides={
                        "apply_fit": False,
                        "apply_predict": True,
                        "sigma": 0.1,
                        "ratio": 1.0,
                        "augmentation": False,
                    },
                ),
            ],
            ["FeatureSqueezing", "GaussianAugmentation"],
            [],
        ),
        (
            [
                _example_defense(
                    "examples/sklearn/config/defense/gaussian-noise.yaml",
                ),
                _example_defense(
                    "examples/sklearn/config/defense/class-labels.yaml",
                ),
            ],
            [],
            ["GaussianNoise", "ClassLabels"],
        ),
        (
            [
                _example_defense(
                    "examples/sklearn/config/defense/feature-squeezing.yaml",
                ),
                _example_defense(
                    "examples/sklearn/config/defense/gaussian-noise.yaml",
                ),
            ],
            ["FeatureSqueezing"],
            ["GaussianNoise"],
        ),
    ],
)
def test_defense_pipeline_applies_expected_art_wrapper_defenses(
    defenses,
    expected_preprocessors,
    expected_postprocessors,
):
    _, model, wrapped = _apply_defense({"defenses": defenses})

    assert len(model.defense.defenses) == len(defenses)
    assert all(
        defense.defense_application_time is not None
        for defense in model.defense.defenses
    )
    assert model.defense_application_time is not None
    assert wrapped.model.__class__.__name__ == "LogisticRegression"
    _assert_wrapper_defenses(
        wrapped,
        expected_preprocessors,
        expected_postprocessors,
    )


def test_get_art_model_does_not_duplicate_apply_fit_preprocessor_defense():
    data = _base_classification_data()
    model = _base_model(
        defense={
            "defenses": [
                _example_defense(
                    "examples/sklearn/config/defense/gaussian-augmentation.yaml",
                ),
            ],
        },
    )
    model.train(data.X_train, data.y_train)

    first_wrapper = model.get_art_model(data)
    second_wrapper = model.get_art_model(data)

    assert _wrapper_defense_names(first_wrapper, "preprocessing_defences") == [
        "GaussianAugmentation",
    ]
    assert _wrapper_defense_names(second_wrapper, "preprocessing_defences") == [
        "GaussianAugmentation",
    ]
    assert getattr(second_wrapper, "_deckard_applied_defense_signatures", None)


# ---------------------------------------------------------------------------
# Hash stability and persistence
# ---------------------------------------------------------------------------


def test_defense_pipeline_config_hash_stable_after_apply():
    data, model, _ = _apply_defense(
        {
            "defenses": [
                _example_defense(
                    "examples/sklearn/config/defense/feature-squeezing.yaml",
                ),
            ],
        },
    )

    original_hash = hash(model.defense)
    # Simulate runtime mutations that should not affect identity.
    model.defense.defense_application_time = 99.9
    model.defense.score_dict["runtime"] = 1
    assert hash(model.defense) == original_hash


def test_model_config_with_defense_hash_stable_after_apply():
    data, model, _ = _apply_defense(
        str(ROOT / "examples/sklearn/config/defense/gaussian-noise.yaml"),
    )
    original_hash = hash(model)
    model.defense_application_time = 99.9
    model.score_dict["extra"] = 1
    assert hash(model) == original_hash


def test_defense_pipeline_scores_persist_and_reload():
    data, model, _ = _apply_defense(
        {
            "defenses": [
                _example_defense(
                    "examples/sklearn/config/defense/feature-squeezing.yaml",
                ),
                _example_defense(
                    "examples/sklearn/config/defense/gaussian-noise.yaml",
                ),
            ],
        },
    )
    scores = {
        "defense_application_time": model.defense_application_time,
        "n_defenses": 2,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "defense_scores.json"
        model.save_scores(scores, path)
        loaded = model.load_scores(str(path))
    assert "n_defenses" in loaded
    assert loaded["n_defenses"] == 2


def test_defense_pipeline_config_object_pickle_roundtrip():
    pipeline = DefenseConfig(
        defenses=[
            _example_defense(
                "examples/sklearn/config/defense/feature-squeezing.yaml",
            ),
            _example_defense(
                "examples/sklearn/config/defense/gaussian-noise.yaml",
            ),
        ],
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pipeline.pkl"
        pipeline.save_object(pipeline, str(path))
        loaded = pipeline.load_object(str(path))
    assert isinstance(loaded, DefenseConfig)
    assert len(loaded.defenses) == 2
