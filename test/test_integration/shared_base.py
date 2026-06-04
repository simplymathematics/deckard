from __future__ import annotations

from deckard.data import DataConfig
from deckard.model import ModelConfig


def make_base_classification_data(
    *,
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_classes: int,
    random_state: int,
    sampler_name: str,
    train_size: int,
    test_size: int,
    stratify: bool | str | None = True,
) -> DataConfig:
    cfg = DataConfig(
        name="make_classification",
        data_params={
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "n_redundant": 0,
            "n_clusters_per_class": 1,
            "n_classes": n_classes,
            "random_state": random_state,
        },
        classifier=True,
        sampler={
            "name": sampler_name,
            "train_size": train_size,
            "test_size": test_size,
            "random_state": 42,
            "stratify": stratify,
        },
    )
    cfg()
    return cfg


def make_logistic_model(*, defense=None, max_iter: int = 25) -> ModelConfig:
    return ModelConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": max_iter},
        defense=defense,
    )
