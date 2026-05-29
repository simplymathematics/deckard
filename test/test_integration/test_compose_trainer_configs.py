"""Compose tests for canonical trainer config groups.

These tests validate trainer group composition into model.trainer and parity
between canonical and backend-prefixed trainer aliases.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

SKLEARN_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
)
PYTORCH_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "config"
)


def _reset_hydra_state():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)


def _compose_sklearn(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    _reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(SKLEARN_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _compose_pytorch(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    _reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(PYTORCH_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


@pytest.mark.parametrize(
    "overrides,expected_target",
    [
        pytest.param([], "deckard.model.trainers.SklearnTrainer", id="default"),
        pytest.param(
            ["trainer@model.trainer=pretrained"],
            "deckard.model.trainers.PretrainedTrainer",
            id="pretrained",
        ),
        pytest.param(
            ["trainer@model.trainer=partial_fit"],
            "deckard.model.trainers.PartialFitTrainer",
            id="partial-fit",
        ),
        pytest.param(
            ["trainer@model.trainer=partial_fit_pruning"],
            "deckard.model.trainers.PartialFitPruningTrainer",
            id="partial-fit-pruning",
        ),
        pytest.param(
            ["trainer@model.trainer=pruning"],
            "deckard.model.trainers.PruningTrainer",
            id="pruning",
        ),
        pytest.param(
            ["trainer@model.trainer=sklearn-pretrained"],
            "deckard.model.trainers.PretrainedTrainer",
            id="sklearn-pretrained-alias",
        ),
        pytest.param(
            ["trainer@model.trainer=sklearn-partial_fit"],
            "deckard.model.trainers.PartialFitTrainer",
            id="sklearn-partial-fit-alias",
        ),
        pytest.param(
            ["trainer@model.trainer=sklearn-partial_fit_pruning"],
            "deckard.model.trainers.PartialFitPruningTrainer",
            id="sklearn-partial-fit-pruning-alias",
        ),
        pytest.param(
            ["trainer@model.trainer=sklearn-pruning"],
            "deckard.model.trainers.PruningTrainer",
            id="sklearn-pruning-alias",
        ),
    ],
)
def test_sklearn_trainer_group_composes_to_model_trainer(overrides, expected_target):
    cfg = _compose_sklearn("default", overrides=overrides)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert model_cfg["trainer"]["_target_"] == expected_target


@pytest.mark.parametrize(
    "overrides,expected_target",
    [
        pytest.param([], "deckard.model.trainers.PytorchTrainer", id="default"),
        pytest.param(
            ["trainer@model.trainer=pretrained"],
            "deckard.model.trainers.PretrainedTrainer",
            id="pretrained",
        ),
        pytest.param(
            ["trainer@model.trainer=partial_fit"],
            "deckard.model.trainers.PartialFitTrainer",
            id="partial-fit",
        ),
        pytest.param(
            ["trainer@model.trainer=partial_fit_pruning"],
            "deckard.model.trainers.PartialFitPruningTrainer",
            id="partial-fit-pruning",
        ),
        pytest.param(
            ["trainer@model.trainer=pruning"],
            "deckard.model.trainers.PruningTrainer",
            id="pruning",
        ),
        pytest.param(
            ["trainer@model.trainer=pytorch-pretrained"],
            "deckard.model.trainers.PretrainedTrainer",
            id="pytorch-pretrained-alias",
        ),
        pytest.param(
            ["trainer@model.trainer=pytorch-partial_fit"],
            "deckard.model.trainers.PartialFitTrainer",
            id="pytorch-partial-fit-alias",
        ),
        pytest.param(
            ["trainer@model.trainer=pytorch-partial_fit_pruning"],
            "deckard.model.trainers.PartialFitPruningTrainer",
            id="pytorch-partial-fit-pruning-alias",
        ),
        pytest.param(
            ["trainer@model.trainer=pytorch-pruning"],
            "deckard.model.trainers.PruningTrainer",
            id="pytorch-pruning-alias",
        ),
    ],
)
def test_pytorch_trainer_group_composes_to_model_trainer(overrides, expected_target):
    cfg = _compose_pytorch("torch_default", overrides=overrides)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert model_cfg["trainer"]["_target_"] == expected_target


@pytest.mark.parametrize(
    "canonical,backend_prefixed",
    [
        pytest.param("pretrained", "sklearn-pretrained", id="sklearn-pretrained"),
        pytest.param("partial_fit", "sklearn-partial_fit", id="sklearn-partial-fit"),
        pytest.param(
            "partial_fit_pruning",
            "sklearn-partial_fit_pruning",
            id="sklearn-partial-fit-pruning",
        ),
        pytest.param("pruning", "sklearn-pruning", id="sklearn-pruning"),
    ],
)
def test_sklearn_backend_prefixed_alias_parity(canonical, backend_prefixed):
    canonical_cfg = _compose_sklearn(
        "default",
        overrides=[f"trainer@model.trainer={canonical}"],
    )
    alias_cfg = _compose_sklearn(
        "default",
        overrides=[f"trainer@model.trainer={backend_prefixed}"],
    )
    canonical_trainer = OmegaConf.to_container(canonical_cfg.model.trainer, resolve=True)
    alias_trainer = OmegaConf.to_container(alias_cfg.model.trainer, resolve=True)
    assert canonical_trainer == alias_trainer


@pytest.mark.parametrize(
    "canonical,backend_prefixed",
    [
        pytest.param("pretrained", "pytorch-pretrained", id="pytorch-pretrained"),
        pytest.param("partial_fit", "pytorch-partial_fit", id="pytorch-partial-fit"),
        pytest.param(
            "partial_fit_pruning",
            "pytorch-partial_fit_pruning",
            id="pytorch-partial-fit-pruning",
        ),
        pytest.param("pruning", "pytorch-pruning", id="pytorch-pruning"),
    ],
)
def test_pytorch_backend_prefixed_alias_parity(canonical, backend_prefixed):
    canonical_cfg = _compose_pytorch(
        "torch_default",
        overrides=[f"trainer@model.trainer={canonical}"],
    )
    alias_cfg = _compose_pytorch(
        "torch_default",
        overrides=[f"trainer@model.trainer={backend_prefixed}"],
    )
    canonical_trainer = OmegaConf.to_container(canonical_cfg.model.trainer, resolve=True)
    alias_trainer = OmegaConf.to_container(alias_cfg.model.trainer, resolve=True)
    assert canonical_trainer == alias_trainer
