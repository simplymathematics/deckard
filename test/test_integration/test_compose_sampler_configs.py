"""Compose tests for canonical sampler config groups.

These tests keep runtime low by using one representative non-default sampler
path per framework instead of a cross-product over every alias and sampler.
They still validate group composition into ``data.sampler`` plus parity between
canonical and backend-prefixed aliases, including mirrored search entries.
"""

from pathlib import Path

import yaml

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

SKLEARN_SYNTHETIC_OVERRIDES = ["data=test-classification"]


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


def test_sklearn_sampler_group_composes_fold_on_small_synthetic_data():
    cfg = _compose_sklearn(
        "default",
        overrides=SKLEARN_SYNTHETIC_OVERRIDES + ["sampler@data.sampler=fold"],
    )
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)
    assert isinstance(data_cfg.get("sampler"), dict)
    assert data_cfg["name"] == "make_classification"
    assert data_cfg["data_params"]["n_samples"] == 80
    assert data_cfg["sampler"]["name"] == "deckard.data.sample.KFoldSampler"


def test_pytorch_sampler_group_composes_fold_to_data_sampler():
    cfg = _compose_pytorch("torch_default", overrides=["sampler@data.sampler=fold"])
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)
    assert isinstance(data_cfg.get("sampler"), dict)
    assert (
        data_cfg["sampler"]["name"]
        == "deckard.frameworks.pytorch.sample.PytorchFoldSampler"
    )


def test_sklearn_backend_prefixed_sampler_alias_parity():
    canonical_cfg = _compose_sklearn(
        "default",
        overrides=SKLEARN_SYNTHETIC_OVERRIDES + ["sampler@data.sampler=fold"],
    )
    alias_cfg = _compose_sklearn(
        "default",
        overrides=SKLEARN_SYNTHETIC_OVERRIDES + ["sampler@data.sampler=sklearn-fold"],
    )
    canonical_sampler = OmegaConf.to_container(
        canonical_cfg.data.sampler, resolve=True
    )
    alias_sampler = OmegaConf.to_container(alias_cfg.data.sampler, resolve=True)
    assert canonical_sampler == alias_sampler


def test_pytorch_backend_prefixed_sampler_alias_parity():
    canonical_cfg = _compose_pytorch(
        "torch_default",
        overrides=["sampler@data.sampler=fold"],
    )
    alias_cfg = _compose_pytorch(
        "torch_default",
        overrides=["sampler@data.sampler=pytorch-fold"],
    )
    canonical_sampler = OmegaConf.to_container(
        canonical_cfg.data.sampler, resolve=True
    )
    alias_sampler = OmegaConf.to_container(alias_cfg.data.sampler, resolve=True)
    assert canonical_sampler == alias_sampler


def test_sklearn_search_sampler_entry_matches_backend_alias():
    canonical_path = SKLEARN_CONFIG_DIR / "search" / "samplers" / "fold.yaml"
    alias_path = SKLEARN_CONFIG_DIR / "search" / "samplers" / "sklearn-fold.yaml"

    assert yaml.safe_load(
        canonical_path.read_text(encoding="utf-8")
    ) == yaml.safe_load(
        alias_path.read_text(encoding="utf-8"),
    )


def test_pytorch_search_sampler_entry_matches_backend_alias():
    canonical_path = PYTORCH_CONFIG_DIR / "search" / "samplers" / "fold.yaml"
    alias_path = PYTORCH_CONFIG_DIR / "search" / "samplers" / "pytorch-fold.yaml"

    assert yaml.safe_load(
        canonical_path.read_text(encoding="utf-8")
    ) == yaml.safe_load(
        alias_path.read_text(encoding="utf-8"),
    )
