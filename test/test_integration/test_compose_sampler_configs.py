"""Compose tests for canonical sampler config groups.

These tests keep runtime low by using one representative non-default sampler
path per framework instead of a cross-product over every alias and sampler.
They still validate group composition into ``data.sampler`` plus parity between
canonical and backend-prefixed aliases, including mirrored search entries.
"""

import yaml
from omegaconf import OmegaConf

from .shared_compose import (
    PYTORCH_CONFIG_DIR,
    SKLEARN_CONFIG_DIR,
    compose_pytorch,
    compose_sklearn,
)

SKLEARN_SYNTHETIC_OVERRIDES = ["data=test-classification"]


def test_sklearn_sampler_group_composes_fold_on_small_synthetic_data():
    cfg = compose_sklearn(
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
    cfg = compose_pytorch("torch_default", overrides=["sampler@data.sampler=fold"])
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)
    assert isinstance(data_cfg.get("sampler"), dict)
    assert (
        data_cfg["sampler"]["name"]
        == "deckard.frameworks.pytorch.sample.PytorchFoldSampler"
    )


def test_sklearn_backend_prefixed_sampler_alias_parity():
    canonical_cfg = compose_sklearn(
        "default",
        overrides=SKLEARN_SYNTHETIC_OVERRIDES + ["sampler@data.sampler=fold"],
    )
    alias_cfg = compose_sklearn(
        "default",
        overrides=SKLEARN_SYNTHETIC_OVERRIDES + ["sampler@data.sampler=sklearn-fold"],
    )
    canonical_sampler = OmegaConf.to_container(
        canonical_cfg.data.sampler,
        resolve=True,
    )
    alias_sampler = OmegaConf.to_container(alias_cfg.data.sampler, resolve=True)
    assert canonical_sampler == alias_sampler


def test_pytorch_backend_prefixed_sampler_alias_parity():
    canonical_cfg = compose_pytorch(
        "torch_default",
        overrides=["sampler@data.sampler=fold"],
    )
    alias_cfg = compose_pytorch(
        "torch_default",
        overrides=["sampler@data.sampler=pytorch-fold"],
    )
    canonical_sampler = OmegaConf.to_container(
        canonical_cfg.data.sampler,
        resolve=True,
    )
    alias_sampler = OmegaConf.to_container(alias_cfg.data.sampler, resolve=True)
    assert canonical_sampler == alias_sampler


def test_sklearn_search_sampler_entry_matches_backend_alias():
    canonical_path = SKLEARN_CONFIG_DIR / "search" / "samplers" / "fold.yaml"
    alias_path = SKLEARN_CONFIG_DIR / "search" / "samplers" / "sklearn-fold.yaml"

    assert yaml.safe_load(
        canonical_path.read_text(encoding="utf-8"),
    ) == yaml.safe_load(
        alias_path.read_text(encoding="utf-8"),
    )


def test_pytorch_search_sampler_entry_matches_backend_alias():
    canonical_path = PYTORCH_CONFIG_DIR / "search" / "samplers" / "fold.yaml"
    alias_path = PYTORCH_CONFIG_DIR / "search" / "samplers" / "pytorch-fold.yaml"

    assert yaml.safe_load(
        canonical_path.read_text(encoding="utf-8"),
    ) == yaml.safe_load(
        alias_path.read_text(encoding="utf-8"),
    )
