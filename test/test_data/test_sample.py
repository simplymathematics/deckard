"""Tests for deckard/data/sample.py and the sampler integration in DataConfig."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
from helpers import load_canonical_data_profile

from deckard.data import (
    BaseSampler,
    DataConfig,
    KFoldSampler,
    ShuffleSampler,
    SplitSampler,
)
import pytest


def _make_clf_config(**kwargs):
    """Return a small classification DataConfig, loading data but not yet sampling."""
    defaults = load_canonical_data_profile("classification", framework="sklearn")
    defaults["data_params"].update(
        {
            "n_samples": 120,
            "n_features": 5,
            "n_informative": 3,
            "n_redundant": 0,
            "random_state": 0,
            "n_clusters_per_class": 1,
        },
    )
    defaults.update({"classifier": True})
    defaults.update(kwargs)
    if "sample" in defaults and "sampler" not in defaults:
        defaults["sampler"] = defaults.pop("sample")
    sampler_train_size = defaults.pop("train_size", None)
    sampler_test_size = defaults.pop("test_size", 0.2)
    sampler_val_size = defaults.pop("val_size", None)
    sampler_random_state = defaults.pop("random_state", 42)
    sampler_stratify = defaults.pop("stratify", True)
    defaults.pop("n_splits", None)
    defaults.pop("shuffle", None)
    defaults.pop("split", None)
    if "sampler" not in kwargs and "sample" not in kwargs:
        defaults["sampler"] = SplitSampler(
            train_size=sampler_train_size,
            test_size=sampler_test_size,
            val_size=sampler_val_size,
            random_state=sampler_random_state,
            stratify=sampler_stratify,
        )
    cfg = DataConfig(**defaults)
    cfg.load_dataset()
    return cfg


def _make_reg_config(**kwargs):
    """Return a small regression DataConfig."""
    defaults = load_canonical_data_profile("regression", framework="sklearn")
    defaults["data_params"].update(
        {
            "n_samples": 100,
            "n_features": 4,
            "n_informative": 2,
            "random_state": 1,
        },
    )
    defaults.update({"classifier": False})
    defaults.update(kwargs)
    if "sample" in defaults and "sampler" not in defaults:
        defaults["sampler"] = defaults.pop("sample")
    sampler_train_size = defaults.pop("train_size", None)
    sampler_test_size = defaults.pop("test_size", 0.2)
    sampler_val_size = defaults.pop("val_size", None)
    sampler_random_state = defaults.pop("random_state", 1)
    sampler_stratify = defaults.pop("stratify", False)
    defaults.pop("n_splits", None)
    defaults.pop("shuffle", None)
    defaults.pop("split", None)
    if "sampler" not in kwargs and "sample" not in kwargs:
        defaults["sampler"] = SplitSampler(
            train_size=sampler_train_size,
            test_size=sampler_test_size,
            val_size=sampler_val_size,
            random_state=sampler_random_state,
            stratify=sampler_stratify,
        )
    cfg = DataConfig(**defaults)
    cfg.load_dataset()
    return cfg


# ---------------------------------------------------------------------------
# BaseSampler
# ---------------------------------------------------------------------------


class TestBaseSampler:
    def test_base_sampler_raises(self):
        sampler = BaseSampler()
        with pytest.raises(NotImplementedError):
            sampler(None)


# ---------------------------------------------------------------------------
# _get_stratify_col
# ---------------------------------------------------------------------------


class TestGetStratifyCol:
    def test_stratify_true_returns_y(self):
        cfg = _make_clf_config()
        col = BaseSampler._get_stratify_col(cfg)
        assert isinstance(col, pd.Series)
        assert len(col) == len(cfg._y)

    def test_stratify_false_returns_none(self):
        cfg = _make_clf_config(stratify=False)
        assert BaseSampler._get_stratify_col(cfg) is None

    def test_stratify_none_returns_none(self):
        cfg = _make_clf_config(stratify=None)
        assert BaseSampler._get_stratify_col(cfg) is None

    def test_stratify_column_name(self):
        cfg = _make_clf_config(stratify=False)
        # Add a column to _X so we can use it
        cfg._X["strat_col"] = np.tile([0, 1], len(cfg._X) // 2 + 1)[: len(cfg._X)]
        cfg.sampler.stratify = "strat_col"
        col = BaseSampler._get_stratify_col(cfg)
        assert isinstance(col, pd.Series)
        assert len(col) == len(cfg._X)

    def test_stratify_invalid_column_raises(self):
        cfg = _make_clf_config(stratify=False)
        cfg.sampler.stratify = "nonexistent_column"
        with pytest.raises(ValueError):
            BaseSampler._get_stratify_col(cfg)

    def test_stratify_invalid_type_raises(self):
        cfg = _make_clf_config(stratify=False)
        cfg.sampler.stratify = 42
        with pytest.raises(ValueError):
            BaseSampler._get_stratify_col(cfg)


# ---------------------------------------------------------------------------
# SplitSampler
# ---------------------------------------------------------------------------


class TestSplitSampler:
    def setup_method(self):
        self.cfg = _make_clf_config(val_size=0.15)

    def test_returns_three_arrays(self):
        sampler = SplitSampler(
            test_size=0.2, val_size=0.15, random_state=42, stratify=True
        )
        result = sampler(self.cfg)
        assert len(result) == 3

    def test_indices_are_disjoint(self):
        sampler = SplitSampler(
            test_size=0.2, val_size=0.15, random_state=42, stratify=True
        )
        train, test, val = sampler(self.cfg)
        assert len(set(train) & set(test)) == 0
        assert len(set(train) & set(val)) == 0
        assert len(set(test) & set(val)) == 0

    def test_total_covers_dataset(self):
        sampler = SplitSampler(
            test_size=0.2, val_size=0.15, random_state=42, stratify=True
        )
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        assert total == len(self.cfg._X)

    def test_two_way_split_without_val_size(self):
        cfg = _make_clf_config()
        cfg.val_size = None
        sampler = SplitSampler(
            test_size=0.2, val_size=None, random_state=42, stratify=True
        )
        train, test, val = sampler(cfg)
        assert len(val) == 0
        assert len(train) > 0
        assert len(test) > 0

    def test_stratified_class_distribution(self):
        sampler = SplitSampler(
            test_size=0.2, val_size=0.15, random_state=42, stratify=True
        )
        train, test, val = sampler(self.cfg)
        y = self.cfg._y
        for idx, name in [(train, "train"), (test, "test"), (val, "val")]:
            props = y.iloc[idx].value_counts(normalize=True)
            overall = y.value_counts(normalize=True)
            for cls in overall.index:
                assert (
                    abs(props.get(cls, 0.0) - overall[cls]) < 0.1
                ), f"Class {cls} proportion off in {name}"

    def test_integration_with_dataconfig(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 200,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 7,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler=SplitSampler(
                test_size=0.2, val_size=0.1, random_state=42, stratify=True
            ),
        )
        cfg()
        assert cfg.X_val is not None
        assert cfg.y_val is not None
        assert isinstance(cfg.X_val, pd.DataFrame)
        assert isinstance(cfg.y_val, pd.Series)
        assert cfg.val_n is not None
        assert cfg.val_n > 0

    def test_score_dict_contains_val_fields(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 200,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 7,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler=SplitSampler(
                test_size=0.2, val_size=0.1, random_state=42, stratify=True
            ),
        )
        scores = cfg()
        assert "val_n" in scores

    def test_regression_val_score(self):
        cfg = DataConfig(
            dataset_name="make_regression",
            data_params={
                "n_samples": 100,
                "n_features": 4,
                "n_informative": 2,
                "random_state": 1,
            },
            classifier=False,
            sampler=SplitSampler(
                test_size=0.2, val_size=0.1, random_state=1, stratify=False
            ),
        )
        scores = cfg()
        assert "val_n" in scores
        assert "mutual_information_mean" in scores["test"]

    def test_sample_dict_spec(self):
        """DataConfig should accept a dict sampler spec and instantiate it."""
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 150,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 9,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "test_size": 0.2,
                "val_size": 0.1,
                "random_state": 42,
                "stratify": True,
            },
        )
        cfg()
        assert cfg.X_val is not None

    def test_regression_no_stratify(self):
        cfg = _make_reg_config(val_size=0.15)
        sampler = SplitSampler(
            test_size=0.2, val_size=0.15, random_state=1, stratify=False
        )
        train, test, val = sampler(cfg)
        assert len(set(train) & set(test)) == 0
        assert len(set(train) & set(val)) == 0


# ---------------------------------------------------------------------------
# KFoldSampler
# ---------------------------------------------------------------------------


class TestKFoldSampler:
    def setup_method(self):
        self.cfg = _make_clf_config()

    def test_returns_three_arrays(self):
        sampler = KFoldSampler(
            n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True
        )
        result = sampler(self.cfg)
        assert len(result) == 3

    def test_indices_are_disjoint(self):
        sampler = KFoldSampler(
            n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True
        )
        train, test, val = sampler(self.cfg)
        assert len(set(train) & set(test)) == 0
        assert len(set(train) & set(val)) == 0
        assert len(set(test) & set(val)) == 0

    def test_total_covers_dataset(self):
        sampler = KFoldSampler(
            n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True
        )
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        assert total == len(self.cfg._X)

    def test_different_folds_have_different_val_sets(self):
        results = []
        for fold in range(3):
            cfg = _make_clf_config()
            sampler = KFoldSampler(
                n_splits=5, split=fold, test_size=0.2, random_state=42, stratify=True
            )
            _, _, val = sampler(cfg)
            results.append(set(val))
        # Val sets across different folds must not all be identical
        assert not (results[0] == results[1] == results[2])

    def test_fold_out_of_range_raises(self):
        sampler = KFoldSampler(
            n_splits=5, split=99, test_size=0.2, random_state=42, stratify=True
        )
        with pytest.raises(ValueError):
            sampler(self.cfg)

    def test_no_stratify(self):
        cfg = _make_clf_config(stratify=False)
        sampler = KFoldSampler(
            n_splits=5, split=0, test_size=0.2, shuffle=False, stratify=False
        )
        train, test, val = sampler(cfg)
        assert len(train) + len(test) + len(val) == len(cfg._X)

    def test_integration_with_dataconfig(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 200,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 7,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler=KFoldSampler(
                n_splits=5, split=1, test_size=0.2, random_state=42, stratify=True
            ),
        )
        cfg()
        assert cfg.X_val is not None
        assert cfg.y_val is not None
        assert cfg.val_n is not None
        assert cfg.val_n > 0

    def test_fold_none_defaults_to_zero(self):
        cfg = _make_clf_config()
        cfg_split0 = _make_clf_config()
        sampler = KFoldSampler(
            n_splits=5, split=None, test_size=0.2, random_state=42, stratify=True
        )
        sampler_split0 = KFoldSampler(
            n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True
        )
        _, _, val_none = sampler(cfg)
        _, _, val_zero = sampler_split0(cfg_split0)
        assert sorted(val_none) == sorted(val_zero)

    def test_caps_produce_expected_sizes_for_1200_samples(self):
        """Lock cap semantics: val_size caps val fold, train_size caps train+test pool."""
        with patch.dict(os.environ, {"DECKARD_TEST_MAX_SAMPLES": ""}):
            for split in range(5):
                cfg = DataConfig(
                    dataset_name="make_classification",
                    data_params={
                        "n_samples": 1200,
                        "n_features": 20,
                        "n_informative": 10,
                        "n_redundant": 5,
                        "random_state": 42,
                        "n_clusters_per_class": 1,
                    },
                    classifier=True,
                    sampler={
                        "name": "deckard.data.sample.KFoldSampler",
                        "n_splits": 5,
                        "split": split,
                        "train_size": 1000,
                        "test_size": 200,
                        "val_size": 200,
                        "random_state": 42,
                        "stratify": True,
                    },
                )
                cfg.load_dataset()
                cfg.fit()
                assert len(cfg.X_train) == 800
                assert len(cfg.X_test) == 200
                assert len(cfg.X_val) == 200

    def test_integer_test_size_guardrail_raises(self):
        """test_size must be <= train_size // n_splits for integer sizing."""
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 1200,
                "n_features": 10,
                "n_informative": 5,
                "n_redundant": 2,
                "random_state": 1,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler={
                "name": "deckard.data.sample.KFoldSampler",
                "n_splits": 5,
                "split": 0,
                "train_size": 1000,
                "test_size": 201,
                "val_size": 200,
                "random_state": 42,
                "stratify": True,
            },
        )
        cfg.load_dataset()
        with pytest.raises(ValueError):
            cfg.fit()


# ---------------------------------------------------------------------------
# ShuffleSampler
# ---------------------------------------------------------------------------


class TestShuffleSampler:
    def setup_method(self):
        self.cfg = _make_clf_config(val_size=0.15)

    def test_returns_three_arrays(self):
        sampler = ShuffleSampler(
            n_splits=5,
            split=0,
            test_size=0.2,
            val_size=0.15,
            random_state=42,
            stratify=True,
        )
        result = sampler(self.cfg)
        assert len(result) == 3

    def test_indices_are_disjoint_within_fold(self):
        sampler = ShuffleSampler(
            n_splits=5,
            split=0,
            test_size=0.2,
            val_size=0.15,
            random_state=42,
            stratify=True,
        )
        train, test, val = sampler(self.cfg)
        assert len(set(train) & set(test)) == 0
        assert len(set(train) & set(val)) == 0
        assert len(set(test) & set(val)) == 0

    def test_total_covers_dataset(self):
        sampler = ShuffleSampler(
            n_splits=5,
            split=0,
            test_size=0.2,
            val_size=0.15,
            random_state=42,
            stratify=True,
        )
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        assert total == len(self.cfg._X)

    def test_raises_without_val_size(self):
        cfg = _make_clf_config()
        sampler = ShuffleSampler(
            n_splits=5,
            split=0,
            test_size=0.2,
            val_size=None,
            random_state=42,
            stratify=True,
        )
        with pytest.raises(ValueError):
            sampler(cfg)

    def test_fold_out_of_range_raises(self):
        sampler = ShuffleSampler(
            n_splits=5,
            split=99,
            test_size=0.2,
            val_size=0.15,
            random_state=42,
            stratify=True,
        )
        with pytest.raises(ValueError):
            sampler(self.cfg)

    def test_integration_with_dataconfig(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 200,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 7,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler=ShuffleSampler(
                n_splits=5,
                split=2,
                test_size=0.2,
                val_size=0.15,
                random_state=42,
                stratify=True,
            ),
        )
        cfg()
        assert cfg.X_val is not None
        assert cfg.y_val is not None
        assert cfg.val_n is not None
        assert cfg.val_n > 0

    def test_no_stratify(self):
        cfg = _make_reg_config(val_size=0.15)
        sampler = ShuffleSampler(
            n_splits=5,
            split=0,
            test_size=0.2,
            val_size=0.15,
            random_state=1,
            stratify=False,
        )
        train, test, val = sampler(cfg)
        assert len(train) + len(test) + len(val) == len(cfg._X)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2-way split (sampler="split", no val_size)
# ---------------------------------------------------------------------------


class TestLegacySplitUnchanged:
    def test_no_val_set_when_no_val_size(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 100,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 0,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler="split",
        )
        cfg()
        assert cfg.X_val is None
        assert cfg.y_val is None
        assert len(cfg.val_indices) == 0
        assert len(cfg.X_train) + len(cfg.X_test) == 100

    def test_score_dict_no_val_fields_without_val_size(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 100,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 0,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler="split",
        )
        scores = cfg()
        assert "val_n" not in scores
        assert "val_class_counts" not in scores

    def test_val_size_none_gives_two_way_split(self):
        """sampler='split' with val_size=None should not create a val set."""
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 100,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 0,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler="split",
        )
        cfg()
        assert cfg.X_val is None
        assert cfg.y_val is None
        assert len(cfg.X_train) + len(cfg.X_test) == 100


# ---------------------------------------------------------------------------
# OmegaConf DictConfig sampler spec
# ---------------------------------------------------------------------------


class TestOmegaConfSampleSpec:
    def test_omegaconf_dictconfig_sample(self):
        """_resolve_sample should handle an OmegaConf DictConfig spec."""
        from omegaconf import OmegaConf

        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 150,
                "n_features": 5,
                "n_informative": 3,
                "n_redundant": 0,
                "random_state": 5,
                "n_clusters_per_class": 1,
            },
            classifier=True,
            sampler=SplitSampler(
                test_size=0.2, val_size=0.1, random_state=42, stratify=True
            ),
        )
        # Simulate Hydra passing an OmegaConf DictConfig for the sampler
        cfg.sampler = OmegaConf.create(
            {
                "name": "deckard.data.sample.SplitSampler",
                "test_size": 0.2,
                "val_size": 0.1,
                "random_state": 42,
                "stratify": True,
            },
        )
        cfg.load_dataset()
        cfg.fit()
        assert cfg.X_val is not None
        assert len(cfg.X_val) > 0


# ---------------------------------------------------------------------------
# Hydra ConfigStore registration
# ---------------------------------------------------------------------------


class TestConfigStoreRegistration:
    def test_register_sampler_configs_runs_without_error(self):
        from deckard.data.sample import register_sampler_configs

        # Should not raise even when called multiple times
        register_sampler_configs()
        register_sampler_configs()

    def test_configstore_has_expected_groups(self):
        from hydra.core.config_store import ConfigStore

        from deckard.data.sample import register_sampler_configs

        register_sampler_configs()
        cs = ConfigStore.instance()
        # Verify that our entries are present under the 'sample' group.
        # cs.list() returns a list of config names in the given group.
        listed_names = set(cs.list("sample"))
        for expected in (
            "split.yaml",
            "kfold.yaml",
            "shuffle.yaml",
            "none.yaml",
        ):
            assert (
                expected in listed_names
            ), f"Expected '{expected}' in ConfigStore sample group, got: {listed_names}"
