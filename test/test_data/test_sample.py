"""Tests for deckard/data/sample.py and the sampler integration in DataConfig."""

import os
import unittest
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    defaults.update(
        {"test_size": 0.2, "random_state": 42, "stratify": True, "classifier": True},
    )
    defaults.update(kwargs)
    if "sample" in defaults and "sampler" not in defaults:
        defaults["sampler"] = defaults.pop("sample")
    defaults.setdefault(
        "sampler",
        SplitSampler(
            train_size=defaults.get("train_size", None),
            test_size=defaults.get("test_size", 0.2),
            val_size=defaults.get("val_size", None),
            random_state=defaults.get("random_state", 42),
            stratify=defaults.get("stratify", True),
        ),
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
    defaults.update(
        {"test_size": 0.2, "random_state": 1, "stratify": False, "classifier": False},
    )
    defaults.update(kwargs)
    if "sample" in defaults and "sampler" not in defaults:
        defaults["sampler"] = defaults.pop("sample")
    defaults.setdefault(
        "sampler",
        SplitSampler(
            train_size=defaults.get("train_size", None),
            test_size=defaults.get("test_size", 0.2),
            val_size=defaults.get("val_size", None),
            random_state=defaults.get("random_state", 1),
            stratify=defaults.get("stratify", False),
        ),
    )
    cfg = DataConfig(**defaults)
    cfg.load_dataset()
    return cfg


# ---------------------------------------------------------------------------
# BaseSampler
# ---------------------------------------------------------------------------


class TestBaseSampler(unittest.TestCase):
    def test_base_sampler_raises(self):
        sampler = BaseSampler()
        with self.assertRaises(NotImplementedError):
            sampler(None)


# ---------------------------------------------------------------------------
# _get_stratify_col
# ---------------------------------------------------------------------------


class TestGetStratifyCol(unittest.TestCase):
    def test_stratify_true_returns_y(self):
        cfg = _make_clf_config()
        col = cfg._get_stratify_col()
        self.assertIsInstance(col, pd.Series)
        self.assertEqual(len(col), len(cfg._y))

    def test_stratify_false_returns_none(self):
        cfg = _make_clf_config(stratify=False)
        self.assertIsNone(cfg._get_stratify_col())

    def test_stratify_none_returns_none(self):
        cfg = _make_clf_config(stratify=None)
        self.assertIsNone(cfg._get_stratify_col())

    def test_stratify_column_name(self):
        cfg = _make_clf_config(stratify=False)
        # Add a column to _X so we can use it
        cfg._X["strat_col"] = np.tile([0, 1], len(cfg._X) // 2 + 1)[: len(cfg._X)]
        cfg.stratify = "strat_col"
        col = cfg._get_stratify_col()
        self.assertIsInstance(col, pd.Series)
        self.assertEqual(len(col), len(cfg._X))

    def test_stratify_invalid_column_raises(self):
        cfg = _make_clf_config(stratify=False)
        cfg.stratify = "nonexistent_column"
        with self.assertRaises(ValueError):
            cfg._get_stratify_col()

    def test_stratify_invalid_type_raises(self):
        cfg = _make_clf_config(stratify=False)
        cfg.stratify = 42
        with self.assertRaises(ValueError):
            cfg._get_stratify_col()


# ---------------------------------------------------------------------------
# SplitSampler
# ---------------------------------------------------------------------------


class TestSplitSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_clf_config(val_size=0.15)

    def test_returns_three_arrays(self):
        sampler = SplitSampler(test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        result = sampler(self.cfg)
        self.assertEqual(len(result), 3)

    def test_indices_are_disjoint(self):
        sampler = SplitSampler(test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(test) & set(val)), 0)

    def test_total_covers_dataset(self):
        sampler = SplitSampler(test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        self.assertEqual(total, len(self.cfg._X))

    def test_two_way_split_without_val_size(self):
        cfg = _make_clf_config()
        cfg.val_size = None
        sampler = SplitSampler(test_size=0.2, val_size=None, random_state=42, stratify=True)
        train, test, val = sampler(cfg)
        self.assertEqual(len(val), 0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_stratified_class_distribution(self):
        sampler = SplitSampler(test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        y = self.cfg._y
        for idx, name in [(train, "train"), (test, "test"), (val, "val")]:
            props = y.iloc[idx].value_counts(normalize=True)
            overall = y.value_counts(normalize=True)
            for cls in overall.index:
                self.assertAlmostEqual(
                    props.get(cls, 0.0),
                    overall[cls],
                    delta=0.1,
                    msg=f"Class {cls} proportion off in {name}",
                )

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
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler=SplitSampler(test_size=0.2, val_size=0.1, random_state=42, stratify=True),
        )
        cfg()
        self.assertIsNotNone(cfg.X_val)
        self.assertIsNotNone(cfg.y_val)
        self.assertIsInstance(cfg.X_val, pd.DataFrame)
        self.assertIsInstance(cfg.y_val, pd.Series)
        self.assertIsNotNone(cfg.val_n)
        self.assertGreater(cfg.val_n, 0)

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
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler=SplitSampler(test_size=0.2, val_size=0.1, random_state=42, stratify=True),
        )
        scores = cfg()
        self.assertIn("val_n", scores)

    def test_regression_val_score(self):
        cfg = DataConfig(
            dataset_name="make_regression",
            data_params={
                "n_samples": 100,
                "n_features": 4,
                "n_informative": 2,
                "random_state": 1,
            },
            test_size=0.2,
            val_size=0.1,
            random_state=1,
            stratify=False,
            classifier=False,
            sampler=SplitSampler(test_size=0.2, val_size=0.1, random_state=1, stratify=False),
        )
        scores = cfg()
        self.assertIn("val_n", scores)
        self.assertIn("mutual_information_mean", scores["test"])

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
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            stratify=True,
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
        self.assertIsNotNone(cfg.X_val)

    def test_regression_no_stratify(self):
        cfg = _make_reg_config(val_size=0.15)
        sampler = SplitSampler(test_size=0.2, val_size=0.15, random_state=1, stratify=False)
        train, test, val = sampler(cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)


# ---------------------------------------------------------------------------
# KFoldSampler
# ---------------------------------------------------------------------------


class TestKFoldSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_clf_config()

    def test_returns_three_arrays(self):
        sampler = KFoldSampler(n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True)
        result = sampler(self.cfg)
        self.assertEqual(len(result), 3)

    def test_indices_are_disjoint(self):
        sampler = KFoldSampler(n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(test) & set(val)), 0)

    def test_total_covers_dataset(self):
        sampler = KFoldSampler(n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        self.assertEqual(total, len(self.cfg._X))

    def test_different_folds_have_different_val_sets(self):
        results = []
        for fold in range(3):
            cfg = _make_clf_config()
            sampler = KFoldSampler(n_splits=5, split=fold, test_size=0.2, random_state=42, stratify=True)
            _, _, val = sampler(cfg)
            results.append(set(val))
        # Val sets across different folds must not all be identical
        self.assertFalse(results[0] == results[1] == results[2])

    def test_fold_out_of_range_raises(self):
        sampler = KFoldSampler(n_splits=5, split=99, test_size=0.2, random_state=42, stratify=True)
        with self.assertRaises(ValueError):
            sampler(self.cfg)

    def test_no_stratify(self):
        cfg = _make_clf_config(stratify=False)
        sampler = KFoldSampler(n_splits=5, split=0, test_size=0.2, shuffle=False, stratify=False)
        train, test, val = sampler(cfg)
        self.assertEqual(len(train) + len(test) + len(val), len(cfg._X))

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
            test_size=0.2,
            random_state=42,
            split=1,
            stratify=True,
            classifier=True,
            sampler=KFoldSampler(n_splits=5, split=1, test_size=0.2, random_state=42, stratify=True),
        )
        cfg()
        self.assertIsNotNone(cfg.X_val)
        self.assertIsNotNone(cfg.y_val)
        self.assertIsNotNone(cfg.val_n)
        self.assertGreater(cfg.val_n, 0)

    def test_fold_none_defaults_to_zero(self):
        cfg = _make_clf_config()
        cfg_split0 = _make_clf_config()
        sampler = KFoldSampler(n_splits=5, split=None, test_size=0.2, random_state=42, stratify=True)
        sampler_split0 = KFoldSampler(n_splits=5, split=0, test_size=0.2, random_state=42, stratify=True)
        _, _, val_none = sampler(cfg)
        _, _, val_zero = sampler_split0(cfg_split0)
        self.assertEqual(sorted(val_none), sorted(val_zero))

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
                    train_size=1000,
                    test_size=200,
                    val_size=200,
                    random_state=42,
                    stratify=True,
                    classifier=True,
                    split=split,
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
                cfg.split_data()
                self.assertEqual(len(cfg.X_train), 800)
                self.assertEqual(len(cfg.X_test), 200)
                self.assertEqual(len(cfg.X_val), 200)

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
            train_size=1000,
            test_size=201,
            val_size=200,
            random_state=42,
            stratify=True,
            classifier=True,
            split=0,
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
        with self.assertRaises(ValueError):
            cfg.split_data()


# ---------------------------------------------------------------------------
# ShuffleSampler
# ---------------------------------------------------------------------------


class TestShuffleSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_clf_config(val_size=0.15)

    def test_returns_three_arrays(self):
        sampler = ShuffleSampler(n_splits=5, split=0, test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        result = sampler(self.cfg)
        self.assertEqual(len(result), 3)

    def test_indices_are_disjoint_within_fold(self):
        sampler = ShuffleSampler(n_splits=5, split=0, test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(test) & set(val)), 0)

    def test_total_covers_dataset(self):
        sampler = ShuffleSampler(n_splits=5, split=0, test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        self.assertEqual(total, len(self.cfg._X))

    def test_raises_without_val_size(self):
        cfg = _make_clf_config()
        sampler = ShuffleSampler(n_splits=5, split=0, test_size=0.2, val_size=None, random_state=42, stratify=True)
        with self.assertRaises(ValueError):
            sampler(cfg)

    def test_fold_out_of_range_raises(self):
        sampler = ShuffleSampler(n_splits=5, split=99, test_size=0.2, val_size=0.15, random_state=42, stratify=True)
        with self.assertRaises(ValueError):
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
            test_size=0.2,
            val_size=0.15,
            split=2,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler=ShuffleSampler(n_splits=5, split=2, test_size=0.2, val_size=0.15, random_state=42, stratify=True),
        )
        cfg()
        self.assertIsNotNone(cfg.X_val)
        self.assertIsNotNone(cfg.y_val)
        self.assertIsNotNone(cfg.val_n)
        self.assertGreater(cfg.val_n, 0)

    def test_no_stratify(self):
        cfg = _make_reg_config(val_size=0.15)
        sampler = ShuffleSampler(n_splits=5, split=0, test_size=0.2, val_size=0.15, random_state=1, stratify=False)
        train, test, val = sampler(cfg)
        self.assertEqual(len(train) + len(test) + len(val), len(cfg._X))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2-way split (sampler="split", no val_size)
# ---------------------------------------------------------------------------


class TestLegacySplitUnchanged(unittest.TestCase):
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
            test_size=0.2,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler="split",
        )
        cfg()
        self.assertIsNone(cfg.X_val)
        self.assertIsNone(cfg.y_val)
        self.assertEqual(len(cfg.val_indices), 0)
        self.assertEqual(len(cfg.X_train) + len(cfg.X_test), 100)

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
            test_size=0.2,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler="split",
        )
        scores = cfg()
        self.assertNotIn("val_n", scores)
        self.assertNotIn("val_class_counts", scores)

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
            test_size=0.2,
            val_size=None,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler="split",
        )
        cfg()
        self.assertIsNone(cfg.X_val)
        self.assertIsNone(cfg.y_val)
        self.assertEqual(len(cfg.X_train) + len(cfg.X_test), 100)


# ---------------------------------------------------------------------------
# OmegaConf DictConfig sampler spec
# ---------------------------------------------------------------------------


class TestOmegaConfSampleSpec(unittest.TestCase):
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
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            stratify=True,
            classifier=True,
            sampler=SplitSampler(test_size=0.2, val_size=0.1, random_state=42, stratify=True),
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
        cfg.split_data()
        self.assertIsNotNone(cfg.X_val)
        self.assertGreater(len(cfg.X_val), 0)


# ---------------------------------------------------------------------------
# Hydra ConfigStore registration
# ---------------------------------------------------------------------------


class TestConfigStoreRegistration(unittest.TestCase):
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
            self.assertIn(
                expected,
                listed_names,
                msg=f"Expected '{expected}' in ConfigStore sample group, got: {listed_names}",
            )


if __name__ == "__main__":
    unittest.main()
