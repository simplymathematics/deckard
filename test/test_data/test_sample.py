"""Tests for deckard/data/sample.py and the sampler integration in DataConfig."""

import unittest

import numpy as np
import pandas as pd

from deckard.data import (
    DataConfig,
    BaseSampler,
    KFoldSampler,
    ShuffleSampler,
    SplitSampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clf_config(**kwargs):
    """Return a small classification DataConfig, loading data but not yet sampling."""
    defaults = dict(
        dataset_name="make_classification",
        data_params={
            "n_samples": 120,
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
    )
    defaults.update(kwargs)
    cfg = DataConfig(**defaults)
    cfg._load_data()
    return cfg


def _make_reg_config(**kwargs):
    """Return a small regression DataConfig."""
    defaults = dict(
        dataset_name="make_regression",
        data_params={
            "n_samples": 100,
            "n_features": 4,
            "n_informative": 2,
            "random_state": 1,
        },
        test_size=0.2,
        random_state=1,
        stratify=False,
        classifier=False,
    )
    defaults.update(kwargs)
    cfg = DataConfig(**defaults)
    cfg._load_data()
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
        sampler = SplitSampler()
        result = sampler(self.cfg)
        self.assertEqual(len(result), 3)

    def test_indices_are_disjoint(self):
        sampler = SplitSampler()
        train, test, val = sampler(self.cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(test) & set(val)), 0)

    def test_total_covers_dataset(self):
        sampler = SplitSampler()
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        self.assertEqual(total, len(self.cfg._X))

    def test_two_way_split_without_val_size(self):
        cfg = _make_clf_config()
        cfg.val_size = None
        sampler = SplitSampler()
        train, test, val = sampler(cfg)
        self.assertEqual(len(val), 0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_stratified_class_distribution(self):
        sampler = SplitSampler()
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
            sample=SplitSampler(),
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
            sample=SplitSampler(),
        )
        scores = cfg()
        self.assertIn("val_n", scores)
        self.assertIn("val_class_counts", scores)

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
            sample=SplitSampler(),
        )
        scores = cfg()
        self.assertIn("val_n", scores)
        self.assertIn("val_y_cdf", scores)

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
            sample={"name": "deckard.data.sample.SplitSampler"},
        )
        cfg()
        self.assertIsNotNone(cfg.X_val)

    def test_regression_no_stratify(self):
        cfg = _make_reg_config(val_size=0.15)
        sampler = SplitSampler()
        train, test, val = sampler(cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)


# ---------------------------------------------------------------------------
# KFoldSampler
# ---------------------------------------------------------------------------


class TestKFoldSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_clf_config()
        self.cfg.fold = 0

    def test_returns_three_arrays(self):
        sampler = KFoldSampler(n_splits=5)
        result = sampler(self.cfg)
        self.assertEqual(len(result), 3)

    def test_indices_are_disjoint(self):
        sampler = KFoldSampler(n_splits=5)
        train, test, val = sampler(self.cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(test) & set(val)), 0)

    def test_total_covers_dataset(self):
        sampler = KFoldSampler(n_splits=5)
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        self.assertEqual(total, len(self.cfg._X))

    def test_different_folds_have_different_val_sets(self):
        results = []
        for fold in range(3):
            cfg = _make_clf_config()
            cfg.fold = fold
            sampler = KFoldSampler(n_splits=5)
            _, _, val = sampler(cfg)
            results.append(set(val))
        # Val sets across different folds must not all be identical
        self.assertFalse(results[0] == results[1] == results[2])

    def test_fold_out_of_range_raises(self):
        self.cfg.fold = 99
        sampler = KFoldSampler(n_splits=5)
        with self.assertRaises(ValueError):
            sampler(self.cfg)

    def test_no_stratify(self):
        cfg = _make_clf_config(stratify=False)
        cfg.fold = 0
        sampler = KFoldSampler(n_splits=5, shuffle=False)
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
            fold=1,
            stratify=True,
            classifier=True,
            sample=KFoldSampler(n_splits=5),
        )
        cfg()
        self.assertIsNotNone(cfg.X_val)
        self.assertIsNotNone(cfg.y_val)
        self.assertIsNotNone(cfg.val_n)
        self.assertGreater(cfg.val_n, 0)

    def test_fold_none_defaults_to_zero(self):
        cfg = _make_clf_config()
        cfg.fold = None
        cfg_fold0 = _make_clf_config()
        cfg_fold0.fold = 0
        sampler = KFoldSampler(n_splits=5)
        _, _, val_none = sampler(cfg)
        _, _, val_zero = sampler(cfg_fold0)
        self.assertEqual(sorted(val_none), sorted(val_zero))


# ---------------------------------------------------------------------------
# ShuffleSampler
# ---------------------------------------------------------------------------


class TestShuffleSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_clf_config(val_size=0.15)
        self.cfg.fold = 0

    def test_returns_three_arrays(self):
        sampler = ShuffleSampler(n_splits=5)
        result = sampler(self.cfg)
        self.assertEqual(len(result), 3)

    def test_indices_are_disjoint_within_fold(self):
        sampler = ShuffleSampler(n_splits=5)
        train, test, val = sampler(self.cfg)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(test) & set(val)), 0)

    def test_total_covers_dataset(self):
        sampler = ShuffleSampler(n_splits=5)
        train, test, val = sampler(self.cfg)
        total = len(train) + len(test) + len(val)
        self.assertEqual(total, len(self.cfg._X))

    def test_raises_without_val_size(self):
        cfg = _make_clf_config()
        cfg.val_size = None
        cfg.fold = 0
        sampler = ShuffleSampler(n_splits=5)
        with self.assertRaises(ValueError):
            sampler(cfg)

    def test_fold_out_of_range_raises(self):
        self.cfg.fold = 99
        sampler = ShuffleSampler(n_splits=5)
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
            fold=2,
            random_state=42,
            stratify=True,
            classifier=True,
            sample=ShuffleSampler(n_splits=5),
        )
        cfg()
        self.assertIsNotNone(cfg.X_val)
        self.assertIsNotNone(cfg.y_val)
        self.assertIsNotNone(cfg.val_n)
        self.assertGreater(cfg.val_n, 0)

    def test_no_stratify(self):
        cfg = _make_reg_config(val_size=0.15)
        cfg.fold = 0
        sampler = ShuffleSampler(n_splits=5)
        train, test, val = sampler(cfg)
        self.assertEqual(len(train) + len(test) + len(val), len(cfg._X))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2-way split (sample="split", no val_size)
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
            sample="split",
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
            sample="split",
        )
        scores = cfg()
        self.assertNotIn("val_n", scores)
        self.assertNotIn("val_class_counts", scores)

    def test_val_size_none_gives_two_way_split(self):
        """sample='split' with val_size=None should not create a val set."""
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
            sample="split",
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
        )
        # Simulate Hydra passing an OmegaConf DictConfig for the sampler
        cfg.sample = OmegaConf.create(
            {"name": "deckard.data.sample.SplitSampler"},
        )
        cfg._load_data()
        cfg._sample()
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
        from deckard.data.sample import register_sampler_configs
        from hydra.core.config_store import ConfigStore

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
