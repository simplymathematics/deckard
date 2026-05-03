import unittest

import numpy as np

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig, SurvivalExperimentConfig
from deckard.file import FileConfig
from deckard.model import ModelConfig
from deckard.score import DefaultClassifierConfig


class TestKFoldExperiment(unittest.TestCase):
    """ExperimentConfig should loop over all folds when sample='fold'."""

    N_FOLDS = 3

    def _make_exp(self):
        from deckard.data.sample import KFoldSampler

        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                test_size=0.2,
                random_state=42,
                classifier=True,
                sample=KFoldSampler(n_splits=self.N_FOLDS),
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            attack=None,
            files=FileConfig(),
        )

    def test_per_fold_keys_present(self):
        exp = self._make_exp()
        scores = exp()
        for k in range(self.N_FOLDS):
            self.assertIn(
                f"accuracy_fold_{k}",
                scores,
                f"missing accuracy_fold_{k}",
            )

    def test_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        self.assertIn("accuracy", scores)

    def test_mean_equals_average_of_folds(self):
        exp = self._make_exp()
        scores = exp()
        fold_accs = [scores[f"accuracy_fold_{k}"] for k in range(self.N_FOLDS)]
        self.assertAlmostEqual(
            scores["accuracy"],
            float(np.mean(fold_accs)),
            places=10,
        )


class TestShuffleExperiment(unittest.TestCase):
    """ExperimentConfig should loop over all shuffle splits when sample='shuffle'."""

    N_SPLITS = 3

    def _make_exp(self):
        from deckard.data.sample import ShuffleSampler

        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                test_size=0.2,
                val_size=0.1,
                random_state=42,
                classifier=True,
                sample=ShuffleSampler(n_splits=self.N_SPLITS),
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            attack=None,
            files=FileConfig(),
        )

    def test_per_split_keys_present(self):
        exp = self._make_exp()
        scores = exp()
        for k in range(self.N_SPLITS):
            self.assertIn(
                f"accuracy_split_{k}",
                scores,
                f"missing accuracy_split_{k}",
            )

    def test_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        self.assertIn("accuracy", scores)

    def test_mean_equals_average_of_splits(self):
        exp = self._make_exp()
        scores = exp()
        split_accs = [
            scores[f"accuracy_split_{k}"] for k in range(self.N_SPLITS)
        ]
        self.assertAlmostEqual(
            scores["accuracy"],
            float(np.mean(split_accs)),
            places=10,
        )


class TestExperimentValidationScoring(unittest.TestCase):
    def _make_exp(self, *, val_size=0.1, evaluation_mode="tuning"):
        return ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                test_size=0.2,
                val_size=val_size,
                random_state=42,
                classifier=True,
                sample="split",
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            score={"experiment": DefaultClassifierConfig()},
            attack=None,
            files=FileConfig(),
            experiment_name="validation-scoring-test",
            evaluation_mode=evaluation_mode,
        )

    def test_tuning_mode_emits_validation_scores(self):
        exp = self._make_exp(evaluation_mode="tuning")
        scores = exp()
        self.assertIn("validation_accuracy", scores)
        self.assertNotIn("training_accuracy", scores)

    def test_report_mode_emits_train_test_and_validation_scores(self):
        exp = self._make_exp(evaluation_mode="report")
        scores = exp()
        self.assertIn("training_accuracy", scores)
        self.assertIn("accuracy", scores)
        self.assertIn("validation_accuracy", scores)

    def test_tuning_mode_without_validation_split_raises(self):
        exp = self._make_exp(val_size=None, evaluation_mode="tuning")
        with self.assertRaises(ValueError):
            exp()


class TestSurvivalExperimentConfig(unittest.TestCase):
    def test_allows_survival_only_config_without_attack(self):
        config = SurvivalExperimentConfig(
            data=DataConfig(dataset_name="make_regression", classifier=False),
            model=None,
            classifier=False,
            survival_model="cox",
            duration_col="T",
            event_col="E",
        )
        self.assertIsInstance(config, SurvivalExperimentConfig)
        self.assertIsNone(config.model)

    def test_requires_model_config_when_attack_present(self):
        with self.assertRaises(ValueError):
            SurvivalExperimentConfig(
                data=DataConfig(
                    dataset_name="make_regression",
                    classifier=False,
                ),
                model=None,
                classifier=False,
                survival_model="cox",
                duration_col="T",
                event_col="E",
                attack=AttackConfig(
                    attack_type="art.attacks.evasion.HopSkipJump",
                ),
            )

    def test_requires_data_config(self):
        with self.assertRaises(ValueError):
            SurvivalExperimentConfig(
                data=None,
                model=ModelConfig(
                    model_type="sklearn.linear_model.LogisticRegression",
                    classifier=True,
                    model_params={"max_iter": 10},
                ),
                classifier=False,
            )

    def test_survival_config_initializes(self):
        config = SurvivalExperimentConfig(
            data=DataConfig(
                dataset_name="make_regression",
                classifier=False,
                target=None,
            ),
            model=ModelConfig(
                model_type="sklearn.linear_model.LinearRegression",
                classifier=False,
                model_params={},
            ),
            classifier=False,
            survival_model="cox",
            duration_col="T",
            event_col="E",
        )
        self.assertIsInstance(config, SurvivalExperimentConfig)
        self.assertEqual(config.survival_model, "cox")


if __name__ == "__main__":
    unittest.main()
