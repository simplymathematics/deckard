import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig, SurvivalExperimentConfig
from deckard.file import FileConfig
from deckard.model import ModelConfig
from deckard.score import DefaultClassifierConfig
from helpers import make_runtime_env
from deckard.attack.base import resolve_class as base_resolve_class


class DummyPoisonAttack:
    """Simple local attack used to exercise poisoning integration flow."""

    def __init__(self, classifier, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs

    def poison(self, x_trigger, y_trigger, x_train, y_train):
        _ = x_trigger
        y_target = int(np.argmax(np.asarray(y_trigger), axis=1)[0])
        x_poison = np.asarray(x_train).copy()
        y_poison = np.asarray(y_train).copy()
        if y_poison.ndim == 2 and y_poison.shape[1] > 1:
            y_poison[0] = 0
            y_poison[0, y_target] = 1
        else:
            y_poison = y_poison.reshape(-1)
            y_poison[0] = y_target
        return x_poison, y_poison


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
        split_accs = [scores[f"accuracy_split_{k}"] for k in range(self.N_SPLITS)]
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


class _FakeDetector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.score_dict = {}

    def __post_init__(self):
        return None

    def __call__(self, data, model, attack):
        _ = data, model, attack
        self.score_dict = {"detector_accuracy": 0.75, "detector_n": 10}
        return self.score_dict


class TestExperimentDetectorPhase(unittest.TestCase):
    def test_detector_phase_runs_after_attack(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 60,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 0,
                    "n_classes": 2,
                    "random_state": 7,
                },
                train_size=40,
                test_size=20,
                random_state=7,
                stratify=True,
                classifier=True,
            ),
            model=ModelConfig(
                model_type="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 30},
            ),
            attack=AttackConfig(
                attack_type="art.attacks.evasion.FastGradientMethod",
                attack_params={"eps": 0.1},
                attack_size=10,
            ),
            detector=_FakeDetector(),
            files=FileConfig(),
        )

        scores = exp()

        self.assertIn("detector_accuracy", scores)
        self.assertIn("detector_n", scores)


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


class TestPoisoningExperimentIntegration(unittest.TestCase):
    def test_poisoning_experiment_emits_benign_and_poisoned_accuracy(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 120,
                    "n_features": 8,
                    "n_informative": 6,
                    "n_redundant": 2,
                    "random_state": 4,
                },
                test_size=0.2,
                random_state=4,
                classifier=True,
            ),
            model=ModelConfig(
                model_type="sklearn.linear_model.LogisticRegression",
                classifier=True,
                model_params={"max_iter": 150},
            ),
            attack=AttackConfig(
                attack_type="art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack",
                attack_params={"class_source": 0, "class_target": 1},
                attack_size=20,
            ),
            files=FileConfig(),
        )

        with patch(
            "deckard.attack.base.resolve_class",
            wraps=base_resolve_class,
        ) as mocked_resolve:

            def _resolve(name):
                if (
                    name
                    == "art.attacks.poisoning.gradient_matching_attack.GradientMatchingAttack"
                ):
                    return DummyPoisonAttack
                return base_resolve_class(name)

            mocked_resolve.side_effect = _resolve
            scores = exp()

        self.assertIn("benign_accuracy", scores)
        self.assertIn("poisoned_accuracy", scores)

    def test_deckard_optimize_help_smoke(self):
        examples_dir = Path(__file__).resolve().parents[1] / "examples" / "sklearn"
        rc_path = examples_dir / ".deckard_rc"
        env = make_runtime_env(rc_path)

        result = subprocess.run(
            [sys.executable, "-m", "deckard", "optimize", "--help"],
            cwd=str(examples_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_single_pass_with_attack_runs_experiment_scorer_once(self):
        exp = ExperimentConfig(
            data=DataConfig(
                dataset_name="make_classification",
                data_params={
                    "n_samples": 80,
                    "n_features": 6,
                    "n_informative": 4,
                    "n_redundant": 2,
                    "random_state": 0,
                    "n_clusters_per_class": 1,
                },
                train_size=60,
                test_size=20,
                random_state=42,
                classifier=True,
                sample="split",
            ),
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5, "random_state": 0},
            ),
            attack=AttackConfig(
                attack_type="art.attacks.evasion.FastGradientMethod",
                attack_params={"eps": 0.1},
                attack_size=10,
            ),
            score={"experiment": DefaultClassifierConfig()},
            files=FileConfig(),
            experiment_name="single-pass-scorer-call-count",
            evaluation_mode="standard",
        )

        with (
            patch.object(
                AttackConfig,
                "__call__",
                autospec=True,
                side_effect=lambda self, data, model, **kwargs: {
                    "evasion_accuracy": 0.5,
                    "attack_generation_time": 0.01,
                    "attack_prediction_time": 0.01,
                    "attack_score_time": 0.01,
                },
            ),
            patch.object(
                ExperimentConfig,
                "_run_experiment_scorer_modes",
                autospec=True,
                wraps=ExperimentConfig._run_experiment_scorer_modes,
            ) as mocked_run_modes,
        ):
            _ = exp()

        assert mocked_run_modes.call_count == 1


if __name__ == "__main__":
    unittest.main()
