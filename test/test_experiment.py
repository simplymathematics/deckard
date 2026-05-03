import unittest
import numpy as np
import tempfile
import shutil
from unittest.mock import patch
from deckard.experiment import ExperimentConfig, SurvivalExperimentConfig
from deckard.data import DataConfig, DataPipelineConfig, FairlearnDataConfig
from deckard.model import ModelConfig, FairlearnModelConfig
from deckard.model.defend import DefensePipelineConfig
from deckard.attack import AttackConfig
from deckard.score import DefaultClassifierConfig

# from deckard.score import ScorerDictConfig  # Removed unused import
from deckard.file import FileConfig


class TestExperimentConfig(unittest.TestCase):
    def setUp(self):
        # Set up temporary directories and mock data for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_config = DataConfig(dataset_name="adult")
        self.model_config = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
        )
        self.attack_config = AttackConfig(attack_size=1)
        self.file_config = FileConfig()
        self.experiment_config = ExperimentConfig(
            data=self.data_config,
            model=self.model_config,
            attack=self.attack_config,
            files=self.file_config,
            experiment_name="test_experiment",
        )
        self.experiment_config()

    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.temp_dir)

    def test_experiment_initialization(self):
        # Test initialization of ExperimentConfig
        self.assertEqual(self.experiment_config.experiment_name, "test_experiment")
        self.assertIsInstance(self.experiment_config.data, DataConfig)
        self.assertIsInstance(self.experiment_config.model, ModelConfig)
        self.assertIsInstance(self.experiment_config.attack, AttackConfig)
        self.assertIsInstance(self.experiment_config.files, FileConfig)

    def test_set_random_seed(self):
        # Test setting random seed
        self.experiment_config.library = "sklearn"
        self.experiment_config.set_random_seed()
        random_state = np.random.get_state()
        self.assertEqual(random_state[1][0], self.experiment_config.random_state)

    def test_call_with_mock_data(self):
        # Test the __call__ method with mock data
        scores = self.experiment_config()
        self.assertIsInstance(scores, dict)
        self.assertIn("accuracy", scores)
        self.assertIn("evasion_accuracy", scores)
        self.assertIn("data_load_time", scores)

    def test_hash_stable_after_call_for_experiment_config(self):
        """Test that ExperimentConfig hash remains stable after execution."""
        experiment = ExperimentConfig(
            data=self.data_config,
            model=self.model_config,
            attack=self.attack_config,
            files=self.file_config,
            experiment_name="test_experiment",
        )
        original_hash = hash(experiment)
        cls = experiment.__class__
        original_call = cls.__call__

        def fake_call(self):
            self._execution_time = 5.0
            self._runtime_field = {"executed": True}
            if hasattr(self, "score_dict") and isinstance(self.score_dict, dict):
                self.score_dict["runtime"] = 1
            return {"ok": 1}

        setattr(cls, "__call__", fake_call)
        try:
            experiment.execute_without_mercy()
        finally:
            setattr(cls, "__call__", original_call)

        self.assertEqual(
            original_hash,
            hash(experiment),
            msg="Hash changed after call for ExperimentConfig",
        )

    def test_data_dict_with_pipeline_infers_data_pipeline_config(self):
        exp = ExperimentConfig(
            data={
                "dataset_name": "make_classification",
                "data_params": {"n_samples": 20, "n_features": 6},
                "pipeline": {},
                "classifier": True,
            },
            model=None,
            attack=None,
            files=FileConfig(),
            classifier=True,
        )
        self.assertIsInstance(exp.data, DataPipelineConfig)

    def test_data_dict_with_fairness_keys_infers_fairness_data_config(self):
        exp = ExperimentConfig(
            data={
                "dataset_name": "make_classification",
                "data_params": {"n_samples": 20, "n_features": 6},
                "sensitive_columns": ["feature_0"],
                "pipeline": {},
                "classifier": True,
            },
            model=None,
            attack=None,
            files=FileConfig(),
            classifier=True,
        )
        self.assertIsInstance(exp.data, FairlearnDataConfig)

    def test_fairness_data_auto_uses_fairness_model_config(self):
        exp = ExperimentConfig(
            data={
                "dataset_name": "make_classification",
                "data_params": {"n_samples": 40, "n_features": 6},
                "sensitive_columns": ["feature_0"],
                "pipeline": {},
                "classifier": True,
            },
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 5},
            ),
            attack=None,
            files=FileConfig(),
            classifier=True,
        )

        self.assertIsInstance(exp.data, FairlearnDataConfig)
        self.assertIsInstance(exp.model, FairlearnModelConfig)

    def test_fairness_data_keeps_standard_defense_config(self):
        exp = ExperimentConfig(
            data={
                "dataset_name": "adult",
                "classifier": True,
                "sensitive_columns": ["sex"],
            },
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 10},
            ),
            defense={
                "defenses": [
                    {
                        "defense_name": "art.defences.postprocessor.ClassLabels",
                        "defense_params": {"apply_fit": False, "apply_predict": True},
                        "alias": "class-labels",
                        "model_type": "sklearn.ensemble.RandomForestClassifier",
                        "classifier": True,
                    },
                ],
            },
            attack=None,
            files=None,
            score=None,
        )

        self.assertIsInstance(exp.data, FairlearnDataConfig)
        self.assertIsInstance(exp.model, FairlearnModelConfig)
        self.assertIsInstance(exp.model.defense, DefensePipelineConfig)
        self.assertIs(exp.model.data, exp.data)

    def test_pytorch_device_propagates_from_experiment(self):
        data = DataConfig(dataset_name="make_classification")
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
        )
        exp = ExperimentConfig(
            data=data,
            model=model,
            attack=None,
            files=FileConfig(),
            library="pytorch",
            device="cpu",
            classifier=True,
        )

        self.assertEqual(str(exp.device), "cpu")
        self.assertEqual(str(getattr(exp.data, "device")), "cpu")
        self.assertEqual(str(getattr(exp.model, "device")), "cpu")

    def test_pytorch_device_propagates_from_model_when_only_one_specified(self):
        data = DataConfig(dataset_name="make_classification")
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
        )
        model.device = "cpu"
        exp = ExperimentConfig(
            data=data,
            model=model,
            attack=None,
            files=FileConfig(),
            library="pytorch",
            classifier=True,
        )

        self.assertEqual(str(exp.device), "cpu")
        self.assertEqual(str(getattr(exp.data, "device")), "cpu")
        self.assertEqual(str(getattr(exp.model, "device")), "cpu")

    def test_pytorch_device_mismatch_raises_assertion(self):
        data = DataConfig(dataset_name="make_classification")
        data.device = "cuda"
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
        )
        model.device = "mps"

        with self.assertRaises(AssertionError):
            ExperimentConfig(
                data=data,
                model=model,
                attack=None,
                files=FileConfig(),
                library="pytorch",
                classifier=True,
            )

    def test_pytorch_auto_device_propagates_to_all_components(self):
        data = DataConfig(dataset_name="make_classification")
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
        )
        attack = AttackConfig(attack_size=1)

        with patch("deckard.experiment.base.resolve_torch_device") as resolve_device:
            resolve_device.return_value = "cpu"
            exp = ExperimentConfig(
                data=data,
                model=model,
                attack=attack,
                files=FileConfig(),
                library="pytorch",
                classifier=True,
            )

        self.assertEqual(str(exp.device), "cpu")
        self.assertEqual(str(getattr(exp.data, "device")), "cpu")
        self.assertEqual(str(getattr(exp.model, "device")), "cpu")
        self.assertEqual(str(getattr(exp.attack, "device")), "cpu")


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
            self.assertIn(f"accuracy_fold_{k}", scores, f"missing accuracy_fold_{k}")

    def test_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        self.assertIn("accuracy", scores)

    def test_mean_equals_average_of_folds(self):
        exp = self._make_exp()
        scores = exp()
        fold_accs = [scores[f"accuracy_fold_{k}"] for k in range(self.N_FOLDS)]
        self.assertAlmostEqual(scores["accuracy"], float(np.mean(fold_accs)), places=10)

    def test_correct_number_of_folds_run(self):
        exp = self._make_exp()
        scores = exp()
        fold_keys = [k for k in scores if k.endswith(f"_fold_{self.N_FOLDS - 1}")]
        self.assertGreater(len(fold_keys), 0)


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
            self.assertIn(f"accuracy_split_{k}", scores, f"missing accuracy_split_{k}")

    def test_mean_key_present(self):
        exp = self._make_exp()
        scores = exp()
        self.assertIn("accuracy", scores)

    def test_mean_equals_average_of_splits(self):
        exp = self._make_exp()
        scores = exp()
        split_accs = [scores[f"accuracy_split_{k}"] for k in range(self.N_SPLITS)]
        self.assertAlmostEqual(scores["accuracy"], float(np.mean(split_accs)), places=10)

    def test_correct_number_of_splits_run(self):
        exp = self._make_exp()
        scores = exp()
        split_keys = [k for k in scores if k.endswith(f"_split_{self.N_SPLITS - 1}")]
        self.assertGreater(len(split_keys), 0)


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
                data=DataConfig(dataset_name="make_regression", classifier=False),
                model=None,
                classifier=False,
                survival_model="cox",
                duration_col="T",
                event_col="E",
                attack=AttackConfig(attack_type="art.attacks.evasion.HopSkipJump"),
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
