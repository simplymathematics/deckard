import unittest
import numpy as np
import tempfile
import shutil
from deckard.experiment import ExperimentConfig, SurvivalExperimentConfig
from deckard.data import DataConfig, DataPipelineConfig, FairnessDataConfig
from deckard.model import ModelConfig, FairnessModelConfig
from deckard.model.defend import DefenseConfig
from deckard.attack import AttackConfig

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
                "groupby_columns": ["feature_0"],
                "sensitive_columns": ["feature_0"],
                "pipeline": {},
                "classifier": True,
            },
            model=None,
            attack=None,
            files=FileConfig(),
            classifier=True,
        )
        self.assertIsInstance(exp.data, FairnessDataConfig)

    def test_fairness_data_auto_uses_fairness_model_config(self):
        exp = ExperimentConfig(
            data={
                "dataset_name": "make_classification",
                "data_params": {"n_samples": 40, "n_features": 6},
                "groupby_columns": ["feature_0"],
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

        self.assertIsInstance(exp.data, FairnessDataConfig)
        self.assertIsInstance(exp.model, FairnessModelConfig)

    def test_fairness_data_keeps_standard_defense_config(self):
        exp = ExperimentConfig(
            data={
                "dataset_name": "adult",
                "classifier": True,
                "groupby_columns": ["sex"],
                "sensitive_columns": ["sex"],
            },
            model=ModelConfig(
                model_type="sklearn.ensemble.RandomForestClassifier",
                classifier=True,
                model_params={"n_estimators": 10},
            ),
            defense={
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "alias": "class-labels",
            },
            attack=None,
            files=None,
            score=None,
        )

        self.assertIsInstance(exp.data, FairnessDataConfig)
        self.assertIsInstance(exp.model, FairnessModelConfig)
        self.assertIsInstance(exp.model.defense, DefenseConfig)
        self.assertIs(exp.model.data, exp.data)


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
