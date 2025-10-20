import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from deckard.experiment import ExperimentConfig
from deckard.data import DataConfig
from deckard.model import ModelConfig
from deckard.model.defend import DefenseConfig
from deckard.attack import AttackConfig

# from deckard.score import ScorerDictConfig  # Removed unused import
from deckard.file import FileConfig


class DummyDataConfig(DataConfig):
    def __call__(self, **kwargs):
        self.X_train = np.array([1, 2, 3])
        self.y_train = np.array([0, 1, 0])
        self.X_test = np.array([4, 5])
        self.y_test = np.array([1, 0])
        self.score_dict = {"acc": 1.0}
        return self


class DummyModelConfig(ModelConfig):
    def __call__(self, data, **kwargs):
        class DummyModel:
            training_predictions = [0, 1, 0]
            predictions = [1, 0]
            score_dict = {"acc": 1.0}

        return data, DummyModel()


class DummyDefenseConfig(DefenseConfig):
    def __call__(self, **kwargs):
        class DummyModel:
            training_predictions = [0, 1, 0]
            predictions = [1, 0]
            score_dict = {"acc": 1.0}

        data = MagicMock()
        data.score_dict = {"acc": 1.0}
        return data, DummyModel()


class DummyAttackConfig(AttackConfig):
    def __call__(self, **kwargs):
        class DummyAttack:
            attack = True
            attack_training_predictions = [0, 1, 0]
            attack_predictions = [1, 0]
            attack_score_dict = {"acc": 1.0}

        data = DataConfig()
        model = ModelConfig()
        return data, model, DummyAttack()


class DummyFileConfig(FileConfig):
    def __call__(self, **kwargs):
        # Return dummy file paths that exist
        temp_dir = tempfile.mkdtemp()
        files = {
            "model_file": str(Path(temp_dir) / "model.pkl"),
            "data_file": str(Path(temp_dir) / "data.csv"),
            "log_file": str(Path(temp_dir) / "log.log"),
        }
        # Create the files
        for f in files.values():
            Path(f).touch()
        return files


class TestExperimentConfig(unittest.TestCase):
    def setUp(self):
        self.data_config = DummyDataConfig()
        self.model_config = DummyModelConfig()
        self.defense_config = DummyDefenseConfig()
        self.attack_config = DummyAttackConfig()
        self.file_config = DummyFileConfig(experiment_name="test_experiment")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_random_seed_sklearn(self):
        config = ExperimentConfig(
            data_config=self.data_config,
            experiment_name="test",
            file_config=self.file_config,
            library="sklearn",
            random_state=123,
        )
        config.set_random_seed()
        arr = np.random.rand(3)
        config.set_random_seed()
        arr2 = np.random.rand(3)
        self.assertTrue(np.allclose(arr, arr2))

    @patch("deckard.experiment.ExperimentConfig.set_device")
    def test_post_init_sets_experiment_name_hash(self, _):
        config = ExperimentConfig(
            data_config=self.data_config,
            experiment_name="{hash}",
            file_config=None,
        )
        config.results_path = self.temp_dir
        config.__post_init__()
        self.assertEqual(len(config.experiment_name), 32)
        self.assertIsInstance(config.file_config, FileConfig)

    def test_hash_from_config_list(self):
        config = ExperimentConfig(
            data_config=self.data_config,
            experiment_name="test",
            file_config=self.file_config,
        )
        hash_str = config._hash_from_config_list([self.data_config])
        self.assertEqual(len(hash_str), 32)

    def test_call_with_model_config(self):
        config = ExperimentConfig(
            data_config=self.data_config,
            model_config=self.model_config,
            experiment_name="test",
            file_config=self.file_config,
        )
        config.results_path = self.temp_dir
        config.__call__()

    def test_call_with_defense_config(self):
        config = ExperimentConfig(
            data_config=self.data_config,
            defense_config=self.defense_config,
            experiment_name="test",
            file_config=self.file_config,
        )
        config.results_path = self.temp_dir
        config.__call__()

    def test_call_with_attack_config(self):
        config = ExperimentConfig(
            data_config=self.data_config,
            model_config=self.model_config,
            attack_config=self.attack_config,
            experiment_name="test",
            file_config=self.file_config,
        )
        config.results_path = self.temp_dir
        config.__call__()

    def test_call_file_not_found_raises(self):
        class BadFileConfig(FileConfig):
            def __call__(self):
                return {"model_file": "/tmp/nonexistent_file.pkl"}

        config = ExperimentConfig(
            data_config=self.data_config,
            experiment_name="test",
            file_config=BadFileConfig(experiment_name="test"),
        )
        config.results_path = self.temp_dir
        with self.assertRaises(FileNotFoundError):
            config.__call__()


if __name__ == "__main__":
    unittest.main()
