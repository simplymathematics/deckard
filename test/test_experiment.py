import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
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


class TestExperimentConfig(unittest.TestCase):
    def setUp(self):
        # Set up temporary directories and mock data for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.defense_config = DefenseConfig()
        self.attack_config = AttackConfig(attack_size=1)
        self.file_config = FileConfig()
        self.experiment_config = ExperimentConfig(
            data=self.data_config,
            model=self.model_config,
            defense=self.defense_config,
            attack=self.attack_config,
            files=self.file_config,
            experiment_name="test_experiment",
        )

    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.temp_dir)

    def test_experiment_initialization(self):
        # Test initialization of ExperimentConfig
        self.assertEqual(self.experiment_config.experiment_name, "test_experiment")
        self.assertIsInstance(self.experiment_config.data, DataConfig)
        self.assertIsInstance(self.experiment_config.model, ModelConfig)
        self.assertIsInstance(self.experiment_config.defense, DefenseConfig)
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
        mock_data = MagicMock()
        mock_data.X_train = pd.DataFrame(np.random.rand(100, 10))
        mock_data.y_train = pd.Series(np.random.randint(0, 2, size=100))
        mock_data.X_test = pd.DataFrame(np.random.rand(20, 10))
        mock_data.y_test = pd.Series(np.random.randint(0, 2, size=20))
        self.experiment_config.data = MagicMock(return_value=mock_data)
        scores = self.experiment_config()
        self.assertIsInstance(scores, dict)

    # def test_initialize_file_config(self):
    #     # Test initializing file configuration
    #     self.experiment_config.files = None
    #     self.experiment_config.initialize_file_config({})
    #     self.assertIsInstance(self.experiment_config.files, dict)


if __name__ == "__main__":
    unittest.main()