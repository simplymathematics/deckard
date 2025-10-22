import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
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
        scores = self.experiment_config()
        self.assertIsInstance(scores, dict)
        self.assertIn("accuracy", scores)
        self.assertIn("evasion_accuracy", scores)
        self.assertIn("data_load_time", scores)



if __name__ == "__main__":
    unittest.main()
