import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import shutil
from deckard.model.defend import DefenseConfig


class DummyDataConfig:
    def __init__(self, X_train, y_train, X_test, y_test):
        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


class TestDefenseConfig(unittest.TestCase):
    def setUp(self):
        # Set up temporary directories and mock data for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_file = os.path.join(self.temp_dir, "model.pkl")
        self.test_predictions_file = os.path.join(self.temp_dir, "predictions.csv")
        self.train_predictions_file = os.path.join(
            self.temp_dir,
            "training_predictions.csv",
        )
        self.score_file = os.path.join(self.temp_dir, "model_score.json")

        # Mock data
        self.data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(100, 10)),
            y_train=pd.Series(np.random.randint(0, 2, size=100)),
            X_test=pd.DataFrame(np.random.rand(20, 10)),
            y_test=pd.Series(np.random.randint(0, 2, size=20)),
        )

        # Default DefenseConfig
        self.defense_config = DefenseConfig(defense_name="art.defences.postprocessor.HighConfidence")

    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.temp_dir)

    def test_defense_config_initialization(self):
        # Test default initialization
        self.assertEqual(
            self.defense_config.model_type,
            "sklearn.linear_model.LogisticRegression",
        )
        self.assertTrue(self.defense_config.classifier)
        self.assertFalse(self.defense_config.probability)
        self.assertIsNone(self.defense_config.clip_values)
        self.assertEqual(
            self.defense_config.defense_name,
            "art.defences.postprocessor.HighConfidence",
        )

    def test_apply_defense_without_model(self):
        # Test applying defense without a fitted model
        with self.assertRaises(ValueError):
            self.defense_config.apply_defense()

    def test_apply_defense_with_invalid_defense_name(self):
        # Test applying defense with an invalid defense name
        self.defense_config.defense_name = "invalid.defense.Class"
        with self.assertRaises(ImportError):
            self.defense_config.apply_defense()

    def test_call_with_unloaded_data(self):
        # Test calling the DefenseConfig with unloaded data
        self.data.X_train = None
        self.data.y_train = None
        with self.assertRaises(ValueError):
            self.defense_config(self.data)

    def test_call_with_mock_data(self):
        # Test the full workflow with mock data
        self.defense_config.model_params = {"random_state": 42}
        score_dict = self.defense_config(
            data=self.data,
            model_file=self.model_file,
            test_predictions_file=self.test_predictions_file,
            train_predictions_file=self.train_predictions_file,
            score_file=self.score_file,
        )
        self.assertIsInstance(score_dict, dict)
        self.assertIn("training_time", score_dict)
        self.assertIn("training_score_time", score_dict)
        self.assertIn("prediction_time", score_dict)
        self.assertIn("prediction_score_time", score_dict)

    def test_save_and_load_model(self):
        # Test saving and loading the model
        self.defense_config.model_params = {"random_state": 42}
        self.defense_config(self.data, model_file=self.model_file)
        self.assertTrue(Path(self.model_file).exists())

    def test_save_and_load_predictions(self):
        # Test saving and loading predictions
        self.defense_config.model_params = {"random_state": 42}
        self.defense_config(
            data=self.data,
            test_predictions_file=self.test_predictions_file,
            train_predictions_file=self.train_predictions_file,
        )
        self.assertTrue(Path(self.test_predictions_file).exists())
        self.assertTrue(Path(self.train_predictions_file).exists())

    def test_hash_function(self):
        # Test the hash function for DefenseConfig
        hash_value = hash(self.defense_config)
        self.assertIsInstance(hash_value, int)

    def test_supported_defense_types(self):
        # Test supported defense types
        supported_types = [
            "detector",
            "preprocessor",
            "postprocessor",
            "trainer",
            "regularizer",
            "transformer",
        ]
        self.assertIn("postprocessor", supported_types)
        self.assertNotIn("unsupported_type", supported_types)
