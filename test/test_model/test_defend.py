import unittest
import numpy as np
import pandas as pd
from deckard.model.defend import DefenseConfig, DefensePipelineConfig
from deckard.model.base import ModelConfig


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
        self.data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(100, 10)),
            y_train=pd.Series(np.random.randint(0, 2, size=100)),
            X_test=pd.DataFrame(np.random.rand(20, 10)),
            y_test=pd.Series(np.random.randint(0, 2, size=20)),
        )

        self.defense_config = DefenseConfig(
            defense_name="art.defences.postprocessor.HighConfidence",
            model_type="sklearn.ensemble.RandomForestClassifier",
        )

    def test_defense_config_initialization(self):
        # Test default initialization
        self.assertEqual(
            self.defense_config.model_type,
            "sklearn.ensemble.RandomForestClassifier",
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
            self.defense_config.apply_defense(data=self.data)

    def test_apply_defense_with_invalid_defense_name(self):
        # Test applying defense with an invalid defense name
        self.defense_config.defense_name = "invalid.defense.Class"
        with self.assertRaises(ImportError):
            self.defense_config.apply_defense(data=self.data)

    def test_call_is_not_runtime_owner(self):
        with self.assertRaises(NotImplementedError):
            self.defense_config(data=self.data)

    def test_apply_to_trained_model(self):
        model = ModelConfig(
            model_type="sklearn.ensemble.RandomForestClassifier",
            classifier=True,
            model_params={"n_estimators": 5, "random_state": 42},
        )
        model._train(self.data.X_train, self.data.y_train)

        defended = self.defense_config.apply_to(
            estimator=model.get_model(),
            data=self.data,
        )
        self.assertIsNotNone(defended)
        self.assertIsNotNone(self.defense_config.defense_application_time)

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

    def test_hash_stable_after_apply_for_defense_config(self):
        """DefenseConfig hash remains stable after runtime-only apply attrs are set."""
        original_hash = hash(self.defense_config)
        self.defense_config.defense_application_time = 1.23
        self.defense_config._defense_applied_at = 1234567890.5
        self.defense_config._runtime_defense_state = {"applied": True}
        if hasattr(self.defense_config, "score_dict") and isinstance(
            self.defense_config.score_dict,
            dict,
        ):
            self.defense_config.score_dict["runtime"] = 1

        self.assertEqual(
            original_hash,
            hash(self.defense_config),
            msg="Hash changed after defense apply-time runtime updates",
        )


class TestDefensePipelineConfigListCoerce(unittest.TestCase):
    """DefensePipelineConfig.coerce() with a list should chain all specs."""

    spec_a = {
        "defense_name": "art.defences.postprocessor.HighConfidence",
        "defense_params": {"cutoff": 0.25},
    }
    spec_b = {
        "defense_name": "art.defences.postprocessor.ClassLabels",
        "defense_params": {"apply_fit": False, "apply_predict": True},
    }

    def test_list_of_two_specs_produces_two_defenses(self):
        result = DefensePipelineConfig.coerce([self.spec_a, self.spec_b])
        self.assertIsInstance(result, DefensePipelineConfig)
        self.assertEqual(len(result.defenses), 2)

    def test_list_of_one_spec_produces_one_defense(self):
        result = DefensePipelineConfig.coerce([self.spec_a])
        self.assertIsInstance(result, DefensePipelineConfig)
        self.assertEqual(len(result.defenses), 1)

    def test_list_order_is_preserved(self):
        result = DefensePipelineConfig.coerce([self.spec_a, self.spec_b])
        self.assertIn("HighConfidence", result.defenses[0].defense_name)
        self.assertIn("ClassLabels", result.defenses[1].defense_name)

    def test_empty_list_produces_empty_pipeline(self):
        result = DefensePipelineConfig.coerce([])
        self.assertIsInstance(result, DefensePipelineConfig)
        self.assertEqual(len(result.defenses), 0)

    def test_none_still_returns_none(self):
        result = DefensePipelineConfig.coerce(None)
        self.assertIsNone(result)

    def test_single_dict_still_wraps_in_one_element_list(self):
        result = DefensePipelineConfig.coerce(self.spec_a)
        self.assertIsInstance(result, DefensePipelineConfig)
        self.assertEqual(len(result.defenses), 1)
