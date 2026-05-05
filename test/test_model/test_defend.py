import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
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


class _OrderTrackingDefense:
    def __init__(self, defense_name, order):
        self.defense_name = defense_name
        self._order = order
        self.defense_application_time = 0.0

    def apply_to(self, estimator, data):
        _ = data
        self._order.append(self.defense_name)
        return estimator


class TestRetrainingDefensePipeline(unittest.TestCase):
    def test_retraining_defense_is_reordered_last_with_warning(self):
        order = []
        retraining = _OrderTrackingDefense(
            "art.defences.trainer.AdversarialTrainerMadryPGD",
            order,
        )
        postprocessor = _OrderTrackingDefense(
            "art.defences.postprocessor.GaussianNoise",
            order,
        )
        pipeline = DefensePipelineConfig(defenses=[retraining, postprocessor])

        with self.assertWarns(RuntimeWarning):
            pipeline.apply(estimator=object(), data=object())

        self.assertEqual(
            order,
            [
                "art.defences.postprocessor.GaussianNoise",
                "art.defences.trainer.AdversarialTrainerMadryPGD",
            ],
        )

    def test_retraining_rejects_non_neural_network_models(self):
        data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(20, 4)),
            y_train=pd.Series(np.random.randint(0, 2, size=20)),
            X_test=pd.DataFrame(np.random.rand(8, 4)),
            y_test=pd.Series(np.random.randint(0, 2, size=8)),
        )
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 20},
        )
        model._train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.trainer.AdversarialTrainerMadryPGD",
            defense_params={"nb_epochs": 1, "batch_size": 4, "max_iter": 1},
        )

        with self.assertRaises(ValueError):
            defense.apply_to(estimator=model.get_model(), data=data)

    def test_real_adversarial_retraining_executes_with_pytorch_model(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        data = SimpleNamespace(
            X_train=torch.rand(16, 3, dtype=torch.float32),
            y_train=torch.randint(0, 2, (16,), dtype=torch.long),
            X_test=torch.rand(8, 3, dtype=torch.float32),
            y_test=torch.randint(0, 2, (8,), dtype=torch.long),
        )

        config_path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "pytorch"
            / "config"
            / "defense"
            / "adversarial_retraining.yaml"
        )
        defense_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        defense = DefenseConfig(
            model_type="torch.nn.Linear",
            classifier=True,
            model_params={"in_features": 3, "out_features": 2},
            **defense_cfg,
        )

        defended = defense.apply_to(estimator=TinyLinear(), data=data)

        self.assertTrue(hasattr(defended, "predict"))
        self.assertTrue(hasattr(defended, "model"))
        self.assertIsNone(defense.defense_training_time)
        self.assertIsNotNone(defense.defense_application_time)

    def test_retraining_handles_existing_art_torch_wrapper(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")
        PyTorchClassifier = pytest.importorskip(
            "art.estimators.classification",
        ).PyTorchClassifier

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        data = SimpleNamespace(
            X_train=torch.rand(16, 3, dtype=torch.float32),
            y_train=torch.randint(0, 2, (16,), dtype=torch.long),
            X_test=torch.rand(8, 3, dtype=torch.float32),
            y_test=torch.randint(0, 2, (8,), dtype=torch.long),
        )

        wrapped_model = TinyLinear()
        wrapped = PyTorchClassifier(
            model=wrapped_model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(wrapped_model.parameters(), lr=0.01),
            input_shape=(3,),
            nb_classes=2,
            clip_values=(0.0, 1.0),
            device_type=("gpu" if torch.cuda.is_available() else "cpu"),
        )
        defense = DefenseConfig(
            defense_name="art.defences.trainer.AdversarialTrainerMadryPGD",
            defense_params={
                "nb_epochs": 1,
                "batch_size": 8,
                "eps": 0.2,
                "eps_step": 0.1,
                "max_iter": 1,
                "num_random_init": 1,
            },
            model_type=None,
            classifier=True,
        )

        defended = defense.apply_to(estimator=wrapped, data=data)

        self.assertIsInstance(defended, PyTorchClassifier)
        self.assertIsNone(defense.defense_training_time)
        self.assertIsNotNone(defense.defense_application_time)

    def test_binary_input_detector_rejects_non_neural_network_models(self):
        data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(20, 4)),
            y_train=pd.Series(np.random.randint(0, 2, size=20)),
            X_test=pd.DataFrame(np.random.rand(8, 4)),
            y_test=pd.Series(np.random.randint(0, 2, size=8)),
        )
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 20},
        )
        model._train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.detector.evasion.BinaryInputDetector",
            defense_params={},
        )

        with self.assertRaises(ValueError):
            defense.apply_to(estimator=model.get_model(), data=data)

    def test_binary_input_detector_handles_raw_torch_model(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        data = SimpleNamespace(
            X_train=torch.rand(16, 3, dtype=torch.float32),
            y_train=torch.randint(0, 2, (16,), dtype=torch.long),
            X_test=torch.rand(8, 3, dtype=torch.float32),
            y_test=torch.randint(0, 2, (8,), dtype=torch.long),
        )

        defense = DefenseConfig(
            defense_name="art.defences.detector.evasion.BinaryInputDetector",
            defense_params={},
            model_type="torch.nn.Linear",
            classifier=True,
            model_params={"in_features": 3, "out_features": 2},
        )

        defended = defense.apply_to(estimator=TinyLinear(), data=data)

        self.assertTrue(hasattr(defended, "predict"))
        self.assertTrue(hasattr(defended, "model"))
        self.assertTrue(hasattr(defended, "_deckard_evasion_detector"))
        self.assertIsNotNone(defense.defense_application_time)

    def test_binary_input_detector_handles_existing_art_wrapper(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")
        PyTorchClassifier = pytest.importorskip(
            "art.estimators.classification",
        ).PyTorchClassifier

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        data = SimpleNamespace(
            X_train=torch.rand(16, 3, dtype=torch.float32),
            y_train=torch.randint(0, 2, (16,), dtype=torch.long),
            X_test=torch.rand(8, 3, dtype=torch.float32),
            y_test=torch.randint(0, 2, (8,), dtype=torch.long),
        )

        wrapped_model = TinyLinear()
        wrapped = PyTorchClassifier(
            model=wrapped_model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(wrapped_model.parameters(), lr=0.01),
            input_shape=(3,),
            nb_classes=2,
            clip_values=(0.0, 1.0),
            device_type=("gpu" if torch.cuda.is_available() else "cpu"),
        )
        defense = DefenseConfig(
            defense_name="art.defences.detector.evasion.BinaryInputDetector",
            defense_params={},
            model_type=None,
            classifier=True,
        )

        defended = defense.apply_to(estimator=wrapped, data=data)

        self.assertIsInstance(defended, PyTorchClassifier)
        self.assertTrue(hasattr(defended, "_deckard_evasion_detector"))
        self.assertIsNotNone(defense.defense_application_time)

    def test_transformer_defense_name_parses_supported_subtype(self):
        defense = DefenseConfig(
            defense_name="art.defences.transformer.evasion.DefensiveDistillation",
            defense_params={"batch_size": 8, "nb_epochs": 1},
        )
        defense_type, defense_subtype, _ = defense.parse_defense_name()

        self.assertEqual(defense_type, "transformer")
        self.assertEqual(defense_subtype, "evasion")

    def test_defensive_distillation_rejects_non_neural_network_models(self):
        data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(20, 4)),
            y_train=pd.Series(np.random.randint(0, 2, size=20)),
            X_test=pd.DataFrame(np.random.rand(8, 4)),
            y_test=pd.Series(np.random.randint(0, 2, size=8)),
        )
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 20},
        )
        model._train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.transformer.evasion.DefensiveDistillation",
            defense_params={"batch_size": 8, "nb_epochs": 1},
        )

        with self.assertRaises(ValueError):
            defense.apply_to(estimator=model.get_model(), data=data)

    def test_neural_cleanse_rejects_non_neural_network_models(self):
        data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(20, 4)),
            y_train=pd.Series(np.random.randint(0, 2, size=20)),
            X_test=pd.DataFrame(np.random.rand(8, 4)),
            y_test=pd.Series(np.random.randint(0, 2, size=8)),
        )
        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 20},
        )
        model._train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.transformer.poisoning.NeuralCleanse",
            defense_params={},
        )

        with self.assertRaises(ValueError):
            defense.apply_to(estimator=model.get_model(), data=data)

    def test_real_defensive_distillation_executes_with_pytorch_model(self):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        data = SimpleNamespace(
            X_train=torch.rand(16, 3, dtype=torch.float32),
            y_train=torch.randint(0, 2, (16,), dtype=torch.long),
            X_test=torch.rand(8, 3, dtype=torch.float32),
            y_test=torch.randint(0, 2, (8,), dtype=torch.long),
        )

        config_path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "pytorch"
            / "config"
            / "defense"
            / "defensive_distillation.yaml"
        )
        defense_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        defense = DefenseConfig(
            model_type="torch.nn.Linear",
            classifier=True,
            model_params={"in_features": 3, "out_features": 2},
            **defense_cfg,
        )

        defended = defense.apply_to(estimator=TinyLinear(), data=data)

        self.assertTrue(hasattr(defended, "predict"))
        self.assertTrue(hasattr(defended, "model"))
        self.assertIsNotNone(defense.defense_application_time)

    def test_real_neural_cleanse_reports_backend_incompatibility_for_pytorch_model(
        self,
    ):
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        class TinyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        data = SimpleNamespace(
            X_train=torch.rand(16, 3, dtype=torch.float32),
            y_train=torch.randint(0, 2, (16,), dtype=torch.long),
            X_test=torch.rand(8, 3, dtype=torch.float32),
            y_test=torch.randint(0, 2, (8,), dtype=torch.long),
        )

        config_path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "pytorch"
            / "config"
            / "defense"
            / "neural_cleanse.yaml"
        )
        defense_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        defense = DefenseConfig(
            model_type="torch.nn.Linear",
            classifier=True,
            model_params={"in_features": 3, "out_features": 2},
            **defense_cfg,
        )

        with self.assertRaises(ValueError):
            defense.apply_to(estimator=TinyLinear(), data=data)
