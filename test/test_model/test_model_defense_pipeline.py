import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf
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


def test_defense_behavior_defaults_signature_and_apply_to_paths(monkeypatch):
    defense = DefenseConfig.__new__(DefenseConfig)
    defense.model_type = None
    defense.classifier = True
    defense.model_params = {}
    defense.probability = False
    defense.alias = ""
    defense.defense_name = "art.defences.postprocessor.HighConfidence"
    defense.defense_params = None
    defense.score_dict = None
    defense._target_ = None

    defense.__post_init__()

    assert defense._model is None
    assert defense.score_dict == {}
    assert defense._target_ == "deckard.DefenseConfig"
    assert defense.defense_params == {}

    defense.defense_params = OmegaConf.create(
        {"outer": {"inner": [1, {"value": 2}]}, "flat": [3, 4]},
    )
    signature = defense._defense_signature()
    assert signature[0] == "art.defences.postprocessor.HighConfidence"
    assert isinstance(signature[1], tuple)

    with pytest.raises(ValueError, match="Model is not fitted yet"):
        defense.get_model()

    with pytest.raises(ValueError, match="estimator must be provided"):
        defense.apply_to(estimator=None, data=object())

    defense._model_config = SimpleNamespace(_model=None)
    monkeypatch.setattr(
        DefenseConfig,
        "apply_defense",
        lambda self, data: {"data": data, "model": self._model},
    )
    estimator = object()
    result = defense.apply_to(estimator=estimator, data="payload")
    assert result == {"data": "payload", "model": estimator}
    assert defense._model_config._model is estimator


def test_parse_defense_name_and_get_art_class_edge_paths(monkeypatch):
    defense = DefenseConfig(
        defense_name=None,
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
    )
    defense_type, subtype, defense_class = defense.parse_defense_name()
    assert defense_type is None
    assert subtype is None
    assert defense_class is None

    defense.defense_name = "short.Class"
    with pytest.raises(ImportError, match="Could not parse defense type"):
        defense.parse_defense_name()

    defense.defense_name = "art.defences.postprocessor.Missing"
    monkeypatch.setattr(
        "deckard.model.defend.resolve_class",
        lambda name: (_ for _ in ()).throw(AttributeError(name)),
    )
    with pytest.raises(ImportError, match="Could not import defense class"):
        defense.parse_defense_name()

    defense = DefenseConfig(
        defense_name="art.defences.postprocessor.HighConfidence",
        model_type=None,
        classifier=True,
    )
    with pytest.raises(ValueError, match="model_type must be set"):
        defense.get_art_class(
            SimpleNamespace(X_train=np.zeros((2, 3)), y_train=[0, 1]),
        )

    import deckard.model.defend as defend_module

    custom_art = type("CustomArt", (), {})
    monkeypatch.setitem(
        defend_module.classifier_dict,
        "LogisticRegression",
        custom_art,
    )
    defense.model_type = "sklearn.linear_model.LogisticRegression"
    art_class, init_params = defense.get_art_class(
        SimpleNamespace(X_train=np.zeros((2, 3)), y_train=[0, 1]),
    )
    assert art_class is custom_art
    assert init_params["input_shape"] == (3,)
    assert init_params["nb_classes"] == 2
    assert init_params["preprocessing"] is None


def test_pipeline_coerce_plugin_context_and_stage_helpers(monkeypatch):
    pipeline = DefensePipelineConfig.__new__(DefensePipelineConfig)
    pipeline.defenses = []
    pipeline.plugins = []
    pipeline.score_dict = None
    pipeline._target_ = None
    pipeline.__post_init__()
    assert pipeline.score_dict == {}
    assert pipeline._target_ == "deckard.model.DefensePipelineConfig"

    assert not DefensePipelineConfig._is_pipeline_target(7)
    assert not DefensePipelineConfig._looks_like_single_defense_spec({"defenses": []})
    assert DefensePipelineConfig._looks_like_single_defense_spec(
        {"_target_": "pkg.OtherDefense"},
    )

    class ApplyObj:
        def apply_to(self, estimator, data):
            return estimator

    assert len(DefensePipelineConfig.coerce(ApplyObj()).defenses) == 1
    assert (
        len(
            DefensePipelineConfig.coerce(
                [{"defense_name": "art.defences.postprocessor.HighConfidence"}],
            ).defenses,
        )
        == 1
    )
    assert (
        len(
            DefensePipelineConfig.coerce(
                {
                    "_target_": "deckard.model.DefensePipelineConfig",
                    "defenses": [],
                },
            ).defenses,
        )
        == 0
    )
    with pytest.raises(TypeError, match="Defense config must be"):
        DefensePipelineConfig.coerce(3)

    monkeypatch.setattr(
        "deckard.model.defend.coerce_config",
        lambda obj: (
            [{"defense_name": "art.defences.postprocessor.HighConfidence"}]
            if obj == "legacy-list"
            else obj
        ),
    )
    assert len(DefensePipelineConfig.coerce("legacy-list").defenses) == 1

    resolved_plugins = []

    def _resolve_plugin(name):
        def _factory(**kwargs):
            resolved_plugins.append((name, kwargs))
            return {"name": name, **kwargs}

        return _factory

    monkeypatch.setattr("deckard.model.defend.resolve_class", _resolve_plugin)
    plugin_pipeline = DefensePipelineConfig(defenses=[])
    assert plugin_pipeline._instantiate_plugin(
        {"name": "pkg.Plugin", "flag": True},
    ) == {"name": "pkg.Plugin", "flag": True}
    with pytest.raises(ValueError, match="must include 'name' or '_target_'"):
        plugin_pipeline._instantiate_plugin({"flag": True})
    assert plugin_pipeline._instantiate_plugin("pkg.Plugin") == {"name": "pkg.Plugin"}

    class LocalPlugin:
        def __init__(self):
            self.tag = "local"

    assert isinstance(plugin_pipeline._instantiate_plugin(LocalPlugin), LocalPlugin)
    marker = object()
    assert plugin_pipeline._instantiate_plugin(marker) is marker

    plugin_pipeline.plugins = "bad"
    plugin_pipeline._plugin_objects = None
    with pytest.raises(TypeError, match="plugins must be a list"):
        plugin_pipeline._get_plugins()

    plugin_pipeline.plugins = [
        {"name": "pkg.Plugin", "alpha": 1},
        "pkg.Plugin",
        LocalPlugin,
        marker,
    ]
    plugin_pipeline._plugin_objects = None
    assert len(plugin_pipeline._get_plugins()) == 4

    plugin_pipeline._run_plugin_hook = lambda hook_name, **kwargs: [
        " tuned ",
        {"stage": "attack"},
        {"defense_stage": "train"},
        None,
    ]
    assert plugin_pipeline.resolve_stage() == "train"


def test_pipeline_single_defense_coercion_and_context_inheritance(monkeypatch):
    pipeline = DefensePipelineConfig(defenses=[])

    class CustomDefense:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FairDefense:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _resolve(name):
        if name == "pkg.CustomDefense":
            return CustomDefense
        if name == "deckard.model.fairness.FairlearnDefenseConfig":
            return FairDefense
        raise RuntimeError(name)

    monkeypatch.setattr("deckard.model.defend.resolve_class", _resolve)

    coerced_target = pipeline._coerce_single_defense(
        {"_target_": "pkg.CustomDefense", "x": 1},
    )
    assert isinstance(coerced_target, CustomDefense)
    assert coerced_target.kwargs == {"x": 1}

    coerced_fair = pipeline._coerce_single_defense(
        {"defense_name": "fairlearn.reductions.ExponentiatedGradient", "eps": 0.1},
    )
    assert isinstance(coerced_fair, FairDefense)
    assert coerced_fair.kwargs["eps"] == 0.1

    monkeypatch.setattr(
        "deckard.model.defend.resolve_class",
        lambda name: (_ for _ in ()).throw(RuntimeError(name)),
    )
    coerced_plain = pipeline._coerce_single_defense(
        {"defense_name": "fairlearn.reductions.ExponentiatedGradient"},
    )
    assert isinstance(coerced_plain, DefenseConfig)
    with pytest.raises(TypeError, match="Unsupported defense specification"):
        pipeline._coerce_single_defense(3)

    defense = SimpleNamespace(
        model_type=None,
        classifier=None,
        model_params=None,
        probability=False,
    )
    estimator = SimpleNamespace(
        model=SimpleNamespace(
            _estimator_type="classifier",
            get_params=lambda: {"depth": 3},
            predict_proba=lambda x: x,
        ),
    )
    pipeline._inherit_model_context(defense, estimator)
    assert defense.model_type.endswith("SimpleNamespace")
    assert defense.classifier is True
    assert defense.model_params == {"depth": 3}
    assert defense.probability is True

    reg_defense = SimpleNamespace(
        model_type="",
        classifier=None,
        model_params={},
        probability=False,
    )
    reg_estimator = SimpleNamespace(
        _estimator_type="regressor",
        get_params=lambda: {"alpha": 1},
    )
    pipeline._inherit_model_context(reg_defense, reg_estimator)
    assert reg_defense.classifier is False
    assert reg_defense.model_params == {"alpha": 1}

    assert pipeline.normalize_defenses(None) == []
    assert (
        len(
            pipeline.normalize_defenses(
                {"defense_name": "art.defences.postprocessor.HighConfidence"},
            ),
        )
        == 1
    )


def test_pipeline_apply_validation_and_elapsed_fallback(monkeypatch):
    pipeline = DefensePipelineConfig(defenses=[])

    with pytest.raises(ValueError, match="estimator must be provided"):
        pipeline.apply(estimator=None, data=object())

    estimator = object()
    assert pipeline.apply(estimator=estimator, data=object()) is estimator

    bad_pipeline = DefensePipelineConfig(defenses=[SimpleNamespace(apply_to=1)])
    with pytest.raises(TypeError, match="must implement apply_to"):
        bad_pipeline.apply(estimator=object(), data=object())

    hook_calls = []

    class TimelessDefense:
        defense_name = "custom.timeless"
        defense_application_time = None
        data = None
        model_type = None
        classifier = None
        model_params = None
        probability = False

        def apply_to(self, estimator, data):
            _ = data
            hook_calls.append("apply")
            return {"wrapped": estimator}

    timed_pipeline = DefensePipelineConfig(defenses=[TimelessDefense()])
    monkeypatch.setattr(
        timed_pipeline,
        "_run_plugin_hook",
        lambda hook_name, **kwargs: hook_calls.append(hook_name) or [],
    )
    monkeypatch.setattr(
        "deckard.model.defend.time.process_time",
        iter([10.0, 12.5]).__next__,
    )

    result = timed_pipeline.apply(estimator="model", data="payload")
    assert result == {"wrapped": "model"}
    assert timed_pipeline.defense_application_time == 2.5
    assert timed_pipeline.defenses[0].data == "payload"
    assert "before_apply_defense" in hook_calls
    assert "after_apply_defense" in hook_calls
