from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from deckard.model.base import ModelConfig
from deckard.model.defense.base import DefenseConfig, DefensePipelineConfig

DummyDataConfig = SimpleNamespace


class TestRetrainingDefensePipeline:
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

        pipeline.apply(estimator=object(), data=object())

        assert order == [
            "art.defences.postprocessor.GaussianNoise",
            "art.defences.trainer.AdversarialTrainerMadryPGD",
        ]

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
        model.train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.trainer.AdversarialTrainerMadryPGD",
            defense_params={"nb_epochs": 1, "batch_size": 4, "max_iter": 1},
        )

        with pytest.raises(ValueError):
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
            Path(__file__).resolve().parents[3]
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

        assert hasattr(defended, "predict")
        assert hasattr(defended, "model")
        assert defense.defense_training_time is None
        assert defense.defense_application_time is not None

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

        assert isinstance(defended, PyTorchClassifier)
        assert defense.defense_training_time is None
        assert defense.defense_application_time is not None

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
        model.train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.detector.evasion.BinaryInputDetector",
            defense_params={},
        )

        with pytest.raises(ValueError):
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

        assert hasattr(defended, "predict")
        assert hasattr(defended, "model")
        assert hasattr(defended, "_deckard_evasion_detector")
        assert defense.defense_application_time is not None

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

        assert isinstance(defended, PyTorchClassifier)
        assert hasattr(defended, "_deckard_evasion_detector")
        assert defense.defense_application_time is not None

    def test_transformer_defense_name_parses_supported_subtype(self):
        defense = DefenseConfig(
            defense_name="art.defences.transformer.evasion.DefensiveDistillation",
            defense_params={"batch_size": 8, "nb_epochs": 1},
        )
        defense_type, defense_subtype, _ = defense.parse_defense_name()

        assert defense_type == "transformer"
        assert defense_subtype == "evasion"

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
        model.train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.transformer.evasion.DefensiveDistillation",
            defense_params={"batch_size": 8, "nb_epochs": 1},
        )

        with pytest.raises(ValueError):
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
        model.train(data.X_train, data.y_train)
        defense = DefenseConfig(
            defense_name="art.defences.transformer.poisoning.NeuralCleanse",
            defense_params={},
        )

        with pytest.raises(ValueError):
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
            Path(__file__).resolve().parents[3]
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

        assert hasattr(defended, "predict")
        assert hasattr(defended, "model")
        assert defense.defense_application_time is not None

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
            Path(__file__).resolve().parents[3]
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

        with pytest.raises(ValueError):
            defense.apply_to(estimator=TinyLinear(), data=data)


class _OrderTrackingDefense:
    def __init__(self, defense_name, order):
        self.defense_name = defense_name
        self._order = order
        self.defense_application_time = 0.0

    def apply_to(self, estimator, data):
        _ = data
        self._order.append(self.defense_name)
        return estimator
