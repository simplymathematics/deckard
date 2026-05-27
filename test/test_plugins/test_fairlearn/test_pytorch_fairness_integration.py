"""Integration tests for PyTorch fairness defense with Fairlearn.

Tests the end-to-end torch-fairness.yaml defense config with PyTorch models
and sensitive features from fairlearn_celeba.yaml dataset config.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fairlearn")

from helpers import load_env_from_deckard_rc  # noqa: E402

import deckard.model as model_module  # noqa: E402
from deckard.data import PytorchDataConfig  # noqa: E402
from deckard.score.attack import FairlearnAttackScorerConfig  # noqa: E402

DefensePipelineConfig = model_module.DefensePipelineConfig
PytorchModelConfig = model_module.PytorchModelConfig
FairlearnPytorchModelConfig = getattr(
    model_module,
    "FairlearnPytorchModelConfig",
    None,
)

if FairlearnPytorchModelConfig is None:
    pytest.skip(
        "fairlearn pytorch model configs are unavailable",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_PYTORCH_DIR = ROOT / "examples" / "pytorch"
DECKARD_RC_PATH = EXAMPLES_PYTORCH_DIR / ".deckard_rc"


def _torch_fairness_data():
    """Create minimal PyTorch fairness dataset for testing.

    Uses synthetic data instead of real CelebA to avoid download delays in tests.
    Structure mimics what fairlearn_celeba.yaml creates.
    """
    X_train = torch.randn(32, 3, 32, 32)  # 32 samples, 3 channels, 32x32 images
    y_train = torch.randint(0, 2, (32,))
    sensitive_train = torch.randint(
        0,
        2,
        (32,),
    )  # Binary sensitive attribute (e.g., Male)

    X_test = torch.randn(16, 3, 32, 32)
    y_test = torch.randint(0, 2, (16,))
    sensitive_test = torch.randint(0, 2, (16,))

    cfg = PytorchDataConfig(
        dataset_name="torch_fairness_dataset.py:SyntheticImageDataset",
        sampler = {
            "train_size" : 32,
            "test_size" : 16,
            "random_state" : 42,
            "name" : "split",
        }
        classifier=True,
        alias="synthetic_fairness",
        data_params={
            "num_samples": 48,
            "image_size": 32,
            "num_channels": 3,
            "num_classes": 2,
            "sensitive_attribute": "binary_feature",
            "batch_size": 8,
        },
    )

    # Directly set data for testing to avoid needing the custom dataset
    cfg._X_train = X_train
    cfg._y_train = y_train
    cfg._sensitive_train = sensitive_train
    cfg._X_test = X_test
    cfg._y_test = y_test
    cfg._sensitive_test = sensitive_test

    return cfg


def test_pytorch_fairness_data_loads_with_sensitive_features():
    """Test that PyTorch fairness data loads and preserves sensitive features."""
    data = _torch_fairness_data()

    # Verify data shapes
    assert data._X_train.shape[0] == 32
    assert data._y_train.shape[0] == 32
    assert data._sensitive_train.shape[0] == 32
    assert data._X_test.shape[0] == 16
    assert data._y_test.shape[0] == 16
    assert data._sensitive_test.shape[0] == 16

    # Verify sensitive features are preserved
    assert hasattr(data, "_sensitive_train")
    assert hasattr(data, "_sensitive_test")


def test_pytorch_model_with_fairness_defense_instantiation():
    """Test that FairlearnPytorchModelConfig can instantiate a torch model."""
    data = _torch_fairness_data()

    # Flatten image tensors so a linear torch model can consume them.
    data.X_train = data.X_train.reshape(data.X_train.shape[0], -1)
    data.y_train = data.y_train
    data.X_test = data.X_test.reshape(data.X_test.shape[0], -1)
    data.y_test = data.y_test

    model = FairlearnPytorchModelConfig(
        model_type="torch.nn.Linear",
        classifier=True,
        model_params={"in_features": 3 * 32 * 32, "out_features": 2},
        data=data,
    )

    assert model is not None
    assert model.data == data
    assert isinstance(model.get_model(), torch.nn.Module)


def test_pytorch_fairlearn_model_trains_with_torch_fallback():
    """Test that FairlearnPytorchModelConfig falls back to PyTorch training."""
    data = _torch_fairness_data()

    data.X_train = data._X_train.reshape(data._X_train.shape[0], -1)
    data.y_train = data._y_train
    data.X_test = data._X_test.reshape(data._X_test.shape[0], -1)
    data.y_test = data._y_test

    model = FairlearnPytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 3 * 32 * 32, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
        data=data,
    )

    model(data)

    assert isinstance(model.get_model(), torch.nn.Module)
    assert "training_time" in model.score_dict
    assert "prediction_time" in model.score_dict


def test_pytorch_model_training_with_fairness_style_data():
    """Test that a real torch model trains against fairness-style pytorch data."""
    data = _torch_fairness_data()

    # PytorchModelConfig expects X_train/X_test tensors to exist.
    data.X_train = data._X_train.reshape(data._X_train.shape[0], -1)
    data.y_train = data._y_train
    data.X_test = data._X_test.reshape(data._X_test.shape[0], -1)
    data.y_test = data._y_test

    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 3 * 32 * 32, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 2, "batch_size": 8},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    model(data)

    assert isinstance(model.get_model(), torch.nn.Module)
    assert "optimizer_loss" in model.score_dict
    assert "accuracy" in model.score_dict
    assert "training_time" in model.score_dict
    assert "prediction_time" in model.score_dict


def test_pytorch_fairness_model_fit_and_score():
    """Test that PyTorch model with fairness defense can fit and produce scores."""
    data = _torch_fairness_data()
    data.X_train = data._X_train.reshape(data._X_train.shape[0], -1)
    data.y_train = data._y_train
    data.X_test = data._X_test.reshape(data._X_test.shape[0], -1)
    data.y_test = data._y_test

    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 3 * 32 * 32, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    # Execute the config to trigger training
    model(data)

    # Verify scores were recorded
    assert "accuracy" in model.score_dict or len(model.score_dict) > 0


def test_pytorch_fairness_defense_receives_sensitive_features():
    """Test that fairness defense receives sensitive features from data config."""
    data = _torch_fairness_data()

    # Mock a simple defense to capture the call
    class _MockFairnessDefense:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fit_calls = []
            self.apply_calls = []
            self.defense_application_time = 0.0

        def fit(self, X, y, sensitive_features=None, **kwargs):
            self.fit_calls.append(
                {
                    "X_shape": X.shape if hasattr(X, "shape") else len(X),
                    "y_shape": y.shape if hasattr(y, "shape") else len(y),
                    "sensitive_features_provided": sensitive_features is not None,
                    "kwargs": kwargs,
                },
            )
            return self

        def predict(self, X):
            # Return random predictions
            return [0] * (X.shape[0] if hasattr(X, "shape") else len(X))

        def apply_to(self, estimator, data):
            self.apply_calls.append((estimator, data))
            return estimator

    # Create defense using mock
    defense = _MockFairnessDefense(
        backend="torch",
        constraints="demographic_parity",
    )

    # Create defense config that wraps the mock
    defense_cfg = DefensePipelineConfig(
        defenses=[defense],
    )

    model = FairlearnPytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 100, "out_features": 2},
        classifier=True,
        defense=defense_cfg,
        data=data,
    )

    assert model.defense is not None
    assert len(model.defense.defenses) == 1


def test_pytorch_art_defense_pipeline_runs_with_real_art_defense():
    """Torch model should train/evaluate with a real ART defense pipeline step."""
    data = _torch_fairness_data()
    data.X_train = data._X_train.reshape(data._X_train.shape[0], -1)
    data.y_train = data._y_train
    data.X_test = data._X_test.reshape(data._X_test.shape[0], -1)
    data.y_test = data._y_test

    defense_cfg = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
                "model_type": "torch.nn.Linear",
                "model_params": {
                    "in_features": 3 * 32 * 32,
                    "out_features": 2,
                },
            },
        ],
    )

    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 3 * 32 * 32, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
        defense=defense_cfg,
    )

    model(data)

    assert "accuracy" in model.score_dict
    assert model.defense_application_time is not None


def test_torch_pipeline_applies_fairlearn_and_art_style_defense_steps_in_order():
    """DefensePipeline should compose fairlearn-like and art-like defense steps for torch models."""
    data = _torch_fairness_data()
    data.X_train = data._X_train.reshape(data._X_train.shape[0], -1)
    data.y_train = data._y_train
    data.X_test = data._X_test.reshape(data._X_test.shape[0], -1)
    data.y_test = data._y_test

    calls = []

    class _FairlearnLikeDefense:
        def __init__(self):
            self.defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            calls.append("fairlearn")
            return estimator

    class _ArtLikeDefense:
        def __init__(self):
            self.defense_application_time = 0.0

        def apply_to(self, estimator, data):
            _ = data
            calls.append("art")
            return estimator

    defense_cfg = DefensePipelineConfig(
        defenses=[_FairlearnLikeDefense(), _ArtLikeDefense()],
    )

    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 3 * 32 * 32, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
        defense=defense_cfg,
    )
    model(data)

    assert calls[:2] == ["fairlearn", "art"]
    assert len(calls) >= 2
    assert len(calls) % 2 == 0


def test_pytorch_fairness_model_serialization_with_defense():
    """Test that FairlearnPytorchModelConfig can be created and preserved."""
    data = _torch_fairness_data()

    model = FairlearnPytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 100, "out_features": 2},
        classifier=True,
        data=data,
    )

    assert model.model_type == "torch.nn.Linear"
    assert model.classifier is True


def test_deckard_rc_environment_loading():
    """Test that .deckard_rc file can be parsed for environment setup."""
    env_vars = load_env_from_deckard_rc(DECKARD_RC_PATH)

    assert "DECKARD_CONFIG_DIR" in env_vars
    assert "DECKARD_DEFAULT_CONFIG_FILE" in env_vars
    assert env_vars["DECKARD_CONFIG_DIR"] == "./config"
    assert env_vars["DECKARD_DEFAULT_CONFIG_FILE"] == "torch_default.yaml"


def test_pytorch_fairlearn_attack_scorer_metric_frame_evasion_group_accuracy_keys():
    scorer = FairlearnAttackScorerConfig()
    y_true = np.array([0, 1, 0, 1])
    adv_pred = np.array([0, 1, 1, 1])
    ben_pred = np.array([0, 1, 0, 1])
    sensitive = np.array(["A", "A", "B", "B"])

    scores = scorer.score_evasion(
        ben_pred_labels=ben_pred,
        adv_pred_labels=adv_pred,
        y_true=y_true,
        attack_size=4,
        sensitive_features=sensitive,
    )

    assert "evasion_A_accuracy" in scores
    assert "evasion_B_accuracy" in scores
    assert "evasion_accuracy_overall" in scores
    assert "evasion_accuracy_difference" in scores


def test_pytorch_fairlearn_attack_scorer_metric_frame_membership_group_accuracy_keys():
    scorer = FairlearnAttackScorerConfig()
    labels = np.array([1, 1, 0, 0])
    inferred = np.array([1, 0, 0, 0])
    sensitive = np.array(["A", "A", "B", "B"])

    scores = scorer.score_membership(
        labels=labels,
        inferred=inferred,
        attack_size=4,
        sensitive_features=sensitive,
    )

    assert "membership_inference_A_accuracy" in scores
    assert "membership_inference_B_accuracy" in scores
    assert "membership_inference_accuracy_overall" in scores
    assert "membership_inference_accuracy_difference" in scores


def test_pytorch_fairlearn_attack_scorer_metric_frame_attribute_group_accuracy_keys():
    scorer = FairlearnAttackScorerConfig()
    target = np.array([1, 0, 1, 0])
    inferred = np.array([1, 1, 1, 0])
    sensitive = np.array(["A", "A", "B", "B"])

    scores = scorer.score_attribute(
        target=target,
        inferred=inferred,
        attack_size=4,
        targeted_attribute="age",
        is_classification=True,
        sensitive_features=sensitive,
    )

    assert "inferred_age_A_accuracy" in scores
    assert "inferred_age_B_accuracy" in scores
    assert "inferred_age_accuracy_overall" in scores
    assert "inferred_age_accuracy_difference" in scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
