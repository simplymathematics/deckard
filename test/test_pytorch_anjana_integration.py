"""Integration tests for torch data with anjana-style scoring paths."""

from pathlib import Path

import pytest


from deckard.data.pytorch import PytorchDataConfig
from deckard.model.pytorch import PytorchModelConfig

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PYTORCH_DIR = ROOT / "examples" / "pytorch"


def _make_torch_data():
    X = torch.randn(60, 8)
    y = torch.randint(0, 2, (60,))
    data = PytorchDataConfig(
        dataset_name="torch.utils.data.TensorDataset",
        train_size=40,
        test_size=20,
        classifier=True,
        random_state=42,
        data_params={"_args_": [X, y], "batch_size": 16},
    )
    data()
    return data


def test_torch_data_supports_anjana_style_data_scoring_payload_shape():
    data = _make_torch_data()
    # Anjana score configs expect y_true/y_pred slots to exist and be coherent in length.
    assert hasattr(data, "y_train") and hasattr(data, "X_train")
    assert len(data.y_train) == len(data.X_train)


def test_torch_model_scores_include_benign_metrics_for_chain_readiness():
    data = _make_torch_data()
    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 8, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 16},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    model(data)

    assert "accuracy" in model.score_dict
    assert "training_time" in model.score_dict
    assert "prediction_time" in model.score_dict


@pytest.mark.skipif(
    not (EXAMPLES_PYTORCH_DIR / ".deckard_rc").exists(),
    reason="examples/pytorch/.deckard_rc not found",
)
def test_examples_pytorch_deckard_rc_exists_for_runtime_env_loading():
    assert (EXAMPLES_PYTORCH_DIR / ".deckard_rc").exists()
