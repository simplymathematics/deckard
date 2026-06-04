from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from deckard.frameworks.pytorch.attack import PytorchAttackConfig

pytest.importorskip("torch")

# TOOD Create PytorchAttackConfig and test it here rather than a generic one.


class TestPytorchAttackConfigPrep:
    def test_prepare_features_for_attack_preserves_tensors_and_coerces_pandas(self):
        torch = pytest.importorskip("torch")
        cfg = PytorchAttackConfig(name="art.attacks.evasion.FastGradientMethod")

        tensor = torch.tensor([[1.0, 2.0]])
        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        series = pd.Series([1.0, 2.0])

        assert cfg._prepare_features_for_attack(tensor) is tensor
        assert cfg._prepare_features_for_attack(frame).dtype == np.dtype("float32")
        assert cfg._prepare_features_for_attack(series).dtype == np.dtype("float32")

    def test_prepare_labels_for_attack_preserves_tensors_and_coerces_pandas(self):
        torch = pytest.importorskip("torch")
        cfg = PytorchAttackConfig(name="art.attacks.evasion.FastGradientMethod")

        tensor = torch.tensor([1, 0])
        frame = pd.DataFrame({"label": [1, 0]})
        series = pd.Series([1, 0])

        assert cfg._prepare_labels_for_attack(tensor) is tensor
        assert np.array_equal(cfg._prepare_labels_for_attack(frame), frame.values)
        assert np.array_equal(cfg._prepare_labels_for_attack(series), series.values)

    def test_prepare_features_for_art_covers_tensor_numpy_and_fallback_inputs(self):
        cfg = PytorchAttackConfig(name="art.attacks.evasion.FastGradientMethod")

        float_tensor = object()
        int_tensor = object()
        float_array = np.array([[1.0, 2.0]], dtype=np.float64)
        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        series = pd.Series([1.0, 2.0])

        with (
            patch(
                "deckard.frameworks.pytorch.attack.is_tensor",
                side_effect=lambda value: value in {float_tensor, int_tensor},
            ),
            patch(
                "deckard.frameworks.pytorch.attack.tensor_to_numpy",
                side_effect=[
                    np.array([[1.0, 2.0]], dtype=np.float64),
                    np.array([[1, 2]], dtype=np.int64),
                ],
            ),
        ):
            tensor_result = cfg._prepare_features_for_art(float_tensor)
            int_result = cfg._prepare_features_for_art(int_tensor)

        array_result = cfg._prepare_features_for_art(float_array)
        frame_result = cfg._prepare_features_for_art(frame)
        series_result = cfg._prepare_features_for_art(series)
        list_result = cfg._prepare_features_for_art([[1, 2], [3, 4]])

        assert tensor_result.dtype == np.dtype("float32")
        assert int_result.dtype == np.dtype("int64")
        assert array_result.dtype == np.dtype("float32")
        assert frame_result.dtype == np.dtype("float32")
        assert series_result.dtype == np.dtype("float32")
        assert np.array_equal(list_result, np.asarray([[1, 2], [3, 4]]))

    def test_prepare_labels_for_art_covers_tensor_and_pandas_inputs(self):
        cfg = PytorchAttackConfig(name="art.attacks.evasion.FastGradientMethod")

        tensor = object()
        frame = pd.DataFrame({"label": [1, 0]})
        series = pd.Series([1, 0])
        labels = [1, 0]

        with (
            patch(
                "deckard.frameworks.pytorch.attack.is_tensor",
                side_effect=lambda value: value is tensor,
            ),
            patch(
                "deckard.frameworks.pytorch.attack.tensor_to_numpy",
                return_value=np.array([1, 0]),
            ),
        ):
            tensor_result = cfg._prepare_labels_for_art(tensor)

        assert np.array_equal(tensor_result, np.array([1, 0]))
        assert np.array_equal(cfg._prepare_labels_for_art(frame), frame.values)
        assert np.array_equal(cfg._prepare_labels_for_art(series), series.values)
        assert cfg._prepare_labels_for_art(labels) == labels
