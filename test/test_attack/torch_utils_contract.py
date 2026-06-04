"""Shared test contract for torch utility helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TorchUtilsContract:
    """Reusable test suite for deckard.frameworks.pytorch.torch_utils."""

    @classmethod
    def setup_class(cls):
        try:
            import torch

            cls.torch = torch
        except ImportError:
            cls.torch = None

    def _skip_if_no_torch(self):
        if self.torch is None:
            pytest.skip("torch not installed")

    def test_is_tensor_true_for_tensor(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import is_tensor

        assert is_tensor(self.torch.tensor([1.0]))

    def test_is_tensor_false_for_numpy(self):
        from deckard.frameworks.pytorch.torch_utils import is_tensor

        assert not is_tensor(np.array([1.0]))

    def test_is_torch_model_true(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import is_torch_model

        model = self.torch.nn.Linear(2, 2)
        assert is_torch_model(model)

    def test_is_torch_model_false_for_sklearn(self):
        from sklearn.linear_model import LogisticRegression

        from deckard.frameworks.pytorch.torch_utils import is_torch_model

        assert not is_torch_model(LogisticRegression())

    def test_is_dataloader_true(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import is_dataloader

        ds = TensorDataset(self.torch.randn(4, 2))
        dl = DataLoader(ds, batch_size=2)
        assert is_dataloader(dl)

    def test_is_dataloader_false_for_list(self):
        from deckard.frameworks.pytorch.torch_utils import is_dataloader

        assert not is_dataloader([1, 2, 3])

    def test_tensor_to_numpy_converts(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        t = self.torch.tensor([1.0, 2.0, 3.0])
        arr = tensor_to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_tensor_to_numpy_with_dtype(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        t = self.torch.tensor([1.0, 2.0])
        arr = tensor_to_numpy(t, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_tensor_to_numpy_passthrough_non_tensor(self):
        from deckard.frameworks.pytorch.torch_utils import tensor_to_numpy

        arr = np.array([1, 2])
        result = tensor_to_numpy(arr)
        assert result is arr

    def test_get_torch_model_device_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        model = self.torch.nn.Linear(2, 2)
        device = get_torch_model_device(model)
        assert str(device.type) == "cpu"

    def test_get_torch_model_device_non_torch_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        device = get_torch_model_device(object())
        assert str(device.type) == "cpu"

    def test_get_torch_model_device_model_with_no_parameters_returns_cpu(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import get_torch_model_device

        class EmptyModule(self.torch.nn.Module):
            def forward(self, x):
                return x

        model = EmptyModule()
        device = get_torch_model_device(model)
        assert str(device.type) == "cpu"

    def test_build_torch_art_model_raises_when_torch_flag_disabled(self):
        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        with patch("deckard.frameworks.pytorch.torch_utils.HAS_TORCH", False):
            with pytest.raises(ImportError):
                build_torch_art_model(object(), object())

    def test_collect_subset_raises_when_torch_flag_disabled(self):
        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        with patch("deckard.frameworks.pytorch.torch_utils.HAS_TORCH", False):
            with pytest.raises(ImportError):
                collect_subset_from_dataloader(object(), n=2)

    def test_build_torch_art_model_dataloader_tuple_batch_input_shape(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        class FakePyTorchClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.input_shape = kwargs["input_shape"]
                self._device = None
                self._model = kwargs["model"]
                self.preprocessing = type("P", (), {"_device": None})()
                self.preprocessing_operations = [
                    type("Op", (), {"_device": None})(),
                ]

        torch_mod = self.torch

        class TinyModel(torch_mod.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch_mod.nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        X = self.torch.randn(8, 3)
        y = self.torch.randint(0, 2, (8,))
        dl = DataLoader(TensorDataset(X, y), batch_size=4)
        data = type("D", (), {"X_train": dl, "y_train": y.numpy()})()

        with patch(
            "art.estimators.classification.PyTorchClassifier",
            FakePyTorchClassifier,
        ):
            model = TinyModel()
            estimator = build_torch_art_model(model, data)
        assert estimator.input_shape == (3,)

    def test_build_torch_art_model_dataloader_tensor_batch_input_shape(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, Dataset

        from deckard.frameworks.pytorch.torch_utils import build_torch_art_model

        class TensorOnlyDataset(Dataset):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx]

        class FakePyTorchClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.input_shape = kwargs["input_shape"]
                self._device = None
                self._model = kwargs["model"]
                self.preprocessing = type("P", (), {"_device": None})()
                self.preprocessing_operations = [
                    type("Op", (), {"_device": None})(),
                ]

        torch_mod = self.torch

        class TinyModel(torch_mod.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch_mod.nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        X = self.torch.randn(8, 3)
        y = self.torch.randint(0, 2, (8,))
        dl = DataLoader(TensorOnlyDataset(X), batch_size=4)
        data = type("D", (), {"X_train": dl, "y_train": y.numpy()})()

        with patch(
            "art.estimators.classification.PyTorchClassifier",
            FakePyTorchClassifier,
        ):
            model = TinyModel()
            estimator = build_torch_art_model(model, data)
        assert estimator.input_shape == (3,)

    def test_collect_subset_from_dataloader(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        X = self.torch.randn(10, 3)
        y = self.torch.randint(0, 2, (10,))
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=4)
        x_sub, y_sub = collect_subset_from_dataloader(dl, n=4)
        assert x_sub.shape[0] == 4
        assert y_sub.shape[0] == 4

    def test_collect_subset_clips_to_dataset_len(self):
        self._skip_if_no_torch()
        from torch.utils.data import DataLoader, TensorDataset

        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        X = self.torch.randn(5, 2)
        y = self.torch.randint(0, 2, (5,))
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=5)
        x_sub, y_sub = collect_subset_from_dataloader(dl, n=100)
        assert x_sub.shape[0] == 5

    def test_collect_subset_raises_for_non_dataloader(self):
        self._skip_if_no_torch()
        from deckard.frameworks.pytorch.torch_utils import (
            collect_subset_from_dataloader,
        )

        with pytest.raises(TypeError):
            collect_subset_from_dataloader([1, 2, 3], n=2)
