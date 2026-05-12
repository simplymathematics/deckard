import pytest
import torch
import numpy as np
from deckard.data.fairness_pytorch import FairlearnPytorchDataConfig


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10):
        self.X = torch.randn(n, 3)
        self.y = torch.randint(0, 2, (n,))
        self._sensitive = np.array([i % 2 for i in range(n)])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self._sensitive[idx]


def test_sensitive_extraction():
    ds = DummyDataset(20)
    config = FairlearnPytorchDataConfig(
        dataset_name="test.DummyDataset", sensitive_columns=["_sensitive"]
    )
    config._X = ds  # Pass the dataset, not just the tensor
    config._y = ds.y
    config._sensitive = ds._sensitive
    config.train_size = 0.5
    config.test_size = 0.5
    config.random_state = 42
    config._sample()
    assert hasattr(config, "_sensitive_train")
    assert hasattr(config, "_sensitive_test")
    assert len(config._sensitive_train) + len(config._sensitive_test) == 20
    # Ensure concatenation works for numpy arrays or pandas Series
    all_sensitive = np.concatenate(
        [np.array(config._sensitive_train), np.array(config._sensitive_test)]
    )
    assert set(all_sensitive) <= {0, 1}
