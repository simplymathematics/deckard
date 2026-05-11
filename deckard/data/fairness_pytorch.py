from dataclasses import dataclass
from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset


from ..score.fairness import DefaultFairlearnClassificationConfig, DefaultFairlearnRegressionConfig
from .fairness import FairlearnDataConfig
from .pytorch import PytorchCustomDataConfig

        




class TinyFairness(Dataset):
    
    """Minimal synthetic dataset for fairness testing."""
    def __init__(self, num_samples=40, n_features=4, random_state=123):
        np.random.seed(random_state)
        self._X = np.random.randn(num_samples, n_features)
        # Binary labels 0/1
        self._y = np.random.randint(0, 2, size=num_samples)
        # Sensitive attribute: two groups 'A' and 'B'
        self._sensitive = np.random.choice(['A', 'B'], size=num_samples)

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        x = torch.tensor(self._X[idx], dtype=torch.float32)
        y = torch.tensor(self._y[idx], dtype=torch.long)  # scalar
        return x, y

class SyntheticImageSensitiveDataset(Dataset):
    def __init__(
        self,
        n_samples=1000,
        image_shape=(3, 64, 64),
        n_classes=2,
        seed=42,
    ):
        torch.manual_seed(seed)

        # Synthetic images
        self.images = torch.rand(n_samples, *image_shape)

        # Task labels
        self.labels = torch.randint(0, n_classes, (n_samples,))

        # Sensitive attributes (not returned by __getitem__)
        # sex: 0=female, 1=male
        sex = torch.randint(0, 2, (n_samples,))

        # gender identity example:
        # 0=woman, 1=man, 2=nonbinary
        gender = torch.randint(0, 3, (n_samples,))

        # Optional combined sensitive tensor
        # shape: [N, 2]
        self._sensitive = torch.stack(
            [sex, gender],
            dim=1
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Model only sees image + target
        return self.images[idx], self.labels[idx]


@dataclass(eq=False)
class FairlearnPytorchDataConfig(FairlearnDataConfig, PytorchCustomDataConfig):
    """Fairlearn-compatible DataConfig for PyTorch Datasets with sensitive features."""

    scorer: Any = None

    def __post_init__(self):
        # Call both parent initializers
        FairlearnDataConfig.__post_init__(self)
        PytorchCustomDataConfig.__post_init__(self)
        # Set default scorer if not provided
        from ..score.fairness import DefaultFairlearnClassificationConfig, DefaultFairlearnRegressionConfig
        if self.scorer is None:
            self.scorer = (
                DefaultFairlearnClassificationConfig() if getattr(self, "classifier", True) else DefaultFairlearnRegressionConfig()
            )

    def _sample(self):
        # If _X is a single dataset, split it into train/test using indices
        if not (isinstance(self._X, (tuple, list)) and len(self._X) == 2):
            # Assume self._X is a Dataset, split indices
            num_samples = len(self._X)
            indices = np.arange(num_samples)
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
            if self.train_size is None and self.test_size is None:
                raise ValueError("Either train_size or test_size must be specified.")
            if self.train_size is None:
                test_size = int(self.test_size * num_samples) if isinstance(self.test_size, float) else self.test_size
                train_size = num_samples - test_size
            elif self.test_size is None:
                train_size = int(self.train_size * num_samples) if isinstance(self.train_size, float) else self.train_size
                test_size = num_samples - train_size
            else:
                train_size = int(self.train_size * num_samples) if isinstance(self.train_size, float) else self.train_size
                test_size = int(self.test_size * num_samples) if isinstance(self.test_size, float) else self.test_size
            train_idx = indices[:train_size]
            test_idx = indices[train_size:train_size+test_size]
            from torch.utils.data import Subset
            train_ds = Subset(self._X, train_idx)
            test_ds = Subset(self._X, test_idx)
            self._X = (train_ds, test_ds)
            # Set train/test indices as torch tensors for downstream logic
            import torch as _torch
            self.train_indices = _torch.tensor(train_idx)
            self.test_indices = _torch.tensor(test_idx)
            # Copy _sensitive from dataset if not present
            if not hasattr(self, "_sensitive") or self._sensitive is None:
                if hasattr(self._X[0].dataset, "_sensitive"):
                    self._sensitive = self._X[0].dataset._sensitive
        # Now call parent logic
        PytorchCustomDataConfig._sample(self)
        # After splitting, split sensitive features using the same indices if present
        if hasattr(self, "_sensitive") and self._sensitive is not None:
            sensitive_arr = np.asarray(self._sensitive, dtype=object)
            train_np_idx = self.train_indices.detach().cpu().numpy() if hasattr(self, "train_indices") else None
            test_np_idx = self.test_indices.detach().cpu().numpy() if hasattr(self, "test_indices") else None
            self._sensitive_train = sensitive_arr[train_np_idx].tolist() if train_np_idx is not None else None
            self._sensitive_test = sensitive_arr[test_np_idx].tolist() if test_np_idx is not None else None
            self._sensitive_all = sensitive_arr.tolist()
        else:
            raise ValueError("No sensitive features found.")

    def _score(self) -> dict:
        # Use the same logic as FairlearnDataConfig, but ensure y_true/y_pred/sensitive are torch-compatible
        if self.scorer is None:
            self.scorer = (
                DefaultFairlearnClassificationConfig() if getattr(self, "classifier", True) else DefaultFairlearnRegressionConfig()
            )
        if not callable(self.scorer):
            raise TypeError(f"FairlearnPytorchDataConfig.scorer must be callable or None, got {type(self.scorer)}")
        # y_true and y_pred are torch tensors, convert to numpy
        y_true = self.y_train.cpu().numpy() if hasattr(self.y_train, "cpu") else self.y_train
        # For now, assume y_pred is not used directly (model will provide predictions)
        sensitive = self._sensitive_train
        fairness_scores = self.scorer(
            y_true=y_true,
            y_pred=None,  # y_pred should be provided by the model pipeline
            mode="train",
            data=self,
            sensitive_features=sensitive,
        )
        return fairness_scores


