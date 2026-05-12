import logging

from dataclasses import dataclass
from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Subset  # Ensure Subset is always in scope

from ..score.fairness import DefaultFairlearnClassificationConfig, DefaultFairlearnRegressionConfig
from .fairness import FairlearnDataConfig
from .pytorch import PytorchCustomDataConfig

        



logger = logging.getLogger(__name__)

class TinyFairness(Dataset):
    
    """Minimal synthetic dataset for fairness testing."""
    def __init__(self, num_samples=40, n_features=4, random_state=123, split=None, transform=None):
        np.random.seed(random_state)
        self._X = np.random.randn(num_samples, n_features)
        # Binary labels 0/1
        self._y = np.random.randint(0, 2, size=num_samples)
        # Sensitive attribute: two groups 'A' and 'B'
        self._sensitive = np.random.choice(['A', 'B'], size=num_samples)
        self.split = split
        self.transform = transform

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
    _target_: str = "deckard.data.fairness_pytorch.FairlearnPytorchDataConfig"
    scorer: Any = None



    def __call__(self, *args, **kwargs):
        # Auto-select fairness-compatible scorer if not set
        from deckard.utils import is_default_config_value
        if is_default_config_value(self.scorer, include_best=False) or self.scorer is None:
            from deckard.score import DefaultFairlearnClassificationConfig, DefaultFairlearnRegressionConfig
            self.scorer = (
                DefaultFairlearnClassificationConfig() if self.classifier else DefaultFairlearnRegressionConfig()
            )
        # Ensure mode and scorer_mode are always defined
        mode = kwargs.get('mode', 'pre-sample')
        scorer_mode = mode if mode is not None else 'pre-sample'
        kwargs['mode'] = mode
        result = super().__call__(*args, **kwargs)
        # If sensitive_features was not passed, re-run scoring with the correct mode-aware attribute
        if 'sensitive_features' not in kwargs and hasattr(self, 'scorer') and callable(self.scorer):
            if mode == 'train' and hasattr(self, '_sensitive_train'):
                sensitive = self._sensitive_train
                y_true = self.y_train
            elif mode == 'test' and hasattr(self, '_sensitive_test'):
                sensitive = self._sensitive_test
                y_true = self.y_test
            elif mode == 'val' and hasattr(self, '_sensitive_val'):
                sensitive = self._sensitive_val
                y_true = self.y_val
            elif mode == 'all' and hasattr(self, '_sensitive_all'):
                sensitive = self._sensitive_all
                y_true = self.y_all
            if scorer_mode == 'pre-sample' and hasattr(self, '_sensitive_all'):
                # Map pre-sample to 'all' for the scorer
                scorer_mode = 'all'
                sensitive = self._sensitive_all
                # Use self._y as the flat label array for both y_true and y_pred
                y_true = self._y
                y_pred = self._y
            elif scorer_mode == 'train' and hasattr(self, '_sensitive_train'):
                sensitive = self._sensitive_train
                y_true = self.y_train
                y_pred = y_true
            elif scorer_mode == 'test' and hasattr(self, '_sensitive_test'):
                sensitive = self._sensitive_test
                y_true = self.y_test
                y_pred = y_true
            elif scorer_mode == 'val' and hasattr(self, '_sensitive_val'):
                sensitive = self._sensitive_val
                y_true = self.y_val
                y_pred = y_true
            elif scorer_mode == 'all' and hasattr(self, '_sensitive_all'):
                sensitive = self._sensitive_all
                y_true = self.y_all
                y_pred = y_true
            else:
                sensitive = None
                y_true = None
                y_pred = None
                # Flatten fairness_scores if it's a dict
                if isinstance(fairness_scores, dict):
                    flat = {}
                    for k, v in fairness_scores.items():
                        # Flatten tuple keys and nested dicts
                        if isinstance(k, tuple):
                            k = '_'.join(str(x) for x in k)
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if isinstance(kk, tuple):
                                    kk = '_'.join(str(x) for x in kk)
                                flat[f"{k}_{kk}"] = vv
                        else:
                            flat[k] = v
                    fairness_scores = flat
                if sensitive is None or not hasattr(sensitive, "__len__") or y_true is None or len(sensitive) != len(y_true):
                    raise RuntimeError(f"[DIAGNOSE] sensitive_features problem: mode={mode}, type={type(sensitive)}, value={repr(sensitive)[:200]}, y_true type={type(y_true)}, y_true len={len(y_true) if hasattr(y_true, '__len__') else 'N/A'}, sensitive len={len(sensitive) if hasattr(sensitive, '__len__') else 'N/A'}")
                fairness_scores = self.scorer(
                    y_true=y_true,
                    y_pred=y_pred,
                    mode=scorer_mode,
                    data=self,
                    sensitive_features=sensitive,
                )
        assert hasattr(self, "X_train"), ".X_train not found"
        return result

    def _fit_transform_X(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
    ):
        """Bypass pipeline fit/transform for torch types. Returns inputs unchanged, sets timing fields."""
        if pipeline:
            raise NotImplementedError("Pytorch data pipelines not yet implemented.")
        self.pipeline_fit_time = 0.0
        self.pipeline_fit_n = len(X_train) if hasattr(X_train, '__len__') else 0
        self.pipeline_transform_time = 0.0
        self.pipeline_transform_n = len(X_test) if hasattr(X_test, '__len__') else 0
        return X_train, X_test, y_train, y_test
    def _load_data(self):
            # Only call the PyTorch parent's _load_data, skip FairlearnDataConfig's DataFrame logic
            PytorchCustomDataConfig._load_data(self)
    
    def __post_init__(self):
        # Call both parent initializers
        FairlearnDataConfig.__post_init__(self)
        PytorchCustomDataConfig.__post_init__(self)
        # Ensure self.dataset is set for downstream logic
        if not hasattr(self, "dataset") or self.dataset is None or self.dataset == "":
            self.dataset = self.dataset_name
        # Set default scorer if not provided
        if self.scorer is None:
            self.scorer = (
                DefaultFairlearnClassificationConfig() if getattr(self, "classifier", True) else DefaultFairlearnRegressionConfig()
            )

    def _sample(self):
        # Split the dataset and indices as usual
        
        # If not already split, split and set indices
        if not (isinstance(self._X, (tuple, list)) and len(self._X) == 2):
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
            train_ds = Subset(self._X, train_idx)
            test_ds = Subset(self._X, test_idx)
            self._X = (train_ds, test_ds)
            self.train_indices = torch.tensor(train_idx)
            self.test_indices = _torch.tensor(test_idx)
        else:
            # If already split, try to infer indices if possible, else set to None
            self.train_indices = getattr(self, 'train_indices', None)
            self.test_indices = getattr(self, 'test_indices', None)
        # Call parent to set up y_train/y_test, etc.
        PytorchCustomDataConfig._sample(self)

        def _extract_sensitive_for_split(ds):
            # Preferred path: split is a Subset and base dataset exposes _sensitive.
            if isinstance(ds, Subset):
                base_ds = ds.dataset
                base_sensitive = getattr(base_ds, "_sensitive", None)
                if base_sensitive is not None:
                    sens_arr = np.asarray(base_sensitive, dtype=object)
                    return sens_arr[np.asarray(ds.indices)].tolist()
            # Next path: dataset itself exposes _sensitive for this split.
            direct_sensitive = getattr(ds, "_sensitive", None)
            if direct_sensitive is not None:
                return np.asarray(direct_sensitive, dtype=object).tolist()
            return None

        train_ds, test_ds = self._X
        train_sensitive = _extract_sensitive_for_split(train_ds)
        test_sensitive = _extract_sensitive_for_split(test_ds)

        if train_sensitive is None and test_sensitive is None:
            raise ValueError("No sensitive features found in the torch dataset (_sensitive attribute missing or None).")

        self._sensitive_train = train_sensitive
        self._sensitive_test = test_sensitive
        self._sensitive_all = (train_sensitive or []) + (test_sensitive or [])

    def _score(self, mode=None) -> dict:
        # Use the same logic as FairlearnDataConfig, but ensure y_true/y_pred/sensitive are torch-compatible and always injected
        if self.scorer is None:
            self.scorer = (
                DefaultFairlearnClassificationConfig() if getattr(self, "classifier", True) else DefaultFairlearnRegressionConfig()
            )
        if not callable(self.scorer):
            raise TypeError(f"FairlearnPytorchDataConfig.scorer must be callable or None, got {type(self.scorer)}")
        scorer_mode = mode if mode is not None else "train"
        sensitive = None
        y_true = None
        if scorer_mode == 'pre-sample' and hasattr(self, '_sensitive_all'):
            # Map pre-sample to 'all' for the scorer
            scorer_mode = 'all'
            sensitive = self._sensitive_all
            # Extract y_true as a flat list matching sensitive length from the full dataset
            y_true = None
            if hasattr(self, 'dataset_obj') and hasattr(self.dataset_obj, '_y'):
                y_true = self.dataset_obj._y
                if hasattr(y_true, 'cpu'):
                    y_true = y_true.cpu().numpy()
                if hasattr(y_true, 'tolist'):
                    y_true = y_true.tolist()
            elif hasattr(self, '_y'):
                y_true = self._y
                if hasattr(y_true, 'cpu'):
                    y_true = y_true.cpu().numpy()
                if hasattr(y_true, 'tolist'):
                    y_true = y_true.tolist()
            else:
                raise RuntimeError("Could not extract y_true for pre-sample mode; dataset type unsupported.")
            if not isinstance(y_true, list):
                y_true = list(y_true)
            if len(y_true) != len(sensitive):
                raise RuntimeError(f"[DIAGNOSE] y_true and sensitive length mismatch in pre-sample mode: len(y_true)={len(y_true)}, len(sensitive)={len(sensitive)}")
            # Only compute dataset-level metrics, not accuracy or model-dependent metrics
            import collections
            result = {
                "n_samples": len(y_true),
                "label_distribution": dict(collections.Counter(y_true)),
                "sensitive_distribution": dict(collections.Counter(sensitive)),
            }
            return result
        elif scorer_mode == 'train' and hasattr(self, '_sensitive_train'):
            sensitive = self._sensitive_train
            y_true = self.y_train.cpu().numpy() if hasattr(self.y_train, 'cpu') else self.y_train
            y_pred = self.X_train.cpu().numpy() if hasattr(self, 'X_train') and hasattr(self.X_train, 'cpu') else self.X_train
        elif scorer_mode == 'test' and hasattr(self, '_sensitive_test'):
            sensitive = self._sensitive_test
            y_true = self.y_test.cpu().numpy() if hasattr(self.y_test, 'cpu') else self.y_test
            y_pred = self.X_test.cpu().numpy() if hasattr(self, 'X_test') and hasattr(self.X_test, 'cpu') else self.X_test
        elif scorer_mode == 'val' and hasattr(self, '_sensitive_val'):
            sensitive = self._sensitive_val
            y_true = self.y_val.cpu().numpy() if hasattr(self.y_val, 'cpu') else self.y_val
            y_pred = self.X_val.cpu().numpy() if hasattr(self, 'X_val') and hasattr(self.X_val, 'cpu') else self.X_val
        elif scorer_mode == 'all' and hasattr(self, '_sensitive_all'):
            sensitive = self._sensitive_all
            y_true = self.y_all.cpu().numpy() if hasattr(self.y_all, 'cpu') else self.y_all
            y_pred = self.X_all.cpu().numpy() if hasattr(self, 'X_all') and hasattr(self.X_all, 'cpu') else getattr(self, 'X_all', None)
        else:
            y_pred = None
        print(f"[DEBUG] FairlearnPytorchDataConfig._score: mode={mode}, type(sensitive)={type(sensitive)}, sensitive={repr(sensitive)[:200]}")
        logger.info(f"[DEBUG] FairlearnPytorchDataConfig._score: mode={mode}, type(sensitive)={type(sensitive)}, sensitive (first 10)={repr(sensitive)[:200]}")
        if sensitive is None or not hasattr(sensitive, "__len__") or y_true is None or len(sensitive) != len(y_true):
            raise RuntimeError(f"[DIAGNOSE] sensitive_features problem: mode={mode}, type={type(sensitive)}, value={repr(sensitive)[:200]}, y_true type={type(y_true)}, y_true len={len(y_true) if hasattr(y_true, '__len__') else 'N/A'}, sensitive len={len(sensitive) if hasattr(sensitive, '__len__') else 'N/A'}")
        fairness_scores = self.scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=scorer_mode,
            data=self,
            sensitive_features=sensitive,
        )
        # Flatten fairness_scores if it's a dict
        if isinstance(fairness_scores, dict):
            flat = {}
            for k, v in fairness_scores.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        flat[f"{k}_{subk}"] = subv
                else:
                    flat[k] = v
            return flat
        


