import collections
import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    Subset,  # Ensure Subset is always in scope
)

from ...artifacts import ScoreDict
from ...data.base import DataConfig
from ...plugins.fairlearn.data import FairlearnDataConfig
from ...plugins.fairlearn.score import (
    DefaultFairlearnDataScorerDictConfig,
    fairness_stage_to_split_mode,
)
from ...utils import probabilities_from_model_outputs
from .data import PytorchCustomDataConfig
from .score import (
    coerce_to_numpy,
    is_dataloader_like,
    is_dataset_like,
    materialize_dataset,
    resolve_split_arrays,
    resolve_sensitive_features,
)

logger = logging.getLogger(__name__)

RuntimeScalar = str | int | float | bool | None
SerializableValue = (
    RuntimeScalar | list["SerializableValue"] | dict[str, "SerializableValue"]
)


class TinyFairness(Dataset):
    """Minimal synthetic dataset for fairness testing."""

    def __init__(
        self,
        num_samples=40,
        n_features=4,
        random_state=123,
        split=None,
        transform=None,
    ):
        np.random.seed(random_state)
        self._X = np.random.randn(num_samples, n_features)
        # Binary labels 0/1
        self._y = np.random.randint(0, 2, size=num_samples)
        # Sensitive attribute: two groups 'A' and 'B'
        self._sensitive = np.random.choice(["A", "B"], size=num_samples)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        x = torch.tensor(self._X[idx], dtype=torch.float32)
        y = torch.tensor(self._y[idx], dtype=torch.long)  # scalar
        return x, y


class SyntheticImageSensitiveDataset(Dataset):
    """Synthetic image dataset with hidden sensitive attributes.

    Attributes:
        images: Randomly generated image tensor payload.
        labels: Randomly generated target labels.
        _sensitive: Internal sensitive-attribute tensor not exposed by item access.
    """

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
        self._sensitive = torch.stack([sex, gender], dim=1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Model only sees image + target
        return self.images[idx], self.labels[idx]


@dataclass(eq=False, kw_only=True)
class FairlearnPytorchDataConfig(FairlearnDataConfig, PytorchCustomDataConfig):
    """Fairlearn-compatible DataConfig for PyTorch Datasets with sensitive features.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    _target_: str = (
        "deckard.frameworks.pytorch.fairness_data.FairlearnPytorchDataConfig"
    )
    scorer: Any = None

    def _ensure_data_scorer_default(self) -> None:
        scorer_name = type(self.scorer).__name__ if self.scorer is not None else ""
        if self.scorer is None or scorer_name in {
            "DefaultFairlearnClassificationScorerDictConfig",
            "DefaultFairlearnRegressionScorerDictConfig",
        }:
            self.scorer = DefaultFairlearnDataScorerDictConfig(
                classifier=getattr(self, "classifier", True),
            )

    def _ensure_runtime_scorer(self) -> None:
        """Ensure runtime scorer is a fairlearn data scorer with canonical defaults."""
        from ...utils import is_default_config_value

        if (
            is_default_config_value(self.scorer, include_best=False)
            or self.scorer is None
        ):
            self.scorer = DefaultFairlearnDataScorerDictConfig(
                classifier=getattr(self, "classifier", True),
            )
        self._ensure_data_scorer_default()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __post_init__(self):
        FairlearnDataConfig.__post_init__(self)
        PytorchCustomDataConfig.__post_init__(self)
        if not hasattr(self, "dataset") or not self.dataset:
            self.dataset = self.dataset_name
        self._ensure_data_scorer_default()

    # ------------------------------------------------------------------
    # Data loading / splitting
    # ------------------------------------------------------------------

    def load_dataset(self) -> "FairlearnPytorchDataConfig":
        """Use the PyTorch parent dataset-loading flow for runtime materialization.

        Returns:
            This FairlearnPytorchDataConfig instance.
        """
        return cast(
            "FairlearnPytorchDataConfig",
            PytorchCustomDataConfig.load_dataset(self),
        )

    def _fit_transform_X(self, X_train, X_test, y_train, y_test, pipeline):
        """Run canonical DataConfig pipeline fit/transform for torch fairness data."""
        return DataConfig._fit_transform_X(
            self,
            X_train,
            X_test,
            y_train,
            y_test,
            pipeline,
        )

    def fit(self, run_hooks: bool = True) -> "FairlearnPytorchDataConfig":
        """Split the dataset and extract per-split sensitive-feature arrays.

        Args:
            run_hooks: Reserved hook toggle for interface compatibility.

        Returns:
            This FairlearnPytorchDataConfig instance.

        Raises:
            ValueError: If split-size configuration is invalid.
        """
        _ = run_hooks
        if not (isinstance(self._X, (tuple, list)) and len(self._X) == 2):
            num_samples = len(self._X)
            indices = np.arange(num_samples)
            random_state = int(self._get_sampler_option("random_state", 42))
            train_size_cfg = self._get_sampler_option("train_size", None)
            test_size_cfg = self._get_sampler_option("test_size", 0.2)
            np.random.seed(random_state)
            np.random.shuffle(indices)

            if train_size_cfg is None and test_size_cfg is None:
                raise ValueError("Either train_size or test_size must be specified.")
            if train_size_cfg is None:
                test_size = (
                    int(test_size_cfg * num_samples)
                    if isinstance(test_size_cfg, float)
                    else test_size_cfg
                )
                train_size = num_samples - test_size
            elif test_size_cfg is None:
                train_size = (
                    int(train_size_cfg * num_samples)
                    if isinstance(train_size_cfg, float)
                    else train_size_cfg
                )
                test_size = num_samples - train_size
            else:
                train_size = (
                    int(train_size_cfg * num_samples)
                    if isinstance(train_size_cfg, float)
                    else train_size_cfg
                )
                test_size = (
                    int(test_size_cfg * num_samples)
                    if isinstance(test_size_cfg, float)
                    else test_size_cfg
                )

            train_idx = indices[:train_size]
            test_idx = indices[train_size : train_size + test_size]
            train_ds = Subset(self._X, train_idx)
            test_ds = Subset(self._X, test_idx)
            self._X = (train_ds, test_ds)
            self.train_indices = torch.tensor(train_idx)
            self.test_indices = torch.tensor(test_idx)
        else:
            self.train_indices = getattr(self, "train_indices", None)
            self.test_indices = getattr(self, "test_indices", None)

        PytorchCustomDataConfig.fit(self)
        self._extract_sensitive_splits()
        return self

    def _extract_sensitive_splits(self):
        """Populate ``_sensitive_train``, ``_sensitive_test``, ``_sensitive_all``."""

        def _from_split(ds) -> list | None:
            if isinstance(ds, Subset):
                base_sensitive = getattr(ds.dataset, "_sensitive", None)
                if base_sensitive is not None:
                    arr = np.asarray(base_sensitive, dtype=object)
                    return arr[np.asarray(ds.indices)].tolist()
            direct = getattr(ds, "_sensitive", None)
            if direct is not None:
                return np.asarray(direct, dtype=object).tolist()
            return None

        if not (isinstance(self._X, (tuple, list)) and len(self._X) == 2):
            raise RuntimeError(
                "_extract_sensitive_splits called before dataset was split",
            )

        train_ds, test_ds = self._X
        train_sensitive = _from_split(train_ds)
        test_sensitive = _from_split(test_ds)

        if train_sensitive is None and test_sensitive is None:
            raise ValueError(
                "No sensitive features found in the torch dataset "
                "(_sensitive attribute missing or None).",
            )

        self._sensitive_train = train_sensitive
        self._sensitive_test = test_sensitive
        self._sensitive_all = (train_sensitive or []) + (test_sensitive or [])
        self._sensitive_val = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def __call__(
        self,
        *args: object,
        **kwargs: object,
    ) -> dict[str, SerializableValue]:
        """Run fairness-aware torch data execution with a default scorer fallback.

        Args:
            *args: Positional runtime arguments forwarded to the parent call.
            **kwargs: Keyword runtime arguments forwarded to the parent call.

        Returns:
            Lifecycle outputs returned by the parent data configuration.
        """
        self._ensure_runtime_scorer()
        result = super().__call__(*args, **kwargs)
        assert hasattr(self, "X_train"), ".X_train not found"
        return result

    def score(
        self,
        *args: Any,
        mode: str | None = None,
        **kwargs: Any,
    ) -> ScoreDict:
        """Compute fairness scores using canonical helpers for sensitive-feature lookup.

        Args:
            mode: Scoring mode token for stage/split resolution.

        Returns:
            Fairness score mapping for the requested mode.

        Raises:
            TypeError: If configured scorer is not callable.
        """
        _ = (args, kwargs)
        self._ensure_runtime_scorer()
        if not callable(self.scorer):
            raise TypeError(
                f"FairlearnPytorchDataConfig.scorer must be callable, got {type(self.scorer)}",
            )

        scorer_mode = mode if mode is not None else "train"
        stage_to_split_mode = fairness_stage_to_split_mode(scorer_mode)

        # Map "pre-sample" -> dataset-level summary (no model predictions needed).
        if scorer_mode == "pre-sample":
            y_all, sensitive = self._get_full_dataset_labels()
            return ScoreDict(
                {
                    "n_samples": len(y_all),
                    "label_distribution": dict(collections.Counter(y_all)),
                    "sensitive_distribution": dict(collections.Counter(sensitive)),
                },
            )

        # Canonical sensitive-feature lookup.
        sensitive = resolve_sensitive_features(
            self,
            scorer_mode,
            stage_to_split_mode=stage_to_split_mode,
        )
        if sensitive is None:
            logger.warning(
                "No sensitive features for mode '%s'; skipping fairness scoring.",
                scorer_mode,
            )
            return ScoreDict()

        # Extract (X, y) arrays for the requested split.
        X, y_true = self._get_split_arrays(
            scorer_mode,
            stage_to_split_mode=stage_to_split_mode,
        )
        if y_true is None:
            return ScoreDict()

        y_proba = None
        try:
            y_proba = probabilities_from_model_outputs(X)
        except Exception:
            y_proba = None

        fairness_scores = self.scorer(
            y=y_true,
            X=X,
            y_proba=y_proba,
            mode=scorer_mode,
            data=self,
            sensitive_features=sensitive,
        )
        # Flatten nested dicts.
        if isinstance(fairness_scores, dict):
            flat: dict = {}
            for k, v in fairness_scores.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        flat[f"{k}_{subk}"] = subv
                else:
                    flat[k] = v
            return ScoreDict(flat)
        return ScoreDict({"fairness_score": fairness_scores})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_split_arrays(
        self,
        mode: str,
        *,
        stage_to_split_mode: dict[str, str] | None = None,
    ):
        """Return ``(X_array, y_array)`` numpy arrays for the given split *mode*.

        Handles Subset / DataLoader via ``materialize_dataset``.
        """
        X_raw, y_raw = resolve_split_arrays(
            self,
            mode,
            stage_to_split_mode=stage_to_split_mode,
        )

        if is_dataset_like(X_raw) or is_dataloader_like(X_raw):
            X_arr, y_from_ds = materialize_dataset(X_raw)
            y_arr = coerce_to_numpy(y_raw) if y_raw is not None else y_from_ds
        else:
            X_arr = coerce_to_numpy(X_raw)
            y_arr = coerce_to_numpy(y_raw)

        return X_arr, y_arr

    def _get_full_dataset_labels(self):
        """Return ``(y_all, sensitive_all)`` lists for the un-split dataset."""
        sensitive = getattr(self, "_sensitive_all", None)
        if sensitive is None:
            raise RuntimeError("_sensitive_all not set; call _sample() first.")

        y_all = getattr(self, "_y", None)
        if y_all is None and hasattr(self, "_X") and is_dataset_like(self._X):
            _, y_all = materialize_dataset(self._X)
        if y_all is None:
            raise RuntimeError("Could not extract y labels for pre-sample mode.")

        y_all_array = coerce_to_numpy(y_all)
        if y_all_array is None:
            raise RuntimeError("Could not coerce dataset labels to numpy array.")
        return y_all_array.tolist(), sensitive
