from __future__ import annotations

import logging
import tempfile

# Imports
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    Subset,
    TensorDataset,
)
from tqdm.auto import tqdm

from ...data.base import DataConfig, DataPipelineConfig

# deckard
from ...utils import load_class, resolve_torch_device

# Setup logger
logger = logging.getLogger(__name__)


class TorchDatasetSamplingMixin:
    """Sampling adapter returning Dataset objects.

    Required attrs:
        dataset: Dataset
        test_size: float
        train_size: float
        val_size: float
        random_state: int
        sample: Literal["split", "fold", "shuffle"]
        stratify: bool
    """

    dataset: Dataset
    test_size: float
    train_size: float
    val_size: float
    random_state: int
    sample: Literal["split", "fold", "shuffle"]
    stratify: bool

    def _get_targets(self) -> list:
        """Extract labels for stratified sampling."""
        ds = self.dataset

        if hasattr(ds, "targets"):
            return list(ds.targets)

        if hasattr(ds, "labels"):
            return list(ds.labels)

        raise AttributeError(
            "Stratified sampling requires dataset.targets or dataset.labels.",
        )

    def _validate_sizes(self) -> None:
        total = self.train_size + self.val_size + self.test_size

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "train_size + val_size + test_size must equal 1.0",
            )

    def sample(self, *, n_splits: int = 5):
        """
        Modes:
            split   -> (train_ds, val_ds, test_ds)
            fold    -> list[(train_ds, val_ds)]
            shuffle -> dataset
        """
        ds = self.dataset

        if not isinstance(ds, Dataset):
            raise TypeError(
                "dataset must be torch.utils.data.Dataset",
            )

        self._validate_sizes()
        indices = list(range(len(ds)))

        if self.sample == "split":
            return self._sample_split(ds, indices)

        if self.sample == "fold":
            return self._sample_fold(
                ds,
                indices,
                n_splits=n_splits,
            )

        if self.sample == "shuffle":
            return self._sample_shuffle(ds)

        raise ValueError(
            "sample must be 'split', 'fold', or 'shuffle'",
        )

    def _sample_split(
        self,
        ds: Dataset,
        indices: list[int],
    ) -> tuple[Subset, Subset, Subset]:
        """Return train/val/test subsets."""
        y = self._get_targets() if self.stratify else None

        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        y_trainval = [y[i] for i in trainval_idx] if y is not None else None

        val_fraction = self.val_size / (self.train_size + self.val_size)

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_fraction,
            random_state=self.random_state,
            stratify=y_trainval,
        )

        self.train_dataset = Subset(ds, train_idx)
        self.val_dataset = Subset(ds, val_idx)
        self.test_dataset = Subset(ds, test_idx)

        return (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        )

    def _sample_fold(
        self,
        ds: Dataset,
        indices: list[int],
        *,
        n_splits: int,
    ) -> list[tuple[Subset, Subset]]:
        """Return K-fold dataset subsets."""
        y = self._get_targets() if self.stratify else None

        splitter = (
            StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            if self.stratify
            else KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        )

        split_iter = (
            splitter.split(indices, y) if self.stratify else splitter.split(indices)
        )

        folds = []

        for train_idx, val_idx in split_iter:
            train_ds = Subset(ds, train_idx)
            val_ds = Subset(ds, val_idx)
            folds.append((train_ds, val_ds))

        self.folds = folds
        return folds

    def _sample_shuffle(
        self,
        ds: Dataset,
    ) -> Dataset:
        """
        Return the full dataset unchanged.

        Shuffling is deferred to DataLoader.
        """
        return ds


@dataclass(eq=False, kw_only=True)
class PytorchDataPipelineConfig(DataPipelineConfig):
    pass


class TorchDatasetMixin(TorchDatasetSamplingMixin):
    """PyTorch data mixin with dataset-aware sampling behavior."""

    sampler: Union[str, dict, Callable[..., Any], None]
    sampler_params: dict[str, Any]
    dataset_type: Union[str, None]
    n_splits: int

    def resolve_dataset_type(self, dataset_obj: Any) -> str:
        """Classify runtime dataset shape for downstream sampling behavior."""
        if isinstance(dataset_obj, TensorDataset):
            return "tensor"
        if isinstance(dataset_obj, IterableDataset):
            return "iterable"
        if isinstance(dataset_obj, Dataset):
            return "map"
        return "unknown"

    def _normalize_sampler_spec(self) -> tuple[Union[str, None], dict[str, Any]]:
        """Resolve sampler name and params from string/dict/callable specs."""
        sampler_spec = getattr(self, "sampler", None)
        params = dict(getattr(self, "sampler_params", {}) or {})

        if sampler_spec is None:
            return None, params
        if isinstance(sampler_spec, str):
            return sampler_spec.strip().lower(), params
        if isinstance(sampler_spec, dict):
            name = sampler_spec.get("name", sampler_spec.get("_target_", "split"))
            if not isinstance(name, str):
                raise TypeError(
                    f"sampler name must be a string, got {type(name)}",
                )
            merged_params = {
                k: v for k, v in sampler_spec.items() if k not in {"name", "_target_"}
            }
            merged_params.update(params)
            return name.strip().lower(), merged_params
        if callable(sampler_spec):
            return "callable", params
        raise TypeError(
            f"Unsupported sampler specification type: {type(sampler_spec)}",
        )

    def _sample_with_configurable_sampler(
        self,
        dataset_obj: Dataset,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return train/test index tensors based on configured sampler."""
        sampler_name, params = self._normalize_sampler_spec()
        num_samples = len(labels)

        # Default path keeps existing PyTorch behavior.
        if sampler_name in {None, "default", "split", "random_split"}:
            return self._sample_train_test_indices(num_samples)

        if sampler_name in {"fold", "kfold", "stratifiedkfold", "shuffle"}:
            val_size = params.get("val_size", getattr(self, "val_size", None))
            if val_size is None:
                val_size = 0.0
            runtime = TorchDatasetSamplingMixin()
            runtime.dataset = dataset_obj
            runtime.test_size = (
                float(self.test_size)
                if isinstance(self.test_size, (int, float))
                else 0.2
            )
            runtime.train_size = (
                float(self.train_size)
                if isinstance(self.train_size, (int, float))
                else 0.8
            )
            runtime.val_size = float(val_size)
            runtime.random_state = int(self.random_state)
            runtime.sample = (
                "fold"
                if sampler_name in {"fold", "kfold", "stratifiedkfold"}
                else "shuffle"
            )
            runtime.stratify = bool(self.stratify)

            if runtime.sample == "shuffle":
                # Shuffle mode leaves dataset unchanged; fall back to train/test indices.
                return self._sample_train_test_indices(num_samples)

            folds = runtime.sample(
                n_splits=int(params.get("n_splits", getattr(self, "n_splits", 5))),
            )
            if not folds:
                raise ValueError("Configured fold sampler produced no folds")
            train_subset, test_subset = folds[0]
            train_idx = torch.as_tensor(train_subset.indices, dtype=torch.long)
            test_idx = torch.as_tensor(test_subset.indices, dtype=torch.long)
            return train_idx, test_idx

        if sampler_name == "callable" and callable(getattr(self, "sampler", None)):
            result = self.sampler(
                num_samples=num_samples,
                labels=labels,
                random_state=self.random_state,
                train_size=self.train_size,
                test_size=self.test_size,
                **params,
            )
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise ValueError(
                    "Callable sampler must return (train_idx, test_idx)",
                )
            train_idx = torch.as_tensor(result[0], dtype=torch.long)
            test_idx = torch.as_tensor(result[1], dtype=torch.long)
            return train_idx, test_idx

        if "." in sampler_name or ":" in sampler_name:
            sampler_callable = load_class(sampler_name)
            result = sampler_callable(
                num_samples=num_samples,
                labels=labels,
                random_state=self.random_state,
                train_size=self.train_size,
                test_size=self.test_size,
                **params,
            )
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise ValueError(
                    "Loaded sampler callable must return (train_idx, test_idx)",
                )
            return (
                torch.as_tensor(result[0], dtype=torch.long),
                torch.as_tensor(result[1], dtype=torch.long),
            )

        raise ValueError(f"Unsupported sampler mode: {sampler_name}")

    def _sample_train_test_indices(self, num_samples: int) -> tuple[Tensor, Tensor]:
        """Default random train/test split index generation."""
        indices = torch.arange(num_samples)
        perm = torch.randperm(
            num_samples,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        indices = indices[perm]

        if self.train_size is None and self.test_size is None:
            raise ValueError("Either train_size or test_size must be specified.")

        if self.train_size is None:
            test_size = (
                int(self.test_size * num_samples)
                if isinstance(self.test_size, float)
                else int(self.test_size)
            )
            train_size = num_samples - test_size
        elif self.test_size is None:
            train_size = (
                int(self.train_size * num_samples)
                if isinstance(self.train_size, float)
                else int(self.train_size)
            )
            test_size = num_samples - train_size
        else:
            train_size = (
                int(self.train_size * num_samples)
                if isinstance(self.train_size, float)
                else int(self.train_size)
            )
            test_size = (
                int(self.test_size * num_samples)
                if isinstance(self.test_size, float)
                else int(self.test_size)
            )

        if train_size + test_size > num_samples:
            raise ValueError("Train size and test size exceed total samples.")

        train_idx = indices[:train_size]
        test_idx = indices[train_size : train_size + test_size]  # noqa E203
        return train_idx, test_idx

    pass


@dataclass(eq=False, kw_only=True)
class PytorchDataConfig(TorchDatasetMixin, DataConfig):
    """Configuration for PyTorch datasets.

    Attributes:
        dataset_name (str): Fully qualified class name of dataset
            (e.g., "torchvision.datasets.MNIST" or "custom_module.CustomDataset").
        data_params (dict): Additional parameters for dataset loading.
        test_size (Union[float, int, None]): Proportion or absolute number of test samples.
        train_size (Union[float, int, None]): Proportion or absolute number of train samples.
        random_state (int): Random seed for reproducibility.
        stratify (Union[None, str, bool]): Whether to stratify the split.
        pipeline (Dict[str, DataPipelineConfig]): Data processing pipelines.

    """

    dataset_name: str = "torchvision.datasets.MNIST"
    device: Union[str, None] = None
    data_dir: str = "./raw_data"
    test_size: Union[float, int, None] = 0.2
    train_size: Union[float, int, None] = 0.7
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    pipeline: dict[str, Any] = field(default_factory=dict)
    classifier: bool = True
    target: Optional[str] = None
    data_params: dict = field(default_factory=dict)
    drop: List[str] = field(default_factory=list)
    keep: List[str] = field(default_factory=list)
    sampler: Union[str, dict, Callable[..., Any], None] = "split"
    sampler_params: dict[str, Any] = field(default_factory=dict)
    dataset_type: Union[str, None] = None
    n_splits: int = 5

    def _normalize_sensitive_item(self, sensitive_item: Any) -> Any:
        if isinstance(sensitive_item, torch.Tensor):
            if sensitive_item.ndim == 0:
                return sensitive_item.item()
            return tuple(sensitive_item.detach().cpu().tolist())
        if isinstance(sensitive_item, np.ndarray):
            if sensitive_item.ndim == 0:
                return sensitive_item.item()
            return tuple(sensitive_item.tolist())
        if isinstance(sensitive_item, (list, tuple)):
            return tuple(sensitive_item)
        if isinstance(sensitive_item, dict):
            return tuple((k, sensitive_item[k]) for k in sorted(sensitive_item.keys()))
        return sensitive_item

    def _initialize_torch_device(self) -> None:
        self.device = str(resolve_torch_device(self.device))

    def _validate_pytorch_dataset_constraints(self) -> None:
        assert (
            self.target is None
        ), f"Target variable should not be set for PyTorch datasets. Got {self.target}."
        assert (
            len(self.drop) == 0
        ), f"Drop columns should not be set for PyTorch datasets. Got {self.drop}."
        assert (
            len(self.keep) == 0
        ), f"Keep columns should not be set for PyTorch datasets. Got {self.keep}."
        assert (
            self.train_size is not None and self.train_size > 0
        ), "train_size must be specified for PyTorch datasets."
        assert (
            self.test_size is not None and self.test_size > 0
        ), "test_size must be specified for PyTorch datasets."

    def _initialize_data_params(self) -> None:
        if self.data_dir is None:
            self.data_dir = tempfile.gettempdir()
        if self.data_params is None:
            self.data_params = {}
        if (
            "root" not in self.data_params
            and isinstance(self.dataset_name, str)
            and (
                self.dataset_name.startswith("torchvision.datasets.")
                or self.dataset_name.lower()
                in {"mnist", "torch_mnist", "cifar10", "torch_cifar10"}
            )
        ):
            self.data_params["root"] = self.data_dir

    def _initialize_timing_fields(self) -> None:
        self.data_load_time = None
        self.data_sample_time = None
        self.data_score_time = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure self.dataset is set for downstream logic
        if not hasattr(self, "dataset") or self.dataset is None:
            self.dataset = self.dataset_name
        self._initialize_torch_device()
        self._validate_pytorch_dataset_constraints()
        self._initialize_data_params()
        self._initialize_timing_fields()

    def __hash__(self):
        return super().__hash__()

    def _score(self, *args, mode: str | None = None, **kwargs) -> dict:
        """Run base scoring and mirror legacy pre-sample key for compatibility.

        Base data scoring defaults to ``test`` mode when no explicit runtime mode
        is provided. PyTorch framework persistence tests still assert the historical
        ``pre-sample`` bucket, so we expose that key as a compatibility mirror
        without altering the underlying scoring mode selection.
        """
        scores = super()._score(*args, mode=mode, **kwargs)
        if (
            mode is None
            and str(getattr(self, "score_mode", "")).strip().lower() == "test"
            and isinstance(scores, dict)
            and "test" in scores
            and "pre-sample" not in scores
            and isinstance(scores.get("test"), dict)
        ):
            scores = {"pre-sample": dict(scores["test"]), **scores}
        return scores

    def _load_data(self) -> None:
        """Load a PyTorch dataset using load_class for generic instantiation.

        Args:
            Uses self.dataset_name (fully qualified class name) and self.data_params.

        Returns:
            Sets self._X and self._y as torch Tensors.
        """
        dataset_name = self.dataset_name
        start = time.perf_counter()

        try:
            # Backward compatibility for historical shorthand names.
            if "." not in dataset_name and ":" not in dataset_name:
                dataset_aliases = {
                    "mnist": "torchvision.datasets.MNIST",
                    "torch_mnist": "torchvision.datasets.MNIST",
                    "cifar10": "torchvision.datasets.CIFAR10",
                    "torch_cifar10": "torchvision.datasets.CIFAR10",
                }
                if dataset_name.lower() not in dataset_aliases:
                    raise ImportError(
                        f"Unknown dataset alias '{dataset_name}'. Use a fully qualified class path.",
                    )
                dataset_name = dataset_aliases[dataset_name.lower()]

            # If using torchvision image datasets, ensure transform=ToTensor() if not set
            if dataset_name.startswith("torchvision.datasets."):
                try:
                    from torchvision import transforms

                    if (
                        "transform" not in self.data_params
                        or self.data_params["transform"] is None
                    ):
                        self.data_params["transform"] = transforms.ToTensor()
                except ImportError:
                    pass

            # Instantiate the dataset using load_class.
            # Keep DataLoader-only keys out of dataset constructor kwargs.
            loader_only_keys = {
                "batch_size",
                "num_workers",
                "pin_memory",
                "shuffle",
                "drop_last",
                "persistent_workers",
                "prefetch_factor",
            }
            dataset_params = {
                key: value
                for key, value in (self.data_params or {}).items()
                if key not in loader_only_keys
            }
            full_dataset = load_class(dataset_name, **dataset_params)
            self.dataset_obj = full_dataset
            self.dataset_type = self.resolve_dataset_type(full_dataset)

            # Extract data and labels from dataset. For very large datasets,
            # _max_samples can cap materialization for fast iteration.
            dataset_len = len(full_dataset)
            sample_cap = self._resolve_max_samples(dataset_len)
            n_to_load = dataset_len if sample_cap is None else sample_cap
            samples = [full_dataset[i] for i in range(n_to_load)]
            sensitive_values = []

            # Stack tensors and labels
            if isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:

                def _coerce_tensor(value: Any) -> Tensor:
                    if isinstance(value, Tensor):
                        return value
                    try:
                        return torch.as_tensor(value)
                    except Exception:
                        # torchvision datasets can return PIL images; normalize via numpy first.
                        return torch.as_tensor(np.asarray(value))

                X_list = [_coerce_tensor(s[0]) for s in samples]
                y_list = [
                    (s[1] if isinstance(s[1], (int, Tensor)) else _coerce_tensor(s[1]))
                    for s in samples
                ]
                if len(samples[0]) >= 3:
                    sensitive_values = [
                        self._normalize_sensitive_item(s[2]) for s in samples
                    ]
                self._X = torch.stack(X_list)
                if self._X.ndim == 3:
                    self._X = self._X.unsqueeze(1)
                if self._X.dtype == torch.uint8:
                    self._X = self._X.float().div(255.0)
                elif not torch.is_floating_point(self._X):
                    self._X = self._X.float()
                self._y = (
                    torch.stack(y_list)
                    if isinstance(y_list[0], Tensor)
                    else torch.tensor(y_list)
                )
            else:
                raise ValueError(
                    f"Dataset samples must be (X, y) tuples, got {type(samples[0])}",
                )

            # Allow datasets to expose sensitive metadata separately from model inputs.
            if len(sensitive_values) == 0 and hasattr(
                full_dataset,
                "_sensitive",
            ):
                raw_sensitive = getattr(full_dataset, "_sensitive")
                if raw_sensitive is not None:
                    sensitive_values = [
                        self._normalize_sensitive_item(v) for v in list(raw_sensitive)
                    ]

            if len(sensitive_values) > 0:
                if len(sensitive_values) != len(self._y):
                    raise ValueError(
                        "Sensitive metadata length must match labels length for fairness workflows.",
                    )
                self._sensitive = sensitive_values

            end = time.perf_counter()
            self.data_load_time = end - start
            logger.info(
                f"Loaded dataset {self.dataset_name} in {self.data_load_time:.2f} seconds. "
                f"Shape: {self._X.shape}, Labels: {self._y.shape}",
            )

            assert isinstance(
                self._X,
                Tensor,
            ), f"Expected _X to be Tensor, got {type(self._X)}"
            assert isinstance(
                self._y,
                Tensor,
            ), f"Expected _y to be Tensor, got {type(self._y)}"

        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def _sample(self, run_hooks: bool = True):
        """
        Samples training and testing indices from the loaded dataset, optionally using stratification.

        Calculates the number of samples for training and testing based on ``train_size`` and ``test_size``.
        Supports stratified sampling using the target variable.
        Splits the data into training and testing sets, records the sampling time, and stores the resulting indices.

        Raises
        ------
        ValueError
            If data is not loaded, or if ``stratify`` is invalid.

        Side Effects
        ------------
        Sets ``self.train_indices``, ``self.test_indices``, and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Call _load_data first.")

        _ = run_hooks
        num_samples = len(self._X)
        # Determine stratification
        if self.stratify not in (None, True, False):
            raise ValueError(
                f"stratify must be None, True, or False for PyTorch datasets; got {self.stratify}.",
            )

        start_time = time.perf_counter()
        # TODO: deprecate this for new sampling config. make consistent with ModelConfig
        dataset_obj = getattr(self, "dataset_obj", None)
        if isinstance(dataset_obj, Dataset):
            train_idx, test_idx = self._sample_with_configurable_sampler(
                dataset_obj=dataset_obj,
                labels=self._y,
            )
        else:
            train_idx, test_idx = self._sample_train_test_indices(num_samples)

        # Store indices as attributes for downstream compatibility
        self.train_indices = train_idx
        self.test_indices = test_idx

        end_time = time.perf_counter()
        self.data_sample_time = end_time - start_time

        # For compatibility with sklearn-like and torch workflows, set as Subset objects and tensors
        from torch.utils.data import Subset

        # If the original dataset is available, use Subset; else fallback to tensor slices
        if hasattr(self, "dataset_obj") and self.dataset_obj is not None:
            self.X_train = Subset(self.dataset_obj, train_idx.tolist())
            self.X_test = Subset(self.dataset_obj, test_idx.tolist())
        else:
            self.X_train = self._X[train_idx]
            self.X_test = self._X[test_idx]
        self.y_train = self._y[train_idx]
        self.y_test = self._y[test_idx]

        if hasattr(self, "_sensitive") and self._sensitive is not None:
            sensitive_arr = np.asarray(self._sensitive, dtype=object)
            train_np_idx = self.train_indices.detach().cpu().numpy()
            test_np_idx = self.test_indices.detach().cpu().numpy()
            self._sensitive_train = sensitive_arr[train_np_idx].tolist()
            self._sensitive_test = sensitive_arr[test_np_idx].tolist()
            self._sensitive_all = sensitive_arr.tolist()

        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)

        logger.info(
            f"Data sampled in {self.data_sample_time:.2f} seconds. "
            f"Train: {self.train_n}, Test: {self.test_n}",
        )

        # Accept X_train/X_test as Tensor, Subset, or Dataset for torch compatibility
        # Accept y_train/y_test as Tensor or Dataset (for custom loaders)
        from torch.utils.data import Subset

        assert isinstance(
            self.X_train,
            (Tensor, Subset, Dataset),
        ), "X_train must be a Tensor, Subset, or Dataset"
        assert isinstance(
            self.y_train,
            (Tensor, Dataset),
        ), "y_train must be a Tensor or Dataset"
        assert isinstance(
            self.X_test,
            (Tensor, Subset, Dataset),
        ), "X_test must be a Tensor, Subset, or Dataset"
        assert isinstance(
            self.y_test,
            (Tensor, Dataset),
        ), "y_test must be a Tensor or Dataset"

    def __call__(  # noqa: F811
        self,
        data_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> dict:
        """Load, sample, and optionally persist torch data artifacts and scores.

        Args:
            data_file: Optional path used to load/save serialized data runtime state.
            score_file: Optional path used to load/save serialized scoring outputs.

        Returns:
            A score dictionary augmented with runtime timing metrics.
        """
        if data_file is not None:
            assert isinstance(
                data_file,
                str,
            ), "data_file must be a string path."
            if not Path(data_file).exists():
                Path(data_file).parent.mkdir(parents=True, exist_ok=True)
            else:
                pass

        if score_file is not None:
            assert isinstance(
                score_file,
                str,
            ), "score_file must be a string path."
            if Path(score_file).exists():
                pass

        if self.data_load_time is None:
            self._load_data()

        assert self._X is not None, "_X not loaded"
        assert self._y is not None, "_y not loaded"

        if self.data_sample_time is None:
            self._sample()

        assert self.X_train is not None, "X_train not sampled"
        assert self.X_test is not None, "X_test not sampled"
        assert self.y_train is not None, "y_train not sampled"
        assert self.y_test is not None, "y_test not sampled"

        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
        }

        scores = self._score()
        all_scores = {**time_dict, **scores}
        self.score_dict = all_scores

        if data_file is not None:
            pass

        if score_file is not None:
            self.save_scores(scores, score_file)

        return all_scores


@dataclass(eq=False, kw_only=True)
class PytorchCustomDataConfig(PytorchDataConfig):
    """Configuration for HuggingFace datasets loaded via DataLoader.

    Extends PytorchDataConfig to support HuggingFace datasets with custom
    transforms and DataLoader-based loading.
    """

    val: bool = False
    dataset_params: dict = field(default_factory=dict)
    dataset: str = field(default_factory=str)
    test_transform: str | None = field(default_factory=str)
    train_transform: str | None = field(default_factory=str)
    loaders: list = field(init=False, repr=False)
    data_load_time: Union[float, None] = None
    data_sample_time: Union[float, None] = None
    transform_params: dict = field(default_factory=dict)
    score_dict: dict = field(init=False, repr=False)

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        super().__post_init__()
        self._initialize_timing_fields()
        if not self.data_params:
            self.data_params = {}
        if not hasattr(self, "shuffle"):
            self.shuffle = True

    def _as_dataset(self, obj, split: str, transform):
        if isinstance(obj, str):
            if not obj:
                raise ValueError(
                    "dataset path cannot be empty for custom torch dataset",
                )
            obj = load_class(
                obj,
                **self.dataset_params,
                split=split,
                transform=transform,
            )
            return obj
        if isinstance(obj, type) and issubclass(obj, Dataset):
            return obj(**self.dataset_params, split=split, transform=transform)
        if isinstance(obj, Dataset):
            return obj
        raise TypeError(
            f"Invalid dataset object for split '{split}': {type(obj)}",
        )

    @staticmethod
    def _extract_label_tensor(dataset: Dataset) -> torch.Tensor:
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                raise ValueError(
                    "Each dataset sample must be (X, y) or (X, y, sensitive)",
                )
            label = sample[1]
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().reshape(-1)
                labels.append(label[0].item() if label.numel() > 0 else 0)
            else:
                labels.append(label)
        return torch.as_tensor(labels, dtype=torch.long)

    def _truncate_dataset(self, dataset: Dataset, size):
        # Allow float proportions as well as int counts
        if isinstance(size, float):
            n = len(dataset)
            size = int(round(size * n))
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an integer or float proportion. Got: {size}.",
            )
        dataset = Subset(dataset, range(size))
        return dataset

    def _load_data(self):
        """
        Loads train/test datasets as DataLoaders without materializing all samples in memory.

        Updates ``self._X``, ``self._y``, ``s.elf.X_train``, ``self.X_test``,
        ``self.y_train``, ``self.y_test``, ``self.train_n``, ``self.test_n``,
        ``self.data_load_time``, and ``self.data_sample_time``.
        """
        logger.info("Loading custom torch dataset")
        start = time.perf_counter()
        if self.train_transform and isinstance(self.train_transform, str):
            train_transform = load_class(self.train_transform)
        elif isinstance(self.train_transform, Callable):
            train_transform = self.train_transform
        else:
            train_transform = torch.Tensor
        if self.test_transform and isinstance(self.test_transform, str):
            test_transform = load_class(self.test_transform)
        elif isinstance(self.test_transform, Callable):
            test_transform = self.test_transform
        else:
            test_transform = torch.Tensor
        self.train_transform = train_transform
        self.test_transform = test_transform
        valid_split = "test" if self.val else "valid"
        train_ds = self._as_dataset(
            self.dataset,
            split="train",
            transform=train_transform,
        )
        test_ds = self._as_dataset(
            self.dataset,
            split=valid_split,
            transform=test_transform,
        )
        # For custom split datasets, only explicit integer caps should truncate.
        if isinstance(self.train_size, int):
            train_ds = self._truncate_dataset(train_ds, self.train_size)
            self.train_n = len(train_ds)
        else:
            self.train_n = len(train_ds)
        if isinstance(self.test_size, int):
            test_ds = self._truncate_dataset(test_ds, size=self.test_size)
            self.test_n = len(test_ds)
        else:
            self.test_n = len(test_ds)

        # Keep train/test split datasets explicit for sampling and downstream compatibility.
        self._X = (train_ds, test_ds)
        train_labels = self._extract_label_tensor(train_ds)
        test_labels = self._extract_label_tensor(test_ds)
        self._y = torch.cat([train_labels, test_labels], dim=0)

        end = time.perf_counter()
        self.data_load_time = end - start
        # Sampling is already defined by provided train/test splits

        logger.info(
            f"Loaded custom dataset lazily in {self.data_load_time:.2f}s "
            f"(train={self.train_n}, test={self.test_n}).",
        )

    def _sample(self):
        # DataLoader params (lazy loading, no full dataset materialization)
        logger.info("Creating torch data loaders.")
        start = time.perf_counter()
        batch_size = int(self.data_params.get("batch_size", 32))
        num_workers = int(self.data_params.get("num_workers", 0))
        pin_memory = bool(
            self.data_params.get("pin_memory", self.device != "cpu"),
        )
        if not isinstance(self._X, (tuple, list)) or len(self._X) != 2:
            raise ValueError(
                "Expected custom torch _X to contain (train_dataset, test_dataset)",
            )
        train_ds, test_ds = self._X
        torch.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.loaders = [train_loader, test_loader]
        # Materialize batches from loaders into tensors
        train_y_batches = []
        train_sensitive_batches = []
        for batch in tqdm(
            train_loader,
            desc="Materializing train batches",
            total=len(train_loader),
            leave=False,
        ):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Each train batch must be (X, y) or (X, y, sensitive)",
                )
            yb = batch[1]
            train_y_batches.append(yb)
            if len(batch) >= 3:
                train_sensitive_batches.extend(
                    [self._normalize_sensitive_item(v) for v in list(batch[2])],
                )

        test_y_batches = []
        test_sensitive_batches = []
        for batch in tqdm(
            test_loader,
            desc="Materializing test batches",
            total=len(test_loader),
            leave=False,
        ):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Each test batch must be (X, y) or (X, y, sensitive)",
                )
            yb = batch[1]
            test_y_batches.append(yb)
            if len(batch) >= 3:
                test_sensitive_batches.extend(
                    [self._normalize_sensitive_item(v) for v in list(batch[2])],
                )

        def _sensitive_from_split(ds):
            if ds is None:
                return None
            if isinstance(ds, Subset):
                base_sensitive = getattr(ds.dataset, "_sensitive", None)
                if base_sensitive is None:
                    return None
                arr = np.asarray(base_sensitive, dtype=object)
                indices = np.asarray(ds.indices)
                return [
                    self._normalize_sensitive_item(v) for v in arr[indices].tolist()
                ]
            direct_sensitive = getattr(ds, "_sensitive", None)
            if direct_sensitive is None:
                return None
            return [self._normalize_sensitive_item(v) for v in list(direct_sensitive)]

        split_train_sensitive = _sensitive_from_split(train_ds)
        split_test_sensitive = _sensitive_from_split(test_ds)
        if len(train_sensitive_batches) == 0 and split_train_sensitive is not None:
            train_sensitive_batches = split_train_sensitive
        if len(test_sensitive_batches) == 0 and split_test_sensitive is not None:
            test_sensitive_batches = split_test_sensitive

        self.X_train = train_loader
        self.y_train = (
            torch.cat(train_y_batches, dim=0)
            if train_y_batches
            else torch.empty(0, dtype=torch.long)
        )
        self.X_test = test_loader
        self.y_test = (
            torch.cat(test_y_batches, dim=0)
            if test_y_batches
            else torch.empty(0, dtype=torch.long)
        )
        if self.y_train.ndim > 1:
            self.y_train = self.y_train.reshape(self.y_train.shape[0], -1)[:, 0]
        if self.y_test.ndim > 1:
            self.y_test = self.y_test.reshape(self.y_test.shape[0], -1)[:, 0]

        if len(train_sensitive_batches) > 0 or len(test_sensitive_batches) > 0:
            self._sensitive_train = train_sensitive_batches
            self._sensitive_test = test_sensitive_batches
            self._sensitive_all = train_sensitive_batches + test_sensitive_batches

        end = time.perf_counter()
        self.data_sample_time = end - start

    def __call__(
        self,
        data_file: str | None = None,
        score_file: str | None = None,
        mode: Union[str, None] = "test",
        *args,
        **kwargs,
    ) -> dict:
        """Load torch custom data, run sampling/scoring, and persist optional outputs.

        Args:
            data_file: Optional path to serialized runtime data state.
            score_file: Optional path to serialized score output.

        Returns:
            Dictionary of computed and/or loaded scoring values.
        """
        cached_scores = None
        if data_file is not None and Path(data_file).exists():
            self = self.load_object(data_file)
        if score_file is not None and Path(score_file).exists():
            cached_scores = self.load_scores(score_file)

        if cached_scores is not None:
            scores = dict(cached_scores)
            self.score_dict = scores
            if score_file is not None:
                self.save_scores(scores, filepath=score_file)
            if data_file is not None:
                self.save_object(self, data_file)
            return scores

        scores = {}
        if not hasattr(self, "_X") or self._X is None:
            self._load_data()
        if getattr(self, "X_train", None) is None:
            self._sample()
        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
            "data_score_time": self.data_score_time,
        }
        new_scores = self._score(mode=mode, *args, **kwargs)
        scores.update(**time_dict, **new_scores)
        self.score_dict = scores
        if score_file is not None:
            self.save_scores(scores, filepath=score_file)
        if data_file is not None:
            self.save_object(self, data_file)
        return scores
