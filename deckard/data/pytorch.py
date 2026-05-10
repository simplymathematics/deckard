# Imports
import pandas as pd
import time
import logging
import tempfile
from tqdm.auto import tqdm
from pathlib import Path


from dataclasses import dataclass, field
from typing import Any, Union, List, Optional, Callable

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

# deckard
from ..utils import load_class, resolve_torch_device
from .base import DataConfig, DataPipelineConfig
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PytorchDataPipelineConfig(DataPipelineConfig):
    pass


@dataclass(eq=False)
class PytorchDataConfig(DataConfig):
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
    pipeline: Union[PytorchDataPipelineConfig, None] = None
    classifier: bool = True
    target: Optional[str] = None
    data_params: dict = field(default_factory=dict)
    drop: List[str] = field(default_factory=list)
    keep: List[str] = field(default_factory=list)

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
        self._initialize_torch_device()
        self._validate_pytorch_dataset_constraints()
        self._initialize_data_params()
        self._initialize_timing_fields()

    def __hash__(self):
        return super().__hash__()

    def _load_data(self) -> None:
        """Load a PyTorch dataset using load_class for generic instantiation.

        Args:
            Uses self.dataset_name (fully qualified class name) and self.data_params.

        Returns:
            Sets self._X and self._y as torch Tensors.
        """
        dataset_name = self.dataset_name
        start = time.process_time()

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

            end = time.process_time()
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

    def _sample(
        self,
    ):
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

        num_samples = len(self._X)
        indices = torch.arange(num_samples)
        # Determine stratification
        if self.stratify not in (None, True, False):
            raise ValueError(
                f"stratify must be None, True, or False for PyTorch datasets; got {self.stratify}.",
            )

        # Calculate train and test sizes
        train_size: int
        test_size: int

        if self.train_size is None and self.test_size is None:
            raise ValueError(
                "Either train_size or test_size must be specified.",
            )

        if self.train_size is None:
            test_size = (
                int(self.test_size * num_samples)
                if isinstance(self.test_size, float)
                else self.test_size
            )
            train_size = num_samples - test_size
        elif self.test_size is None:
            train_size = (
                int(self.train_size * num_samples)
                if isinstance(self.train_size, float)
                else self.train_size
            )
            test_size = num_samples - train_size
        else:
            train_size = (
                int(self.train_size * num_samples)
                if isinstance(self.train_size, float)
                else self.train_size
            )
            test_size = (
                int(self.test_size * num_samples)
                if isinstance(self.test_size, float)
                else self.test_size
            )

        if train_size + test_size > num_samples:
            raise ValueError("Train size and test size exceed total samples.")

        start_time = time.process_time()

        # Randomly shuffle indices
        perm = torch.randperm(
            num_samples,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        indices = indices[perm]

        # The first train_size indices are for training
        train_idx = indices[:train_size]
        # The next test_size indices are for testing
        test_idx = indices[train_size : train_size + test_size]  # noqa E203

        # Store indices as attributes for downstream compatibility
        self.train_indices = train_idx
        self.test_indices = test_idx

        end_time = time.process_time()
        self.data_sample_time = end_time - start_time

        # Split the data
        self.X_train = self._X[train_idx]
        self.y_train = self._y[train_idx]
        self.X_test = self._X[test_idx]
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

        assert isinstance(self.X_train, Tensor), "X_train must be a Tensor"
        assert isinstance(self.y_train, Tensor), "y_train must be a Tensor"
        assert isinstance(self.X_test, Tensor), "X_test must be a Tensor"
        assert isinstance(self.y_test, Tensor), "y_test must be a Tensor"

    def _classification_feature_scores(self):
        """
        Computes feature importance scores for classification tasks using various statistical methods.

        Returns
        -------
        dict
        A dictionary containing feature importance scores from different methods:
        - 'mutual_info_classif': Mutual information scores.
        - 'chi2': Chi-squared scores.
        - 'f_classif': ANOVA F-value scores.
        - 'class_counts': Counts of each class in the training target.
        """

        # Exit early if data already scores:
        if "class_counts" in getattr(self, "score_dict", {}):
            return {}

        score_dict = {}

        # Class counts
        y_train_np = (
            self.y_train.cpu().numpy()
            if isinstance(self.y_train, Tensor)
            else self.y_train
        )
        y_train_series = pd.Series(y_train_np)
        score_dict["class_counts"] = self._compute_class_counts(y_train_series)
        return score_dict

    def _regression_feature_scores(self):
        """ "
        Computes feature importance scores for regression tasks using various statistical methods.

        Returns
        -------
        dict
            A dictionary containing feature importance scores from different methods:
            - 'mutual_info_regression': Mutual information scores.
            - 'f_regression': F-value scores.
            - 'r_regression': Pearson correlation coefficients.
            - 'y_train_cdf': Empirical CDF of the training target.
            - 'y_test_cdf': Empirical CDF of the testing target.
        """

        # Exit early if data already scores:
        if "y_test_cdf" in getattr(self, "score_dict", {}):
            return {}

        # Ensure data is on CPU for compatibility with sklearn
        y_train_np = (
            self.y_train.cpu().numpy()
            if isinstance(self.y_train, Tensor)
            else self.y_train
        )
        y_test_np = (
            self.y_test.cpu().numpy()
            if isinstance(self.y_test, Tensor)
            else self.y_test
        )

        score_dict = {}
        # Compute metrics
        # Empirical CDFs
        y_train_sorted = np.sort(y_train_np)
        y_test_sorted = np.sort(y_test_np)
        y_train_cdf = np.arange(1, len(y_train_sorted) + 1) / len(
            y_train_sorted,
        )
        y_test_cdf = np.arange(1, len(y_test_sorted) + 1) / len(y_test_sorted)
        score_dict["y_train_cdf"] = y_train_cdf.tolist()
        score_dict["y_test_cdf"] = y_test_cdf.tolist()
        return score_dict

    def _score(self) -> dict:
        """Computes feature importance scores based on the type of task (classification or regression).

        Returns:
            dict: A dictionary containing feature importance scores.
        """
        if self.classifier:
            result = self._classification_feature_scores()
        else:
            result = self._regression_feature_scores()
        return result

    def __call__(
        self,
        data_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> dict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns timing and scoring information.
        Parameters
        ----------
        data_file : Union[str, None]
            Path to save loaded data as CSV. If None, data is not saved.
        score_file : Union[str, None]
            Path to save scores as CSV. If None, scores are not saved.
        Returns
        -------
        dict:
            A dictionary containing:
            - 'data_load_time': Time taken to load the data.
            - 'data_sample_time': Time taken to sample/split the data.
            - Additional times/scores can be added in the future.
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


@dataclass(eq=False)
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
    data_load_time: float = field(init=False, repr=True)
    data_sample_time: float = field(init=False, repr=True)
    transform_params: dict = field(default_factory=dict)
    score_dict: dict = field(init=False, repr=False)

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        self._initialize_timing_fields()
        if not self.data_params:
            self.data_params = {}
        if not hasattr(self, "shuffle"):
            self.shuffle = True

    def _as_dataset(self, obj, split: str, transform):
        if isinstance(obj, str):
            obj = load_class(
                obj,
                **self.dataset_params,
                split=split,
                transform=transform,
            )
            return obj
        elif isinstance(obj, Dataset):
            return obj(**self.dataset_params, split=split, transform=transform)
        raise TypeError(
            f"Invalid dataset object for split '{split}': {type(obj)}",
        )

    def _truncate_dataset(self, dataset: Dataset, size: int):
        assert isinstance(size, int), ValueError(
            f"Size must be an integer. Got: {size}.",
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
        start = time.process_time()
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
        if self.train_size:
            train_ds = self._truncate_dataset(train_ds, self.train_size)
            self.train_n = self.train_size
        else:
            self.train_n = len(train_ds)
        if self.test_size:
            test_ds = self._truncate_dataset(test_ds, size=self.test_size)
            self.test_n = self.test_size
        else:
            self.test_n = len(test_ds)

        # Minimal placeholders to satisfy parent __call__ checks
        self._X = (train_ds, test_ds)
        self._y = (train_ds, test_ds)

        end = time.process_time()
        self.data_load_time = end - start
        # Sampling is already defined by provided train/test splits

        logger.info(
            f"Loaded custom dataset lazily in {self.data_load_time:.2f}s "
            f"(train={self.train_n}, test={self.test_n}).",
        )

    def _sample(self):
        # DataLoader params (lazy loading, no full dataset materialization)
        logger.info("Creating torch data loaders.")
        start = time.process_time()
        batch_size = int(self.data_params.get("batch_size", 32))
        num_workers = int(self.data_params.get("num_workers", 0))
        pin_memory = bool(
            self.data_params.get("pin_memory", self.device != "cpu"),
        )
        train_ds = self._X[0]
        test_ds = self._X[1]
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

        if len(train_sensitive_batches) > 0 or len(test_sensitive_batches) > 0:
            self._sensitive_train = train_sensitive_batches
            self._sensitive_test = test_sensitive_batches
            self._sensitive_all = train_sensitive_batches + test_sensitive_batches

        end = time.process_time()
        self.data_sample_time = end - start

    def __call__(self, data_file=None, score_file=None):
        if data_file is not None and Path(data_file).exists():
            self = self.load_object(data_file)
        if score_file is not None and Path(score_file).exists():
            scores = self.load_scores(score_file)
        else:
            scores = {}
        if not hasattr(self, "_X") or self._X is None:
            self._load_data()
        if not hasattr(self, "X_train"):
            self._sample()
        if not hasattr(self, "score_dict"):
            new_scores = self._classification_feature_scores()
            time_dict = {
                "data_load_time": self.data_load_time,
                "data_sample_time": self.data_sample_time,
                "data_score_time": self.data_score_time,
            }
            scores.update(**new_scores, **time_dict)
            self.score_dict = scores
        if score_file is not None:
            self.save_scores(scores, filepath=score_file)
        if data_file is not None:
            self.save_object(self, data_file)
        return scores
