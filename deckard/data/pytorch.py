# Imports
import pandas as pd
import time
import logging
import tempfile
from tqdm.auto import tqdm
from pathlib import Path
from hashlib import md5


from dataclasses import dataclass, field
from typing import Union, List, Optional, cast, Callable

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

# deckard
from ..utils import load_class
from .data import DataConfig, DataPipelineConfig
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class PytorchDataPipelineConfig(DataPipelineConfig):
    pass


@dataclass
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
    device: str = "cpu"
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

    def __post_init__(self):
        super().__post_init__()

        assert (
            self.target is None
        ), f"Target variable should not be set for PyTorch datasets. Got {self.target}."
        assert (
            len(self.drop) == 0
        ), f"Drop columns should not be set for PyTorch datasets. Got {self.drop}."
        assert (
            len(self.keep) == 0
        ), f"Keep columns should not be set for PyTorch datasets. Got {self.keep}."

        if self.data_dir is None:
            self.data_dir = tempfile.gettempdir()
        assert (
            self.train_size is not None and self.train_size > 0
        ), "train_size must be specified for PyTorch datasets."
        assert (
            self.test_size is not None and self.test_size > 0
        ), "test_size must be specified for PyTorch datasets."

        # Normalize data_params
        if self.data_params is None:
            self.data_params = {}
        # Only inject torchvision-style root automatically.
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

        self.data_load_time = None
        self.data_sample_time = None
        self.data_score_time = None

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
                        f"Unknown dataset alias '{dataset_name}'. Use a fully qualified class path."
                    )
                dataset_name = dataset_aliases[dataset_name.lower()]

            # Instantiate the dataset using load_class.
            full_dataset = load_class(dataset_name, **self.data_params)

            # Extract data and labels from dataset
            samples = [full_dataset[i] for i in range(len(full_dataset))]

            # Stack tensors and labels
            if isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:
                X_list = [
                    s[0] if isinstance(s[0], Tensor) else torch.tensor(s[0])
                    for s in samples
                ]
                y_list = [
                    s[1] if isinstance(s[1], (int, Tensor)) else torch.tensor(s[1])
                    for s in samples
                ]
                self._X = torch.stack(X_list)
                self._y = (
                    torch.stack(y_list)
                    if isinstance(y_list[0], Tensor)
                    else torch.tensor(y_list)
                )
            else:
                raise ValueError(
                    f"Dataset samples must be (X, y) tuples, got {type(samples[0])}"
                )

            end = time.process_time()
            self.data_load_time = end - start
            logger.info(
                f"Loaded dataset {self.dataset_name} in {self.data_load_time:.2f} seconds. "
                f"Shape: {self._X.shape}, Labels: {self._y.shape}"
            )

            assert isinstance(
                self._X, Tensor
            ), f"Expected _X to be Tensor, got {type(self._X)}"
            assert isinstance(
                self._y, Tensor
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
        Sets ``self._train_indices``, ``self._test_indices``, and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Call _load_data first.")

        num_samples = len(self._X)
        indices = torch.arange(num_samples)
        # Determine stratification
        if self.stratify is not None:
            if self.stratify is not True:
                raise ValueError(
                    f"stratify must be None or True for PyTorch datasets; got {self.stratify}."
                )

        # Calculate train and test sizes
        train_size: int
        test_size: int

        if self.train_size is None and self.test_size is None:
            raise ValueError("Either train_size or test_size must be specified.")

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
        test_idx = indices[train_size : train_size + test_size]

        end_time = time.process_time()
        self.data_sample_time = end_time - start_time

        # Split the data
        self.X_train = self._X[train_idx]
        self.y_train = self._y[train_idx]
        self.X_test = self._X[test_idx]
        self.y_test = self._y[test_idx]

        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)

        logger.info(
            f"Data sampled in {self.data_sample_time:.2f} seconds. "
            f"Train: {self.train_n}, Test: {self.test_n}"
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
        y_train_cdf = np.arange(1, len(y_train_sorted) + 1) / len(y_train_sorted)
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
            assert isinstance(data_file, str), "data_file must be a string path."
            if not Path(data_file).exists():
                Path(data_file).parent.mkdir(parents=True, exist_ok=True)
            else:
                pass

        if score_file is not None:
            assert isinstance(score_file, str), "score_file must be a string path."
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


@dataclass
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
        """
        Computes a hash value for the instance.

        Concatenates all non-private attribute names and values, then hashes the resulting string using MD5.
        The hash excludes attributes whose names start with an underscore.

        Returns
        -------
        int
            The integer representation of the MD5 hash of the concatenated attribute string.
        """
        # Hash all fields that do not start with an underscore
        hash_input = "".join(
            f"{k}:{v}" for k, v in self.__dict__.items() if not k.endswith("_")
        )
        return int(md5(hash_input.encode()).hexdigest(), 16)

    def __post_init__(self):

        self.data_load_time = None
        self.data_sample_time = None
        self.data_score_time = None
        if not self.data_params:
            self.data_params = {}
        if not hasattr(self, "shuffle"):
            self.shuffle = True

    def _as_dataset(self, obj, split: str, transform):
        if isinstance(obj, str):
            obj = load_class(obj, split=split, transform=transform)
            return obj
        elif isinstance(obj, Dataset):
            return obj(**self.dataset_params, split=split, transform=transform)
        raise TypeError(f"Invalid dataset object for split '{split}': {type(obj)}")

    def _truncate_dataset(self, dataset: Dataset, size: int):
        assert isinstance(size, int), ValueError(
            f"Size must be an integer. Got: {size}."
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
            self.dataset, split="train", transform=train_transform
        )
        test_ds = self._as_dataset(
            self.dataset, split=valid_split, transform=test_transform
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
        pin_memory = bool(self.data_params.get("pin_memory", self.device != "cpu"))
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
        for _, yb in tqdm(
            train_loader,
            desc="Materializing train batches",
            total=len(train_loader),
            leave=False,
        ):
            train_y_batches.append(yb)

        test_y_batches = []
        for _, yb in tqdm(
            test_loader,
            desc="Materializing test batches",
            total=len(test_loader),
            leave=False,
        ):
            test_y_batches.append(yb)

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

        end = time.process_time()
        self.data_sample_time = end - start

    def __call__(self, data_file=None, score_file=None):
        if data_file is not None and Path(data_file).exists():
            self = self.load_object(data_file)
        if score_file is not None and Path(score_file).exists():
            scores = self.load_scores(score_file)
        else:
            scores = {}
        if not hasattr(self, "X_"):
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
        if "train_n" in getattr(self, "score_dict", {}):
            return self.score_dict
        score_dict = {}
        score_dict["train_n"] = len(self.X_train)
        score_dict["test_n"] = len(self.X_test)
        return score_dict
