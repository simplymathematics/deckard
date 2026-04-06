# Imports
# import pandas as pd
import time
import logging
import tempfile
import importlib
from pathlib import Path


from dataclasses import dataclass, field
from typing import Union, Dict, Callable, List

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms



# deckard
from deckard.data import DataConfig
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    f_classif,
    mutual_info_regression,
    f_regression,
)
from scipy.stats import pearsonr
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


pytorch_dataset_dict = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "fashionmnist": datasets.FashionMNIST,
    "torch_mnist": datasets.MNIST,
    "torch_cifar10": datasets.CIFAR10,
    "torch_cifar": datasets.CIFAR10,
    "torch_fashionmnist": datasets.FashionMNIST,
    # Add more datasets as needed
}


@dataclass
class PytorchDataPipelineConfig:
    pass


@dataclass
class PytorchDataConfig(DataConfig):
    """Configuration for PyTorch datasets.

    Attributes:
        dataset_name (str): Name of the dataset to load.
        data_params (dict): Additional parameters for dataset loading.
        test_size (Union[float, int, None]): Proportion or absolute number of test samples.
        train_size (Union[float, int, None]): Proportion or absolute number of train samples.
        random_state (int): Random seed for reproducibility.
        stratify (Union[None, str, bool]): Whether to stratify the split.
        pipeline (Dict[str, DataPipelineConfig]): Data processing pipelines.

    """

    dataset_name: str = "mnist"
    device: str = "cpu"
    data_dir: str = "./raw_data"
    test_size: Union[float, int, None] = 0.2
    train_size: Union[float, int, None] = 0.7
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    pipeline: Dict[str, PytorchDataPipelineConfig] = field(default_factory=dict)
    classifier: bool = True
    target: None = None
    drop: None = None
    keep: None = None

    def __post_init__(self):
        super().__post_init__()

        # Ensure
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
            len(self.data_params) == 0
        ), f"data_params should not be set for PyTorch datasets. Got {self.data_params}."

        if self.data_dir is None:
            self.data_dir = tempfile.gettempdir()
        assert self.train_size > 0, "train_size must be specified for PyTorch datasets."
        assert self.test_size > 0, "test_size must be specified for PyTorch datasets."

    def __hash__(self):
        return super().__hash__()

    def _load_data(self) -> datasets.VisionDataset:
        """Load a PyTorch dataset.

        Args:
            dataset_name (str): Name of the dataset to load.

        Returns:
            datasets.VisionDataset: The loaded dataset.
        """
        dataset_name = self.dataset_name.lower()
        if dataset_name not in pytorch_dataset_dict:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
        dataset_class = pytorch_dataset_dict[dataset_name]

        # Check if data directory exists, is a directory, and is non-empty
        if (
            Path(self.data_dir).exists()
            and Path(self.data_dir).is_dir()
            and any(Path(self.data_dir).iterdir())
        ):
            logger.info(f"Using existing data directory at {self.data_dir}.")
            download = False
        else:  # Otherwise, create directory and download dataset
            logger.info(
                f"Data directory {self.data_dir} does not exist. Creating and downloading dataset.",
            )
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
            download = True
        start = time.process_time()
        train_loader = dataset_class(
            root=self.data_dir,
            train=True,
            download=download,
            transform=transforms.ToTensor(),
        )
        test_loader = dataset_class(
            root=self.data_dir,
            train=False,
            download=download,
            transform=transforms.ToTensor(),
        )
        # Combine train and test datasets
        full_data = torch.utils.data.ConcatDataset([train_loader, test_loader])
        # Extract data and targets
        self._X = torch.stack([full_data[i][0] for i in range(len(full_data))])
        self._y = torch.tensor([full_data[i][1] for i in range(len(full_data))])
        end = time.process_time()
        self.data_load_time = end - start
        logger.info(f"Loaded {dataset_name} dataset in {self.data_load_time} seconds.")
        assert isinstance(
            self._X,
            Tensor,
        ), f"Expected _X to be a tuple, got {type(self._X)}."
        assert isinstance(
            self._y,
            Tensor,
        ), f"Expected _y to be a tuple, got {type(self._y)}."
        assert isinstance(self.data_load_time, float), "data_load_time is not a float."

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
        if not hasattr(self, "_X") or self._X is None:
            raise ValueError("Data not loaded. Cannot sample.")
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Cannot sample.")

        num_samples = len(self._X)
        indices = torch.arange(num_samples)
        # Determine stratification
        if self.stratify is not None:
            if self.stratify is True:
                _ = self._y
            else:
                raise ValueError(
                    f"stratify must be None or True for PyTorch datasets: got {self.stratify}.",
                )

        # Calculate train and test sizes
        if isinstance(self.train_size, float):
            train_size = int(self.train_size * num_samples)
        elif isinstance(self.train_size, int):
            train_size = self.train_size
        else:
            assert isinstance(
                self.test_size,
                (float, int),
            ), "test_size must be float or int if train_size is None"

        if isinstance(self.test_size, float):
            test_size = int(self.test_size * num_samples)
        else:
            test_size = self.test_size

        if self.train_size is None:
            if isinstance(self.test_size, float):
                self.train_size = 1.0 - self.test_size
                train_size = int(self.train_size * num_samples)
            elif isinstance(self.test_size, int):
                self.train_size = num_samples - self.test_size
                train_size = self.train_size
            else:
                raise ValueError("Either train_size or test_size must be specified.")

        if train_size + test_size > num_samples:
            raise ValueError(
                "Train size and test size exceed the total number of samples",
            )
        start_time = time.process_time()

        # Randomly shuffle indices
        indices = indices[
            torch.randperm(
                num_samples,
                generator=torch.Generator().manual_seed(self.random_state),
            )
        ]
        # The first train_size indices are for training
        train_idx = indices[:train_size]
        # The next test_size indices are for testing
        test_idx = indices[train_size : train_size + test_size]  # NOQA E203
        end_time = time.process_time()
        self.data_sample_time = end_time - start_time
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")

        # Split the data
        self.X_train = self._X[train_idx]
        self.y_train = self._y[train_idx]
        self.X_test = self._X[test_idx]
        self.y_test = self._y[test_idx]
        logger.info(
            f"Training samples: {len(self.X_train)}, Testing samples: {len(self.X_test)}",
        )
        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)

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
            return self.score_dict

        # Ensure data is on CPU for compatibility with sklearn
        X_train_np = self._X.cpu().numpy()
        y_train_np = self._y.cpu().numpy()

        score_dict = {}
        

        # Class counts
        score_dict["class_counts"] = self._compute_class_counts(self.y_train)
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
            return self.score_dict

        # Ensure data is on CPU for compatibility with sklearn
        y_train_np = self.y_train.cpu().numpy()
        y_test_np = self.y_test.cpu().numpy()

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
            return self._classification_feature_scores()
        else:
            return self._regression_feature_scores()

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
                self = self.load_object(data_file)
                assert hasattr(
                    self,
                    "data_load_time",
                ), "Loaded object does not have data_load_time attribute."
                assert hasattr(
                    self,
                    "data_sample_time",
                ), "Loaded object does not have data_sample_time attribute."
        if score_file is not None:
            assert isinstance(score_file, str), "score_file must be a string path."
            if Path(score_file).exists():
                scores = self.load_scores(score_file)
                self.score_dict = (
                    {**self.score_dict, **scores}
                    if hasattr(self, "score_dict")
                    else scores
                )
            else:
                scores = {}
        else:
            scores = {}
        if self.data_load_time is None:
            self._load_data()
        assert self._X is not None, "_X attribute not found after loading data."
        assert self._y is not None, "_y attribute not found after loading data."
        if self.data_sample_time is None:
            self._sample()
        assert (
            self.X_train is not None
        ), "X_train attribute not found after sampling data."
        assert (
            self.X_test is not None
        ), "X_test attribute not found after sampling data."
        assert (
            self.y_train is not None
        ), "y_train attribute not found after sampling data."
        assert (
            self.y_test is not None
        ), "y_test attribute not found after sampling data."
        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
        }
        logger.info(
            f"Data loaded and sampled. Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
        )
        if data_file is not None:
            # Save data logic can be implemented here if required
            pass
        logger.info(
            f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
        )

        scores = self._score()
        all_scores = {**time_dict, **scores}
        self.score_dict = all_scores
        if score_file is not None:
            self.save_scores(scores, score_file)
        if data_file is not None:
            if not Path(data_file).exists():
                self.save(data_file)
        assert hasattr(
            self,
            "score_dict",
        ), "score_dict attribute not found after scoring data."
        return all_scores


@dataclass
class PytorchCustomConfig(PytorchDataConfig):
    """Configuration for HuggingFace datasets loaded via DataLoader.

    Extends PytorchDataConfig to support HuggingFace datasets with custom
    transforms and DataLoader-based loading.
    """

    val: bool = False
    batch_size: int = 32
    num_workers: int = 0
    datasets: List[Union[str, Dataset]] = field(default_factory=list)

    def __hash__(self):
        return super().__hash__()

    def _load_data(self):
        start = time.process_time()
        self.dataloaders = self._get_data_loaders()
        self._X = torch.empty(0)
        self._y = torch.empty(0, dtype=torch.long)
        end = time.process_time()
        self.data_load_time = end - start

    def _get_dataset_from_string(self, file: str) -> Dataset:
        if not isinstance(file, str):
            raise TypeError(f"Dataset reference must be a string, got {type(file)}")
        # Load the custom dataset from submodule.file.dataset
        if "." not in file:
            raise ValueError(
                f"Dataset string must be in 'module.object' format, got '{file}'.",
            )
        module_name, object_name = file.rsplit(".", 1)
        module = importlib.import_module(module_name)
        dataset_obj = getattr(module, object_name)
        if isinstance(dataset_obj, Dataset):
            return dataset_obj
        if isinstance(dataset_obj, type) and issubclass(dataset_obj, Dataset):
            return dataset_obj()
        raise TypeError(f"Resolved object '{file}' is not a torch Dataset.")

    def __post_init__(self):
        super().__post_init__()
        # Iterate over self.datasets. Ensure that each object is a string or a pytorch Dataset. If it is a string, load it and ensure it is a pytorch Dataset. Assume it has the form file.dataset
        if not isinstance(self.datasets, list):
            raise TypeError("datasets must be a list of Dataset objects or strings.")
        normalized_datasets = []
        for ds in self.datasets:
            if isinstance(ds, str):
                ds = self._get_dataset_from_string(ds)
            if not isinstance(ds, Dataset):
                raise TypeError(f"Invalid dataset entry type: {type(ds)}")
            normalized_datasets.append(ds)
        if len(normalized_datasets) < 2:
            raise ValueError("Provide at least [train_dataset, test_dataset].")
        self.datasets = normalized_datasets

    def _get_data_loaders(self):
        train_dataset = self.datasets[0]
        test_dataset = self.datasets[1]
        val_dataset = self.datasets[2] if len(self.datasets) > 2 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            if val_dataset is not None
            else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader, val_loader

    @staticmethod
    def _extract_targets(dataset):
        if hasattr(dataset, "targets"):
            targets = dataset.targets
            return targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        if hasattr(dataset, "labels"):
            labels = dataset.labels
            return labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        return torch.empty(0, dtype=torch.long)
    
    def _resolve_size(size, total, name):
        if size is None:
            return total
        if isinstance(size, float):
            if not (0 < size <= 1):
                raise ValueError(f"{name} as float must be in (0, 1], got {size}.")
            return max(1, int(total * size))
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"{name} as int must be > 0, got {size}.")
            return min(size, total)
        raise TypeError(f"{name} must be None, float, or int, got {type(size)}.")

    def _truncate_loader(loader, n, shuffle):
        if n >= len(loader.dataset):
            return loader
        subset = torch.utils.data.Subset(loader.dataset, range(n))
        return DataLoader(
        subset,
        batch_size=self.batch_size,
        shuffle=shuffle,
        num_workers=self.num_workers,
        pin_memory=True,
        )

    def _sample(self):
        """Use pre-split train/test (or val) loaders and optionally truncate them."""
        start_time = time.process_time()
        train_loader, test_loader, val_loader = self.dataloaders

        # Resolve eval split
        eval_loader = val_loader if self.val and val_loader is not None else test_loader

        # Base targets
        y_train_full = self._extract_targets(train_loader.dataset)
        y_test_full = self._extract_targets(eval_loader.dataset)

        # Resolve sizes
        train_total = len(train_loader.dataset)
        test_total = len(eval_loader.dataset)
        train_n = PytorchCustomConfig._resolve_size(self.train_size, train_total, "train_size")
        test_n = PytorchCustomConfig._resolve_size(self.test_size, test_total, "test_size")

        # Truncate loaders lazily if needed
        self.X_train = self._truncate_loader(train_loader, train_n, shuffle=True)
        self.X_test = self._truncate_loader(eval_loader, test_n, shuffle=False)

        # Truncate targets eagerly
        self.y_train = y_train_full[:train_n]
        self.y_test = y_test_full[:test_n]

        # Track counts and timing
        self.train_n = len(self.y_train)
        self.test_n = len(self.y_test)
        self.data_sample_time = time.process_time() - start_time
        self.time_dict = {
            **getattr(self, "time_dict", {}),
            "data_sample_time": self.data_sample_time,
        }

        logger.info(
            "Custom data sampled in %.2f seconds (train=%d, test=%d)",
            self.data_sample_time,
            self.train_n,
            self.test_n,
        )

    def __call__(self, data_file=None, score_file=None):
        self._load_data()
        self._sample()
        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
        }
        scores = self._score()
        all_scores = {**time_dict, **scores}
        self.score_dict = all_scores
        if score_file is not None:
            self.save_scores(scores, score_file)
        if data_file is not None:
            if not Path(data_file).exists():
                self.save(data_file)
        return all_scores
    