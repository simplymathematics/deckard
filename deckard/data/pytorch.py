# Imports
import pandas as pd
import time
import logging
import tempfile
from pathlib import Path


from dataclasses import dataclass, field
from typing import Union, Dict

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import random_split, TensorDataset
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
        assert (
            self.train_size > 0
        ), "train_size must be specified for PyTorch datasets."
        assert (
            self.test_size > 0
        ), "test_size must be specified for PyTorch datasets."

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
        stratify_col = None
        if self.stratify is not None:
            if self.stratify is True:
                stratify_col = self._y
            else:
                raise ValueError(f"stratify must be None or True for PyTorch datasets")

        # Calculate train and test sizes
        if isinstance(self.train_size, float):
            train_size = int(self.train_size * num_samples)
        elif isinstance(self.train_size, int):
            train_size = self.train_size
        else:
            assert isinstance(self.test_size, (float, int)), "test_size must be float or int if train_size is None"

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
            raise ValueError("Train size and test size exceed the total number of samples")
        start_time = time.process_time()

    
        # Randomly shuffle indices
        indices = indices[torch.randperm(num_samples, generator=torch.Generator().manual_seed(self.random_state))]
        # The first train_size indices are for training
        train_idx = indices[:train_size]
        # The next test_size indices are for testing
        test_idx = indices[train_size : train_size + test_size]
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
        # Compute metrics
        try:
            mutual_info_scores = mutual_info_classif(X_train_np, y_train_np)
            score_dict["mutual_info_classif"] = mutual_info_scores.tolist()
        except Exception as e:
            logger.warning(f"mutual_info_classif failed with error: {e}")
        try:
            chi2_scores, _ = chi2(X_train_np, y_train_np)
            score_dict["chi2"] = chi2_scores.tolist()
        except Exception as e:
            logger.warning(f"chi2 failed with error: {e}")
        try:
            f_classif_scores, _ = f_classif(X_train_np, y_train_np)
            score_dict["f_classif"] = f_classif_scores.tolist()
        except Exception as e:
            logger.warning(f"f_classif failed with error: {e}")

        # Class counts
        class_counts = pd.Series(y_train_np).value_counts().to_dict()
        score_dict["class_counts"] = class_counts
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
        X_train_np = self.X_train.cpu().numpy()
        y_train_np = self.y_train.cpu().numpy()
        y_test_np = self.y_test.cpu().numpy()

        score_dict = {}
        # Compute metrics
        # Mutual Information Regression
        try:
            mutual_info_scores = mutual_info_regression(X_train_np, y_train_np)
            score_dict["mutual_info_regression"] = mutual_info_scores.tolist()
        except Exception as e:
            logger.warning(f"mutual_info_regression failed with error: {e}")
        # F-regression
        try:
            f_regression_scores, _ = f_regression(X_train_np, y_train_np)
            score_dict["f_regression"] = f_regression_scores.tolist()
        except Exception as e:
            logger.warning(f"f_regression failed with error: {e}")
        # Pearson Correlation
        try:
            pearson_scores = [
                pearsonr(X_train_np[:, i], y_train_np)[0]
                for i in range(X_train_np.shape[1])
            ]
            score_dict["r_regression"] = pearson_scores
        except Exception as e:
            logger.warning(f"pearsonr failed with error: {e}")
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
                self.score_dict = {**self.score_dict, **scores} if hasattr(self, "score_dict") else scores
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
        assert len(self.score_dict) >= 3, f"Score dictionary does not contain expected number of entries. Found these keys: {list(self.score_dict.keys())}"
        logger.info(f"Computed scores: {list(self.score_dict.keys())}")
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
