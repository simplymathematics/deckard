# Imports
import pandas as pd
import time
import logging
import importlib
import tempfile
import sys
from pathlib import Path


from dataclasses import dataclass, field
from typing import Tuple, Union, Dict
from omegaconf import DictConfig, OmegaConf

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision import datasets, transforms


# deckard
from . import DataConfig
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.feature_selection import mutual_info_regression, f_regression
from scipy.stats import pearsonr
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


pytorch_dataset_dict = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "fashionmnist": datasets.FashionMNIST,
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
    batch_size: int = 32
    device : str = "cpu"
    data_dir : str = "./raw_data"
    test_size: Union[float, int, None] = .2
    train_size: Union[float, int, None] = .7
    val_size : Union[float, int, None] = .1
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    pipeline: Dict[str, PytorchDataPipelineConfig] = field(default_factory=dict)
    classifier : bool = True
    target: None = None
    drop: None = None
    keep: None = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Ensure
        assert self.target is None, f"Target variable should not be set for PyTorch datasets. Got {self.target}."
        assert len(self.drop) == 0, f"Drop columns should not be set for PyTorch datasets. Got {self.drop}."
        assert len(self.keep) == 0, f"Keep columns should not be set for PyTorch datasets. Got {self.keep}."
        assert len(self.data_params) == 0, f"data_params should not be set for PyTorch datasets. Got {self.data_params}."
        
        if self.data_dir is None:
            self.data_dir = tempfile.gettempdir()
        assert self.train_size is not None, "train_size must be specified for PyTorch datasets."
        assert self.test_size is not None, "test_size must be specified for PyTorch datasets."
        assert self.val_size is not None, "val_size must be specified for PyTorch datasets."
        
        self.X_val = None
        self.y_val = None
        self.val_n = None

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
        if Path(self.data_dir).exists() and Path(self.data_dir).is_dir() and any(Path(self.data_dir).iterdir()):
            logger.info(f"Using existing data directory at {self.data_dir}.")
            download = False
        else: # Otherwise, create directory and download dataset
            logger.info(f"Data directory {self.data_dir} does not exist. Creating and downloading dataset.")
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
        self._X = torch.cat((train_loader.data, test_loader.data), dim=0)
        self._y = torch.cat((train_loader.targets, test_loader.targets), dim=0)
        end = time.process_time()
        self.data_load_time = end - start
        logger.info(f"Loaded {dataset_name} dataset in {self.data_load_time} seconds.")
        assert isinstance(self._X, Tensor), f"Expected _X to be a tuple, got {type(self._X)}."
        assert isinstance(self._y, Tensor), f"Expected _y to be a tuple, got {type(self._y)}."
        assert isinstance(self.data_load_time, float), "data_load_time is not a float."

    def _sample(self):
        """Sample the dataset if needed."""
        # Sampling logic can be implemented here if required
        total_size = len(self._X)
        train_n = self.train_size if isinstance(self.train_size, int) else int(self.train_size * total_size)
        val_n = self.val_size if isinstance(self.val_size, int) else int(self.val_size * total_size)
        test_n = self.test_size if isinstance(self.test_size, int) else int(self.test_size * total_size)
        remaining = total_size - (train_n + val_n + test_n)
        if remaining > 0:
            test_n += remaining  # Adjust test size to use all data
        start = time.process_time()
        torch_dataset = TensorDataset(self._X, self._y)
        train_data, val_data, test_data = random_split(
            torch_dataset,
            [train_n, val_n, test_n],
            generator=torch.Generator().manual_seed(self.random_state),
        )
        end = time.process_time()
        # Create DataLoaders
        self.X_train = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)
        self.y_train = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)
        self.X_val = DataLoader(val_data.dataset, batch_size=self.batch_size, shuffle=False)
        self.y_val = DataLoader(val_data.dataset, batch_size=self.batch_size, shuffle=False)
        self.X_test = DataLoader(test_data.dataset, batch_size=self.batch_size, shuffle=False)
        self.y_test = DataLoader(test_data.dataset, batch_size=self.batch_size, shuffle=False)
        self.train_n = train_n
        self.val_n = val_n
        self.test_n = test_n
        self.data_sample_time = end - start
        logger.info(f"Sampled dataset in {self.data_sample_time} seconds.")
        assert isinstance(self.X_train, DataLoader), "Sampled training data is not a PyTorch Dataset."
        assert isinstance(self.y_train, DataLoader), "Sampled training targets are not a PyTorch Dataset."
        assert isinstance(self.X_val, DataLoader), "Sampled validation data is not a PyTorch Dataset."
        assert isinstance(self.y_val, DataLoader), "Sampled validation targets are not a PyTorch Dataset."
        assert isinstance(self.X_test, DataLoader), "Sampled test data is not a PyTorch Dataset."
        assert isinstance(self.y_test, DataLoader), "Sampled test targets are not a PyTorch Dataset."
        assert isinstance(self.data_sample_time, float), "data_sample_time is not a float."

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
        # Feature importance logic can be implemented here if required
        
        

    def _regression_feature_scores(self):
        """"
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
            pearson_scores = [pearsonr(X_train_np[:, i], y_train_np)[0] for i in range(X_train_np.shape[1])]
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
                    assert hasattr(self, "data_load_time"), "Loaded object does not have data_load_time attribute."
                    assert hasattr(self, "data_sample_time"), "Loaded object does not have data_sample_time attribute."
            if score_file is not None:
                assert isinstance(score_file, str), "score_file must be a string path."
                if Path(score_file).exists():
                    scores = self.load_scores(score_file)
            else:
                scores = {}
            if self.data_load_time is None:
                self._load_data()
            assert self._X is not None, "_X attribute not found after loading data."
            assert self._y is not None, "_y attribute not found after loading data."
            if self.data_sample_time is None:
                self._sample()
            assert self.X_train is not None, "X_train attribute not found after sampling data."
            assert self.X_test is not None, "X_test attribute not found after sampling data."
            assert self.y_train is not None, "y_train attribute not found after sampling data."
            assert self.y_test is not None, "y_test attribute not found after sampling data."
            assert self.X_val is not None, "X_val attribute not found after sampling data."
            assert self.y_val is not None, "y_val attribute not found after sampling data."
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
            data_scores = self._score()
            all_scores = {**scores, **data_scores, **time_dict}
            self.score_dict = all_scores
            if score_file is not None:
                self.save_scores(scores, score_file)
            if data_file is not None:
                if not Path(data_file).exists():
                    self.save(data_file)
            return all_scores