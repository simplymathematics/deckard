# Imports
import pandas as pd
import time
import logging
import importlib
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, Tuple, Union, TYPE_CHECKING
from omegaconf import DictConfig

import numpy as np

# Scikit-learn
from sklearn.datasets import (
    fetch_openml,
    make_classification,
    make_regression,
    load_digits,
    load_diabetes,
    load_iris,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
    r_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer

from scipy.sparse import csr_matrix

# deckard
from ..utils import ConfigBase, data_supported_filetypes, load_class

# Setup logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..score import ScorerDictConfig


def _discover_lifelines_dataset_loaders() -> dict:
    """Discover lifelines dataset loader functions when lifelines is installed."""
    try:
        lifelines_datasets = importlib.import_module("lifelines.datasets")
    except ImportError:
        return {}
    dataset_map = {}
    for attr in dir(lifelines_datasets):
        if not attr.startswith("load_"):
            continue
        loader = getattr(lifelines_datasets, attr)
        if callable(loader):
            dataset_name = attr.replace("load_", "", 1)
            dataset_map[dataset_name] = loader
    return dataset_map


def _lifelines_dataset_loaders() -> dict:
    return _discover_lifelines_dataset_loaders()


@dataclass
class DataConfig(ConfigBase):
    """
    Configuration and utility class for loading, preprocessing, and splitting datasets for machine learning tasks.

    Attributes
    -------
    dataset_name : str
        Name of the dataset to load or path to a data file.
    data_params : dict
        Additional parameters for data loading or generation.
    test_size : float
        Proportion of the dataset to include in the test split (between 0 and 1).
    train_size : float
        Proportion or count of samples to include in the training split.
    random_state : int
        Seed for random number generation to ensure reproducibility.
    stratify : Union[None, str, bool]
        Specifies stratification for sampling; can be None, True (use target), or a column name.
    classifier: bool
        Whether the task is classification (True) or regression (False).
    drop: list
        List of columns to drop from the dataset.
    target: Union[str, None]
        Name of the target column in the dataset (if applicable).
    keep: list
        List of columns to keep in the dataset.
    plugins : list
        Optional data plugin specifications executed during load/sample/score hooks.
    _X : pd.DataFrame
        Loaded feature matrix.
    _y : pd.Series
        Loaded target vector.
    data_load_time : float
        Time taken to load the data.
    data_sample_time : float
        Time taken to sample/split the data.
    train_n : int
        Number of training samples.
    test_n : int
        Number of testing samples.
    alias: str
        Optional alias for the dataset configuration.
    train_indices : list
        Indices for training samples.
    test_indices : list
        Indices for testing samples.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target vector.
    score_dict : dict
        Dictionary to store scores or metrics.
    _target_ : str
        Internal identifier for the class.

    Methods
    -------
    __post_init__()
        Post-initialization method to validate parameters and initialize internal attributes.
    __hash__()
        Computes a hash value for the instance based on non-private attributes.
    _load_adult_income_data()
        Loads and preprocesses the Adult Income dataset from OpenML.
    _load_generic_sklearn(loader_func, **loader_params)
        Loads a dataset using a generic scikit-learn loader function.
    _load_generic_openml(dataset_name, version=1, **loader_params)
        Loads a dataset from OpenML using the specified dataset name and version.
    _make_classification_data()
        Generates a synthetic classification dataset.
    _make_regression_data()
        Generates a synthetic regression dataset.
    _sample()
        Splits the loaded dataset into training and testing sets, optionally using stratification.
    _load_data()
        Loads the dataset based on the specified name or file type.
    __call__(filepath=None)
        Loads and samples the dataset, splits it into training and testing sets, and returns the corresponding features and labels.
    save(filepath)
        Saves the current state of the DataConfig instance to a file.
    load(filepath)
        Loads the state of the DataConfig instance from a file.
    Raises
    ------
    ValueError
        For invalid parameter values or missing data.
    NotImplementedError
        For unsupported datasets or file types.

    Examples
    --------
    >>> config = DataConfig(dataset_name="adult", **kwargs)
    >>> config()
    >>> X_train = config.X_train
    >>> y_train = config.y_train
    >>> X_test = config.X_test
    >>> y_test = config.y_test
    >>> score_dict = config.score_dict
    """

    # Configuration fields
    dataset_name: str = "adult"
    data_params: dict = None
    test_size: Union[float, int, None] = None
    train_size: Union[float, int, None] = None
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    classifier: Union[bool, None, str] = True
    target: Union[str, None] = None
    drop: list = None
    keep: list = None
    plugins: list = field(default_factory=list)
    alias: Union[str, None] = None
    scorer: Union["ScorerDictConfig", None] = None

    # Runtime state fields
    score_dict: dict = field(init=False, repr=True)
    data_load_time: Union[float, None] = None
    data_sample_time: Union[float, None] = None
    _X: Union[pd.DataFrame, pd.Series, None] = None
    _y: Union[pd.Series, None] = None
    train_indices: Union[list, None] = None
    test_indices: Union[list, None] = None
    X_train: Union[pd.DataFrame, pd.Series, None] = None
    y_train: Union[pd.Series, None] = None
    X_test: Union[pd.DataFrame, pd.Series, None] = None
    y_test: Union[pd.Series, None] = None
    train_n: Union[int, None] = None
    test_n: Union[int, None] = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = None

    def _validate_init(self):
        """
        Post-initialization method for setting up data-related attributes.

        Validates that `test_size` is between 0 and 1, then initializes training size and internal attributes
        for data loading, sampling, parameters, and train/test splits.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.
        """
        if self.train_size is None:
            if self.test_size is None:
                self.test_size = 0.2
                self.train_size = 0.8
            else:
                if isinstance(self.test_size, float):
                    if not (0 < self.test_size < 1):
                        raise ValueError("test_size must be between 0 and 1")
                    self.train_size = 1 - self.test_size
                elif isinstance(self.test_size, int):
                    self.train_size = None
                else:
                    raise ValueError("test_size must be a float or int")
        self.data_params = self.data_params if self.data_params is not None else {}
        self.drop = [] if not hasattr(self, "drop") or self.drop is None else self.drop
        self.keep = [] if not hasattr(self, "keep") or self.keep is None else self.keep
        for attr in [
            "data_load_time",
            "data_sample_time",
            "_X",
            "_y",
            "train_indices",
            "test_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "train_n",
            "test_n",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}

        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.data.DataConfig"
        if not self.data_params:
            self.data_params = {}

    def __post_init__(self):
        self._validate_init()

    @property
    def X(self):
        """Convenience alias for the loaded feature matrix."""
        return self._X

    @property
    def y(self):
        """Convenience alias for the loaded target vector."""
        return self._y

    def _instantiate_plugin(self, plugin_spec: Any):
        if isinstance(plugin_spec, dict):
            spec = dict(plugin_spec)
            class_path = spec.pop("name", spec.pop("_target_", None))
            if class_path is None:
                raise ValueError("Plugin dict must include 'name' or '_target_'")
            return load_class(class_path, **spec)

        if isinstance(plugin_spec, str):
            return load_class(plugin_spec)

        if isinstance(plugin_spec, type):
            return plugin_spec()

        return plugin_spec

    def _get_plugins(self) -> list:
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            plugin_specs = self.plugins if self.plugins is not None else []
            if not isinstance(plugin_specs, list):
                raise TypeError(f"plugins must be a list, got {type(plugin_specs)}")
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs):
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    def __hash__(self):
        return super().__hash__()

    def _load_adult_income_data(self):
        """
        Loads and preprocesses the Adult Income dataset from OpenML.

        Steps performed:
            - Fetches the dataset using the specified name and version.
            - Separates features (X) and target variable (y).
            - Converts the target variable 'class' to binary integers (0 for '<=50K', 1 for '>50K').
            - Encodes the 'sex' column as binary (0 for Male, 1 for Female).
            - Converts relevant columns to appropriate numeric types.
            - Converts categorical columns to category dtype.
            - Applies one-hot encoding to categorical features, dropping the first category.
            - Records the time taken to load and preprocess the data.
            - Stores processed features and target in instance variables.

        Returns
        -------
        self : DataConfig
            The instance with loaded and preprocessed data.
        """
        start_time = time.process_time()
        adult = fetch_openml(name=self.dataset_name, version=2, as_frame=True)
        df = adult.frame
        X = df.drop(columns="class")
        y = df["class"].cat.rename_categories({"<=50K": 0, ">50K": 1})
        y = y.astype(int)
        # Replace Male/Female with 1/0
        sex = X.pop("sex")
        # Convert appropriate columns to categorical or numeric types
        X["age"] = X["age"].astype(int)
        X["education-num"] = X["education-num"].astype(int)
        X["hours-per-week"] = X["hours-per-week"].astype(int)
        X["capital-gain"] = X["capital-gain"].astype(int)
        X["capital-loss"] = X["capital-loss"].astype(int)
        X["race"] = X["race"].astype("category")
        X["native-country"] = X["native-country"].astype("category")
        X = pd.get_dummies(X, drop_first=True)
        X["sex"] = sex.cat.rename_categories({"Male": 0, "Female": 1})
        # Convert categorical variables to numeric using one-hot encoding
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = X
        self._y = pd.Series(y)
        assert isinstance(
            self._X,
            pd.DataFrame,
        ), f"Expected DataFrame got {type(self._X)}"
        assert isinstance(self._y, pd.Series), f"Expected Series got {type(self._y)}"
        self._X = self._X.apply(pd.to_numeric, errors="coerce")
        return self

    def _make_classification_data(
        self,
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42,
        **kwargs,
    ):
        """
        Generates a synthetic classification dataset and stores it as instance attributes.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Default is 1000.
        n_features : int, optional
            Total number of features. Default is 20.
        n_informative : int, optional
            Number of informative features. Default is 10.
        n_redundant : int, optional
            Number of redundant features. Default is 5.
        n_clusters_per_class : int, optional
            Number of clusters per class. Default is 2.
        random_state : int, optional
            Seed for random number generation. Default is 42.

        Returns
        -------
        self : DataConfig
            The instance with loaded data and timing information.

        Side Effects
        ------------
        Sets self._X (pd.DataFrame): Feature matrix.
        Sets self._y (pd.Series): Target vector.
        Sets self.data_load_time (float): Time taken to generate the data.
        """
        start_time = time.process_time()
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            random_state=random_state,
            **kwargs,
        )
        self._X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self._y = pd.Series(y)
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        return self

    def _make_regression_data(
        self,
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    ):
        """
        Generates synthetic regression data using scikit-learn's make_regression function and stores it as pandas DataFrame and Series.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Default is 1000.
        n_features : int, optional
            Total number of features. Default is 20.
        n_informative : int, optional
            Number of informative features. Default is 10.
        noise : float, optional
            Standard deviation of the gaussian noise applied to the output. Default is 0.1.
        random_state : int, optional
            Seed for the random number generator. Default is 42.

        Returns
        -------
        self : DataConfig
            The instance with generated data stored in self._X (DataFrame), self._y (Series), and self.data_load_time (float).
        """
        start_time = time.process_time()
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state,
        )
        self._X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self._y = pd.Series(y)
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        return self

    def _sample(
        self,
    ):
        """
        Samples training and testing indices from the loaded dataset, optionally using stratification.

        Calculates the number of samples for training and testing based on ``train_size`` and ``test_size``.
        Supports stratified sampling using the target variable or a specified column.
        Splits the data into training and testing sets, records the sampling time, and stores the resulting indices.

        Raises
        ------
        ValueError
            If data is not loaded, or if the specified stratify column is not found, or if ``stratify`` is invalid.

        Side Effects
        ------------
        Sets ``self.train_indices``, ``self.test_indices``, and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        self._run_plugin_hook("before_sample")
        stratify_col = None
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Cannot sample.")
        if self.stratify is not None:
            if self.stratify is True:
                stratify_col = self._y
            elif isinstance(self.stratify, str):
                if self.stratify in self._X.columns:
                    stratify_col = self._X[self.stratify]
                else:
                    raise ValueError(
                        f"Stratify column {self.stratify} not found in data columns",
                    )
            elif self.stratify is False:
                stratify_col = None
            else:
                raise ValueError("stratify must be None, True, or a column name")
        indices = range(len(self._X))
        start_time = time.process_time()
        try:
            train_idx, test_idx = train_test_split(
                indices,
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_col if self.stratify is not None else None,
            )
        except ValueError as e:
            raise ValueError(
                f"Error during train/test split with train_size={self.train_size}, test_size={self.test_size}, random_state={self.random_state}, stratify={self.stratify}: {e} ",
            )
        end_time = time.process_time()
        self.data_sample_time = end_time - start_time
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")
        self.train_indices = train_idx
        self.test_indices = test_idx
        self.X_train = self._X.iloc[self.train_indices].reset_index(drop=True)
        self.y_train = self._y.iloc[self.train_indices].reset_index(drop=True)
        self.X_test = self._X.iloc[self.test_indices].reset_index(drop=True)
        self.y_test = self._y.iloc[self.test_indices].reset_index(drop=True)
        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)
        assert isinstance(
            self.X_train,
            (pd.DataFrame, pd.Series),
        ), "X_train must be a DataFrame"
        assert isinstance(self.y_train, pd.Series), "y_train must be a Series"
        assert isinstance(
            self.X_test,
            (pd.DataFrame, pd.Series),
        ), "X_test must be a DataFrame"
        assert isinstance(self.y_test, pd.Series), "y_test must be a Series"
        assert (
            hasattr(self, "train_indices") and self.train_indices is not None
        ), "Train indices must be set after sampling"
        assert (
            hasattr(self, "test_indices") and self.test_indices is not None
        ), "Test indices must be set after sampling"
        assert isinstance(
            self.X_train,
            (pd.DataFrame, pd.Series),
        ), f"X_train must be a DataFrame or Series, got {type(self.X_train)}"
        assert isinstance(
            self.y_train,
            pd.Series,
        ), f"y_train must be a Series, got {type(self.y_train)}"
        assert isinstance(
            self.X_test,
            (pd.DataFrame, pd.Series),
        ), f"X_test must be a DataFrame or Series, got {type(self.X_test)}"
        assert isinstance(
            self.y_test,
            pd.Series,
        ), f"y_test must be a Series, got {type(self.y_test)}"
        self._run_plugin_hook("after_sample")

    def _load_generic_sklearn(self, loader_func, **loader_params):
        """
        Loads a dataset using a generic scikit-learn loader function.

        Parameters
        ----------
        loader_func : callable
            A scikit-learn dataset loader function that returns a Bunch object with 'data' and 'target' attributes.
        loader_params : dict
            Additional parameters to pass to the loader function.

        Returns
        -------
        self : DataConfig
            The instance with loaded data and timing information.

        Side Effects
        ------------
        Sets ``self._X``, ``self._y``, and ``self.data_load_time`` with loaded data and timing information.
        """
        start_time = time.process_time()
        dataset = loader_func(**loader_params)
        X = dataset.data
        y = dataset.target
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = pd.DataFrame(X)
        self._y = pd.Series(y)
        return self

    def _load_generic_openml(self, dataset_name, version=1, **loader_params):
        """
        Loads a dataset from OpenML using the specified dataset name and version.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load from OpenML.
        version : int, optional
            The version of the dataset to load. Default is 1.
        loader_params : dict
            Additional parameters to pass to the fetch_openml function.

        Returns
        -------
        self : DataConfig
            The instance with loaded data and timing information.

        Side Effects
        ------------
        Sets ``self._X``, ``self._y``, and ``self.data_load_time`` with loaded data and timing information.
        """
        start_time = time.process_time()
        dataset = fetch_openml(
            name=dataset_name,
            version=version,
            as_frame=True,
            **loader_params,
        )
        X = dataset.data
        y = dataset.target
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = pd.DataFrame(X)
        self._y = pd.Series(y)
        return self

    def _load_lifelines_dataset(self, dataset_name: str, **loader_params):
        """Load a lifelines dataset into DataConfig feature/target fields."""
        lifelines_datasets = _lifelines_dataset_loaders()
        if not lifelines_datasets:
            raise ImportError(
                "Lifelines datasets require optional dependency deckard[lifelines]",
            )
        if dataset_name not in lifelines_datasets:
            raise NotImplementedError(
                f"Lifelines dataset {dataset_name} not found. Supported: {sorted(lifelines_datasets.keys())}",
            )
        start_time = time.process_time()
        loader = lifelines_datasets[dataset_name]
        dataset = loader(**loader_params)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        candidate_target = self.target
        if candidate_target is None:
            for candidate in ["E", "event", "status", "status_group"]:
                if candidate in dataset.columns:
                    candidate_target = candidate
                    break
        if candidate_target is None or candidate_target not in dataset.columns:
            # Fall back to a zero target when no event column is present.
            candidate_target = "event"
            dataset[candidate_target] = 0

        y = dataset.pop(candidate_target)
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = dataset
        self._y = pd.Series(y)
        return self

    def _load_data(self):
        """
        Loads dataset based on the provided dataset name or file type.

        Supported datasets
        ------------------
        - "adult"
        - "make_classification"
        - "make_regression"
        - "diabetes"
        - "digits"
        - "iris"

        Supported file types
        --------------------
        - ".csv" (must contain a 'target' column)

        For built-in datasets, calls the corresponding loader method.
        For CSV files, reads the file and splits features and target.
        Raises NotImplementedError for unsupported datasets or file types.
        Updates ``self._X``, ``self._y``, and ``self.data_load_time`` with loaded data and timing information.

        Raises
        ------
        NotImplementedError
            If the dataset or file type is not supported.
        ValueError
            If a CSV file does not contain a 'target' column.
        """
        if hasattr(self, "data_load_time") and self.data_load_time is not None:
            return
        self._run_plugin_hook("before_load_data")
        supported_datasets = {
            "adult": self._load_adult_income_data,
            "make_classification": self._make_classification_data,
            "make_regression": self._make_regression_data,
            "diabetes": lambda **params: self._load_generic_sklearn(
                load_diabetes,
                **params,
            ),
            "digits": lambda **params: self._load_generic_sklearn(
                load_digits,
                **params,
            ),
            "iris": lambda **params: self._load_generic_sklearn(load_iris, **params),
        }
        for dataset_name in _lifelines_dataset_loaders().keys():
            supported_datasets.setdefault(
                f"lifelines_{dataset_name}",
                lambda _name=dataset_name, **params: self._load_lifelines_dataset(
                    _name,
                    **params,
                ),
            )
            supported_datasets.setdefault(
                f"lifelines.{dataset_name}",
                lambda _name=dataset_name, **params: self._load_lifelines_dataset(
                    _name,
                    **params,
                ),
            )
            if dataset_name not in supported_datasets:
                supported_datasets[dataset_name] = (
                    lambda _name=dataset_name, **params: self._load_lifelines_dataset(
                        _name,
                        **params,
                    )
                )
        filetype = Path(self.dataset_name).suffix
        supported_filetypes = data_supported_filetypes
        if (
            filetype not in supported_filetypes
            and self.dataset_name not in supported_datasets
        ):
            raise NotImplementedError(
                f"Currently only {supported_filetypes} filetypes are supported for loading data. Cannot load {self.dataset_name}",
            )
        if self.dataset_name in supported_datasets:
            start_time = time.process_time()
            supported_datasets[self.dataset_name](**self.data_params)
        elif filetype == ".openml":
            start_time = time.process_time()
            dataset_base_name = Path(self.dataset_name).stem
            self._load_generic_openml(
                dataset_name=dataset_base_name,
                **self.data_params,
            )
        elif filetype in supported_filetypes:
            start_time = time.process_time()
            self._load_from_csv(**self.data_params)
            end_time = time.process_time()
            self.data_load_time = end_time - start_time
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset_name} not implemented",
            )

        assert isinstance(
            self._X,
            (pd.DataFrame, pd.Series),
        ), "_X must be a DataFrame after loading data"
        assert isinstance(self._y, pd.Series), "_y must be a Series after loading data"
        self._run_plugin_hook("after_load_data")
        logger.info(
            f"Data loaded from {self.dataset_name} in {self.data_load_time:.2f} seconds",
        )

    def _load_from_csv(self):
        data = self.load_data(self.dataset_name)
        if self.target is None:
            raise ValueError(
                "CSV file must contain a 'target' column or specify the target column name in the 'target' attribute",
            )
        y = data.pop(self.target)
        if len(self.keep) > 1:
            data = data[self.keep]
        elif len(self.keep) == 1:
            data = data[self.keep[0]]
        for del_col in self.drop:
            assert len(self.keep) == 0, "Cannot specify both keep and drop columns"
            if del_col in data.columns:
                data = data.drop(columns=del_col)
        self._X = data
        self._y = y

    def _score(self) -> dict:
        """
        Computes feature importance scores based on the type of task (classification or regression).

        Parameters
        ----------
        classifier : bool, optional
            If True, computes classification feature scores; otherwise, computes regression feature scores. Default is False.

        Returns
        -------
        dict
            A dictionary containing feature importance scores.
        """
        self._run_plugin_hook("before_score")
        if self.scorer is not None:
            result_dict = self.scorer(
                y_true=self.y_train,
                y_pred=self.y_train,
                mode=None,
                data=self,
            )
        elif self.classifier:
            if isinstance(self.X_train, (pd.DataFrame, pd.Series)):
                result_dict = self._classification_feature_scores()
            else:
                result_dict = {"class_counts": self._compute_class_counts(self.y_train)}
        else:
            if isinstance(self.X_train, (pd.DataFrame, pd.Series)):
                result_dict = self._regression_feature_scores()
            else:
                y_train_cdf = self._empirical_cdf(self.y_train).tolist()
                y_test_cdf = self._empirical_cdf(self.y_test).tolist()
                result_dict = {
                    "y_train_cdf": y_train_cdf,
                    "y_test_cdf": y_test_cdf,
                }
        if isinstance(self.X_train, (pd.DataFrame, pd.Series)):
            columns = self.X_train.columns
            result_dict["column_names"] = columns.to_list()

        plugin_scores = self._run_plugin_hook("after_score", scores=result_dict)
        for plugin_score in plugin_scores:
            if isinstance(plugin_score, dict):
                result_dict.update(plugin_score)
        return result_dict

    def _compute_class_counts(self, y: pd.Series) -> dict:
        if isinstance(y, pd.Series):
            class_dict = y.value_counts()

        else:
            class_dict = pd.Series(y).value_counts()
        class_counts = class_dict.to_dict()
        return class_counts

    def _classification_feature_scores(self) -> dict:
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
        scores = {}
        if self.y_train.nunique() > 1:
            try:
                scores["mutual_info_classif"] = mutual_info_classif(
                    self.X_train,
                    self.y_train,
                    random_state=self.random_state,
                ).tolist()
            except ValueError as e:
                logger.warning(
                    f"Mutual information could not be computed: {e}. Skipping mutual_info_classif scoring.",
                )
            try:
                scores["f_classif"] = f_classif(self.X_train, self.y_train)[0].tolist()
            except ValueError as e:
                logger.warning(
                    f"ANOVA F-value could not be computed: {e}. Skipping f_classif scoring.",
                )
        else:
            logger.warning(
                "Only one class present in y_train; skipping classification feature scoring.",
            )
        # Class counts
        scores["class_counts"] = self._compute_class_counts(self.y_train)
        for score, value in scores.items():
            logger.info(f"Classification feature score - {score}: {value}")
        return scores

    def _empirical_cdf(self, data: pd.Series) -> pd.Series:
        """
        Computes the empirical cumulative distribution function (CDF) for a given pandas Series.

        Parameters
        ----------
        data : pd.Series
            The input data for which to compute the empirical CDF.

        Returns
        -------
        pd.Series
            A pandas Series representing the empirical CDF values corresponding to the input data.
        """
        sorted_data = data.sort_values().reset_index(drop=True)
        cdf_values = (sorted_data.rank(method="first") / len(sorted_data)).values
        cdf_series = pd.Series(cdf_values, index=sorted_data.index)
        return cdf_series

    def _regression_feature_scores(self) -> dict:
        """
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
        scores = {}
        scores["mutual_info_regression"] = mutual_info_regression(
            self.X_train,
            self.y_train,
            random_state=self.random_state,
        ).tolist()
        scores["f_regression"] = f_regression(self.X_train, self.y_train)[0].tolist()
        scores["r_regression"] = r_regression(self.X_train, self.y_train).tolist()
        scores["y_train_cdf"] = self._empirical_cdf(self.y_train).tolist()
        scores["y_test_cdf"] = self._empirical_cdf(self.y_test).tolist()
        return scores

    def _prepare_data_file(self, data_file: Union[str, None]) -> bool:
        """
        Handles loading/saving behavior for data_file.

        Returns
        -------
        Tuple[DataConfig, bool]
            (possibly loaded config instance, save_flag)
        """
        if data_file is not None:
            data_path = Path(data_file)
            if data_path.exists():
                logger.info(f"Loading existing DataConfig from {data_file}")
                return self.load(data_file), False
            logger.debug(f"Creating directory for DataConfig at {data_file}")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            return True

        logger.debug("No data_file provided, data will not be saved")
        return False

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

        Raises
        ------
        AssertionError
            If train or test indices are not set after sampling.
        """
        save_flag = self._prepare_data_file(data_file=data_file)
        scores = self.read_or_initialize_scores(score_file)
        # Load data if not already loaded
        if not hasattr(self, "data_load_time") or self.data_load_time is None:
            self._load_data()
            logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
        time_dict = {"data_load_time": self.data_load_time}
        # Sample data if not already sampled
        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
            # Sample data
            self._sample()
        time_dict["data_sample_time"] = (self.data_sample_time,)
        logger.info(
            f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
        )
        data_scores = self._score()
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        assert hasattr(self, "score_dict"), "score_dict must be set"

        if score_file is not None and not Path(score_file).exists():
            self.save_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict


@dataclass
class DataPipelineConfig(DataConfig):
    """Initializes a data pipeline configuration and fits it to the data in the call() method."""

    pipeline: dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate_init()
        assert isinstance(
            self.pipeline,
            (dict, DictConfig),
        ), f"pipeline must be a dictionary, got {type(self.pipeline)}"
        for attr in [
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        # Validate the pipeline configuration
        for k, v in self.pipeline.items():
            assert isinstance(
                v,
                (dict, DictConfig),
            ), f"Each step in pipeline must be a dictionary, got {type(v)} for step {k}"
            assert (
                "name" in v
            ), f"Each step in pipeline must have a 'name' key, missing in step {k}"
        if self.classifier in ["classifier", True]:
            self.classifier = True
        elif self.classifier in ["regressor", False]:
            self.classifier = False
        else:
            self.classifier = None

    def _init_pipeline(self):
        if not isinstance(self.pipeline, (dict, DictConfig)):
            raise ValueError(f"Invalid pipeline configuration: {self.pipeline}")
        X_pipeline_steps = []
        y_pipeline_steps = []
        dtypes = []
        for step_name, step_config in self.pipeline.items():
            step_class = step_config.get(
                "name",
                ValueError(f"Step {step_name} missing 'name' key"),
            )
            fit_y = step_config.get("fit_y", False)
            fit_Xy = step_config.get("fit_xy", False)
            if fit_Xy is True:
                raise ValueError("fit_xy pipeline steps are no longer supported.")
            dtype = step_config.get("dtype", None)
            step_config_without_name = {**step_config}
            del step_config_without_name["name"]
            if "fit_y" in step_config_without_name:
                del step_config_without_name["fit_y"]
            if "fit_xy" in step_config_without_name:
                del step_config_without_name["fit_xy"]
            if "dtype" in step_config:
                del step_config_without_name["dtype"]
            step_instance = load_class(step_class, **step_config_without_name)
            dtypes.append(dtype)
            if fit_y is not True:
                X_pipeline_steps.append((step_name, step_instance))
            else:
                y_pipeline_steps.append((step_name, step_instance))

        if dtypes is not None and any(x is not None for x in dtypes):

            string_dtypes = {"object", "string", "category"}
            num_dtypes = {"num", "numeric", "float", "int"}

            transformers = []

            for (name, transformer), dtype in zip(X_pipeline_steps, dtypes):

                if dtype in num_dtypes:
                    selector = make_column_selector(dtype_include=np.number)

                elif dtype in string_dtypes:
                    selector = make_column_selector(
                        dtype_include=object,
                    )  # or "string" depending on your data

                else:
                    continue  # skip unknown dtype

                transformers.append((name, transformer, selector))

            X_pipeline = Pipeline(
                steps=[
                    (
                        "preprocess",
                        ColumnTransformer(
                            transformers=transformers,
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                ],
            )

        else:
            X_pipeline = Pipeline(steps=X_pipeline_steps)
        if len(y_pipeline_steps) > 0:
            y_pipeline = y_pipeline_steps
        else:
            y_pipeline = None
        return X_pipeline, y_pipeline

    def _fit_transform_X(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fits the data pipeline to the data and returns the transformed data.

        Parameters
        ----------
        data : DataConfig
            The data configuration object containing the training and testing data.

        Returns
        -------
        pd.DataFrame
            The transformed training and testing data.
        """

        def _resolve_feature_names(input_cols, transformed_X):
            try:
                return list(pipeline.get_feature_names_out(input_cols))
            except AttributeError:
                n_features = transformed_X.shape[1]
                if len(input_cols) == n_features:
                    return list(input_cols)
                return [f"feature_{i}" for i in range(n_features)]

        if not hasattr(self, "pipeline_fit_time") or self.pipeline_fit_time is None:
            logger.info("Fitting data pipeline to training data")
            # Fit and transform the training data
            start = time.process_time()
            pipeline.fit(X_train, y_train)
            end = time.process_time()
            before_shape = X_train.shape
            if isinstance(X_train, pd.DataFrame):
                train_cols = X_train.columns
            elif isinstance(X_train, pd.Series):
                train_cols = [X_train.name]
            else:
                train_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train = pipeline.transform(X_train)
            # If csr_matrix, convert to dense
            if isinstance(X_train, csr_matrix):
                X_train = X_train.toarray()
            train_cols = _resolve_feature_names(train_cols, X_train)
            X_train = pd.DataFrame(X_train, columns=train_cols)
            after_shape = X_train.shape
            assert (
                before_shape[0] == after_shape[0]
            ), f"Number of samples changed during fit_transform from {before_shape[0]} to {after_shape[0]}"
            self.pipeline_fit_time = end - start
            self.pipeline_fit_n = X_train.shape[0]
        if (
            not hasattr(self, "pipeline_transform_time")
            or self.pipeline_transform_time is None
        ):
            # Record transform time
            start = time.process_time()
            # Transform the testing data
            before_shape = X_test.shape
            if isinstance(X_test, pd.DataFrame):
                test_cols = X_test.columns
            elif isinstance(X_test, pd.Series):
                test_cols = [X_test.name]
            else:
                test_cols = [f"feature_{i}" for i in range(X_test.shape[1])]
            X_test = pipeline.transform(X_test)
            if isinstance(X_test, csr_matrix):
                X_test = X_test.toarray()
            test_cols = _resolve_feature_names(test_cols, X_test)
            # Ensure transformed data is a DataFrame with correct columns
            X_test = pd.DataFrame(X_test, columns=test_cols)
            after_shape = X_test.shape
            assert (
                before_shape[0] == after_shape[0]
            ), f"Number of samples changed during transform from {before_shape[0]} to {after_shape[0]}"
            end = time.process_time()
            self.pipeline_transform_time = end - start
            self.pipeline_transform_n = X_test.shape[0]
        return X_train, X_test, y_train, y_test

    def _fit_transform_y(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fits the data pipeline to y_train and transforms y_train/y_test."""

        # Normalize y to 2D for sklearn transformers
        y_train_2d = (
            y_train.to_frame()
            if isinstance(y_train, pd.Series)
            else pd.DataFrame(y_train)
        )
        y_test_2d = (
            y_test.to_frame() if isinstance(y_test, pd.Series) else pd.DataFrame(y_test)
        )

        if not hasattr(self, "pipeline_y_fit_time") or self.pipeline_y_fit_time is None:
            logger.info("Fitting data pipeline to training target")
            start = time.process_time()
            for name, stage in pipeline:
                logger.debug(f"Running data pipeline stage: {name}")
                stage.fit(y_train_2d)
                y_train_t = stage.transform(y_train_2d)
            end = time.process_time()

            before_shape = y_train_2d.shape
            y_train_t = stage.transform(y_train_2d)
            if isinstance(y_train_t, csr_matrix):
                y_train_t = y_train_t.toarray()
                y_train_t = pd.DataFrame(y_train_t)
                after_shape = y_train_t.shape

                assert (
                    before_shape[0] == after_shape[0]
                ), f"Number of samples changed during y fit_transform from {before_shape[0]} to {after_shape[0]}"

            self.pipeline_y_fit_time = end - start
            self.pipeline_y_fit_n = y_train_t.shape[0]
            y_train = pd.Series(y_train_t)

        if (
            not hasattr(self, "pipeline_y_transform_time")
            or self.pipeline_y_transform_time is None
        ):
            start = time.process_time()

            before_shape = y_test_2d.shape
            y_test_t = stage.transform(y_test_2d)
            if isinstance(y_test_t, csr_matrix):
                y_test_t = y_test_t.toarray()
                y_test_t = pd.DataFrame(y_test_t)
                after_shape = y_test_t.shape

                assert (
                    before_shape[0] == after_shape[0]
                ), f"Number of samples changed during y transform from {before_shape[0]} to {after_shape[0]}"

            end = time.process_time()
            self.pipeline_y_transform_time = end - start
            self.pipeline_y_transform_n = y_test_t.shape[0]
            y_test = pd.Series(y_test_t)

        return X_train, X_test, y_train, y_test

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

        Raises
        ------
        AssertionError
            If train or test indices are not set after sampling.
        """
        save_flag = self._prepare_data_file(data_file=data_file)
        scores = self.read_or_initialize_scores(score_file)
        # Load data if not already loaded
        if not hasattr(self, "data_load_time") or self.data_load_time is None:
            self._load_data()
            logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
        time_dict = {"data_load_time": self.data_load_time}
        X_pipeline, y_pipeline = self._init_pipeline()

        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
            self._sample()
        time_dict["data_sample_time"] = (self.data_sample_time,)
        logger.info(
            f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
        )
        # Fit X pipeline
        self.X_train, self.X_test, self.y_train, self.y_test = self._fit_transform_X(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            X_pipeline,
        )
        # Fit y pipeline
        if y_pipeline is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self._fit_transform_y(
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    y_pipeline,
                )
            )
        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
            "pipeline_fit_time": self.pipeline_fit_time,
            "pipeline_fit_n": self.pipeline_fit_n,
            "pipeline_transform_time": self.pipeline_transform_time,
            "pipeline_transform_n": self.pipeline_transform_n,
        }
        self.score_dict.update(**time_dict)
        data_scores = self._score()
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        assert hasattr(self, "score_dict"), "score_dict must be set"

        if score_file is not None and not Path(score_file).exists():
            self.save_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict
