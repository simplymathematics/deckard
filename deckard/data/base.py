# Imports
import importlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
from scipy.sparse import csr_matrix
from sklearn.compose import make_column_selector, make_column_transformer

# Scikit-learn
from sklearn.datasets import (
    fetch_openml,
    load_diabetes,
    load_digits,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.pipeline import Pipeline

from ..frameworks.types import ArrayLike, MatrixLike

# deckard
from ..utils import (
    ConfigBase,
    coerce_to_list,
    data_supported_filetypes,
    load_class,
    merge_list_of_dicts,
)
from ._mixins import (
    DataLoaderMixin,
    DataPipelineMixin,
    DataPluginRuntimeMixin,
    DataSamplerMixin,
    DataScoreMixin,
)

# Setup logger
logger = logging.getLogger(__name__)


AUTO_SCORER = "auto"
DECKARD_TEST_MAX_SAMPLES_ENV = "DECKARD_TEST_MAX_SAMPLES"


def _coerce_scorer_config(*args, **kwargs):
    """Lazy import scorer coercion to avoid data<->score import cycles at module import time."""
    from ..score.base import coerce_scorer_config as _coerce

    return _coerce(*args, **kwargs)


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


def _discover_yellowbrick_dataset_loaders() -> dict:
    """Discover yellowbrick dataset loader functions when yellowbrick is installed."""
    try:
        yellowbrick_datasets = importlib.import_module("yellowbrick.datasets")
    except ImportError:
        return {}
    dataset_map = {}
    for attr in dir(yellowbrick_datasets):
        if not attr.startswith("load_"):
            continue
        loader = getattr(yellowbrick_datasets, attr)
        if callable(loader):
            dataset_name = attr.replace("load_", "", 1)
            dataset_map[dataset_name] = loader
    return dataset_map


def _yellowbrick_dataset_loaders() -> dict:
    return _discover_yellowbrick_dataset_loaders()


SUPPORTED_DATA_SCORING_MODES = ["train", "test", "val", "pre-sample"]


def _supported_data_score_modes() -> set[str]:
    # Import lazily to avoid deckard.data <-> deckard.score import cycles at module import time.
    from ..score.base import SUPPORTED_DATA_SCORE_MODES

    return set(SUPPORTED_DATA_SCORE_MODES)


@dataclass(eq=False, kw_only=True)
class DataConfig(
    DataPluginRuntimeMixin,
    DataLoaderMixin,
    DataSamplerMixin,
    DataScoreMixin,
    DataPipelineMixin,
    ConfigBase,
):
    """
    Configuration and utility class for loading, preprocessing, and splitting datasets for machine learning tasks.

    Args:
        dataset_name: Name of the dataset to load or path to a data file.
        data_params: Additional parameters for data loading or generation.
        test_size: Proportion of the dataset to include in the test split (between 0 and 1).
        train_size: Proportion or count of samples to include in the training split.
        val_size: Proportion or count of samples to include in the validation split when a sampler is provided.
        split: Which split index to use as the validation set when sampler performs cross-validation or shuffle splitting. Defaults to 0.
        sample: Optional pluggable sampler. When None (default) the legacy 2-way train_test_split is used. Can be an instantiated sampler object, a subclass of {class}`deckard.data.sample.BaseSampler`, or a Hydra-style dict with a `name`/`_target_` key pointing to the sampler class.
        random_state: Seed for random number generation to ensure reproducibility.
        stratify: Specifies stratification for sampling; can be None, True (use target), or a column name.
        classifier: Whether the task is classification (True) or regression (False).
        drop: List of columns to drop from the dataset.
        target: Name of the target column in the dataset (if applicable).
        keep: List of columns to keep in the dataset.
        plugins: Optional data plugin specifications executed during load/sample/score hooks.
        alias: Optional alias for the dataset configuration.
        scorer: Scorer specification or AUTO_SCORER.
        score_mode: Which split to score ("train", "test", "val", "pre-sample").

    Attributes:
        _X: Loaded feature matrix.
        _y: Loaded target vector.
        data_load_time: Time taken to load the data.
        data_sample_time: Time taken to sample/split the data.
        train_indices: Indices for training samples.
        test_indices: Indices for testing samples.
        val_indices: Indices for validation samples (set only when a sampler is used).
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Testing feature matrix.
        y_test: Testing target vector.
        X_val: Validation feature matrix (set only when a sampler is used).
        y_val: Validation target vector (set only when a sampler is used).
        train_n: Number of training samples.
        test_n: Number of testing samples.
        val_n: Number of validation samples (set only when a sampler is used).
        score_dict: Dictionary to store scores or metrics.
        _target_: Internal identifier for the class.

    Returns:
        DataConfig: Instantiated and prepared data configuration object.

    Raises:
        ValueError: For invalid parameter values or missing data.
        NotImplementedError: For unsupported datasets or file types.

    Note:
        Hooks are orchestrated by `_run_plugin_hook(hook_name, **kwargs)`. Core hook names used by DataConfig runtime are: `before_load_data`, `after_load_data`, `before_sample`, `after_sample`, `before_score`, and `after_score`. Hook kwargs are phase-specific runtime objects supplied by the caller; HookPlugin forwards them to `method_name` after merging `method_kwargs`.

    Example:
        >>> from deckard.data.sample import SplitSampler
        >>> config = DataConfig(dataset_name="digits", test_size=0.2, val_size=0.1, sample=SplitSampler())
        >>> config()
        >>> X_val, y_val = config.X_val, config.y_val
    """

    # Configuration fields
    dataset_name: str = "adult"
    data_params: dict = None
    test_size: Union[float, int, None] = None
    train_size: Union[float, int, None] = None
    val_size: Union[float, int, None] = None
    split: Union[int, None] = None
    sample: str = "split"
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    classifier: Union[bool, str] = True
    target: Union[str, None] = None
    drop: list = None
    keep: list = None
    plugins: list = field(default_factory=list)
    alias: Union[str, None] = None
    scorer: str | dict | None = AUTO_SCORER
    score_mode: str = "pre-sample"

    # Runtime state fields
    score_dict: dict = field(init=False, repr=True)
    data_load_time: Union[float, None] = None
    data_sample_time: Union[float, None] = None
    _X: Union[pd.DataFrame, pd.Series, None] = None
    _y: Union[pd.Series, None] = None
    train_indices: Union[list, None] = None
    test_indices: Union[list, None] = None
    val_indices: Union[list, None] = None
    X_train: Union[pd.DataFrame, pd.Series, None] = None
    y_train: Union[pd.Series, None] = None
    X_test: Union[pd.DataFrame, pd.Series, None] = None
    y_test: Union[pd.Series, None] = None
    X_val: Union[pd.DataFrame, pd.Series, None] = None
    y_val: Union[pd.Series, None] = None
    train_n: Union[int, None] = None
    test_n: Union[int, None] = None
    val_n: Union[int, None] = None
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

        if self.score_mode is None:
            self.score_mode = "pre-sample"
        self.score_mode = str(self.score_mode).strip().lower()
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
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "split",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}

        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.data.DataConfig"
        if not self.data_params:
            self.data_params = {}

    def _resolve_max_samples(self, dataset_len: int) -> Union[int, None]:
        """Resolve an optional dataset cap from the test-only environment variable."""
        max_samples = os.environ.get(DECKARD_TEST_MAX_SAMPLES_ENV)
        if max_samples in [None, ""]:
            return None
        try:
            max_samples = int(max_samples)
        except (TypeError, ValueError):
            raise ValueError(
                f"{DECKARD_TEST_MAX_SAMPLES_ENV} must be an integer, got {max_samples}",
            )
        if max_samples <= 0:
            return None
        return min(max_samples, dataset_len)

    def _apply_max_samples(self):
        """Truncate loaded tabular data to the configured test-only max sample cap."""
        if self._X is None or self._y is None:
            return

        sample_cap = self._resolve_max_samples(len(self._y))
        if sample_cap is None or sample_cap >= len(self._y):
            return

        if isinstance(self._X, pd.DataFrame):
            self._X = self._X.iloc[:sample_cap].copy()
        elif isinstance(self._X, pd.Series):
            self._X = self._X.iloc[:sample_cap].copy()
        else:
            raise TypeError(
                f"Unsupported _X type for {DECKARD_TEST_MAX_SAMPLES_ENV}: {type(self._X)}",
            )

        self._y = self._y.iloc[:sample_cap].copy()

    def __post_init__(self):
        self._validate_init()
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                (
                    "deckard.score.data.DefaultDataClassificationConfig"
                    if self.classifier
                    else "deckard.score.data.DefaultDataRegressionConfig"
                ),
            ),
        )

    @property
    def X(self) -> MatrixLike | None:
        """Convenience alias for the loaded feature matrix."""
        return self._X

    @property
    def y(self) -> ArrayLike | None:
        """Convenience alias for the loaded target vector."""
        return self._y

    @X.setter
    def X(self, value: MatrixLike | None) -> None:
        """Set the loaded feature matrix."""
        self._X = value

    @y.setter
    def y(self, value: ArrayLike | None) -> None:
        """Set the loaded target vector."""
        self._y = value

    @property
    def split_indices(self) -> tuple[list | None, list | None, list | None]:
        """Return the active train/test/val split index triplet."""
        return (self.train_indices, self.test_indices, self.val_indices)

    @split_indices.setter
    def split_indices(
        self,
        value: tuple[list | None, list | None, list | None],
    ) -> None:
        """Set the active train/test/val split index triplet."""
        train_idx, test_idx, val_idx = value
        self.train_indices = train_idx
        self.test_indices = test_idx
        self.val_indices = val_idx

    @property
    def train_split(self) -> list | None:
        """Convenience alias for train split indices."""
        return self.train_indices

    @train_split.setter
    def train_split(self, value: list | None) -> None:
        """Set train split indices."""
        self.train_indices = value

    @property
    def test_split(self) -> list | None:
        """Convenience alias for test split indices."""
        return self.test_indices

    @test_split.setter
    def test_split(self, value: list | None) -> None:
        """Set test split indices."""
        self.test_indices = value

    @property
    def val_split(self) -> list | None:
        """Convenience alias for validation split indices."""
        return self.val_indices

    @val_split.setter
    def val_split(self, value: list | None) -> None:
        """Set validation split indices."""
        self.val_indices = value

    @property
    def sensitive_train(self) -> Any:
        """Public accessor for training split sensitive features."""
        return getattr(self, "_sensitive_train", None)

    @sensitive_train.setter
    def sensitive_train(self, value: Any) -> None:
        """Set training split sensitive features."""
        self._sensitive_train = value

    @property
    def sensitive_test(self) -> Any:
        """Public accessor for test split sensitive features."""
        return getattr(self, "_sensitive_test", None)

    @sensitive_test.setter
    def sensitive_test(self, value: Any) -> None:
        """Set test split sensitive features."""
        self._sensitive_test = value

    @property
    def sensitive_val(self) -> Any:
        """Public accessor for validation split sensitive features."""
        return getattr(self, "_sensitive_val", None)

    @sensitive_val.setter
    def sensitive_val(self, value: Any) -> None:
        """Set validation split sensitive features."""
        self._sensitive_val = value

    @property
    def sensitive_all(self) -> Any:
        """Public accessor for full-dataset sensitive features."""
        return getattr(self, "_sensitive_all", None)

    @sensitive_all.setter
    def sensitive_all(self, value: Any) -> None:
        """Set full-dataset sensitive features."""
        self._sensitive_all = value

    def _get_stratify_col(self):
        """Return the stratification array (or ``None``) based on ``self.stratify``.

        Returns
        -------
        pd.Series or None
            The column to stratify on, or ``None`` if stratification is disabled.

        Raises
        ------
        ValueError
            If ``stratify`` is a string that is not a column name in ``self._X``,
            or if ``stratify`` is an unrecognized type.
        """
        if self.stratify is None or self.stratify is False:
            return None
        if self.stratify is True:
            if self.classifier is False:
                return None
            return self._y
        if isinstance(self.stratify, str):
            if self._X is not None and self.stratify in self._X.columns:
                return self._X[self.stratify]
            raise ValueError(
                f"Stratify column '{self.stratify}' not found in data columns",
            )
        raise ValueError("stratify must be None, True, False, or a column name")

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
        frame = (
            adult.frame.copy() if getattr(adult, "frame", None) is not None else None
        )
        if frame is None:
            frame = pd.DataFrame(adult.data).copy()
            target_source = pd.Series(adult.target, name="class")
        else:
            target_source = (
                frame.pop("class")
                if "class" in frame.columns
                else pd.Series(adult.target, name="class")
            )

        y_raw = pd.Series(target_source, name="target").copy()
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype(int)
        else:
            y = self._encode_binary_series(
                y_raw.astype(str),
                {"<=50K": 0, ">50K": 1},
            )

        X = frame
        if "sex" not in X.columns:
            raise ValueError("Adult dataset must include a 'sex' column")

        sex = self._encode_binary_series(
            X.pop("sex").astype(str),
            {"Male": 0, "Female": 1},
        )

        for column in [
            "age",
            "education-num",
            "hours-per-week",
            "capital-gain",
            "capital-loss",
            "fnlwgt",
        ]:
            if column in X.columns:
                X[column] = pd.to_numeric(X[column], errors="coerce")

        categorical_columns = X.select_dtypes(
            include=["object", "category"],
        ).columns.tolist()
        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True,
            dummy_na=True,
            dtype=int,
        )
        X["sex"] = sex.astype(int)

        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = X
        self._y = pd.Series(y)
        assert isinstance(
            self._X,
            pd.DataFrame,
        ), f"Expected DataFrame got {type(self._X)}"
        assert isinstance(
            self._y,
            pd.Series,
        ), f"Expected Series got {type(self._y)}"
        self._X = self._X.apply(pd.to_numeric, errors="coerce")
        return self

    def _encode_binary_series(
        self,
        series: pd.Series,
        mapping: dict[str, int],
    ) -> pd.Series:
        encoded = series.map(mapping)
        if encoded.isna().any():
            unique_values = [
                value for value in series.dropna().unique().tolist() if value != "nan"
            ]
            if len(unique_values) != 2:
                raise ValueError(
                    f"Expected a binary series, found values {sorted(unique_values)}",
                )
            fallback_mapping = {
                value: idx for idx, value in enumerate(sorted(unique_values))
            }
            encoded = series.map(fallback_mapping)
        return encoded.astype(int)

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
        self._X = pd.DataFrame(
            X,
            columns=[f"feature_{i}" for i in range(X.shape[1])],
        )
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
        self._X = pd.DataFrame(
            X,
            columns=[f"feature_{i}" for i in range(X.shape[1])],
        )
        self._y = pd.Series(y)
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        return self

    def _sample(
        self,
        run_hooks: bool = True,
    ):
        """
        Samples training, testing, and optionally validation indices from the loaded dataset.

        When ``self.sample`` is set, delegates to the sampler callable which returns
        ``(train_idx, test_idx, val_idx)`` and populates ``X_val``/``y_val`` in addition
        to the standard ``X_train``/``X_test`` splits.

        Without a sample, falls back to the original 2-way ``train_test_split`` behaviour
        (``X_val``/``y_val`` remain ``None``).

        Raises
        ------
        ValueError
            If data is not loaded, or if the specified stratify column is not found, or if ``stratify`` is invalid.

        Side Effects
        ------------
        Sets ``self.train_indices``, ``self.test_indices``, ``self.val_indices`` (may be
        ``None``), and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        if run_hooks:
            self._run_plugin_hook("before_sample")
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Cannot sample.")

        start_time = time.process_time()

        sampler_obj = self.compose_sampling_behavior()
        train_idx, test_idx, val_idx = sampler_obj(self)
        self.train_indices = train_idx
        self.test_indices = test_idx
        self.val_indices = val_idx
        if len(val_idx) > 0:
            self.X_val = self._X.iloc[self.val_indices].reset_index(drop=True)
            self.y_val = self._y.iloc[self.val_indices].reset_index(drop=True)
            self.val_n = len(self.X_val)

        end_time = time.process_time()
        self.data_sample_time = end_time - start_time
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")

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
        if run_hooks:
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

    def _load_yellowbrick_dataset(self, dataset_name: str, **loader_params):
        """Load a yellowbrick dataset into DataConfig feature/target fields."""
        yellowbrick_datasets = _yellowbrick_dataset_loaders()
        if not yellowbrick_datasets:
            raise ImportError(
                "Yellowbrick datasets require optional dependency deckard[yellowbrick]",
            )
        if dataset_name not in yellowbrick_datasets:
            raise NotImplementedError(
                f"Yellowbrick dataset {dataset_name} not found. Supported: {sorted(yellowbrick_datasets.keys())}",
            )

        start_time = time.process_time()
        loader = yellowbrick_datasets[dataset_name]
        dataset = loader(**loader_params)

        if hasattr(dataset, "to_data") and callable(getattr(dataset, "to_data")):
            dataset = dataset.to_data()

        if isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
        elif isinstance(dataset, pd.DataFrame):
            candidate_target = self.target
            if candidate_target is None:
                for candidate in ["target", "y", "label", "class"]:
                    if candidate in dataset.columns:
                        candidate_target = candidate
                        break
            if candidate_target is None or candidate_target not in dataset.columns:
                candidate_target = "target"
                dataset[candidate_target] = 0
            y = dataset.pop(candidate_target)
            X = dataset
        elif hasattr(dataset, "data") and hasattr(dataset, "target"):
            X = dataset.data
            y = dataset.target
        else:
            raise TypeError(
                f"Unsupported Yellowbrick dataset output type: {type(dataset)}",
            )

        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = pd.DataFrame(X)
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
            "iris": lambda **params: self._load_generic_sklearn(
                load_iris,
                **params,
            ),
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
        for dataset_name in _yellowbrick_dataset_loaders().keys():
            supported_datasets.setdefault(
                f"yellowbrick_{dataset_name}",
                lambda _name=dataset_name, **params: self._load_yellowbrick_dataset(
                    _name,
                    **params,
                ),
            )
            supported_datasets.setdefault(
                f"yellowbrick.{dataset_name}",
                lambda _name=dataset_name, **params: self._load_yellowbrick_dataset(
                    _name,
                    **params,
                ),
            )
            if dataset_name not in supported_datasets:
                supported_datasets[dataset_name] = (
                    lambda _name=dataset_name, **params: self._load_yellowbrick_dataset(
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
            params = self.data_params or {}
            supported_datasets[self.dataset_name](**params)
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
        assert isinstance(
            self._y,
            pd.Series,
        ), "_y must be a Series after loading data"
        self._apply_max_samples()
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

    def _score(
        self,
        *args,
        mode: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Delegates all dataset scoring to ``self.scorer``. Supports pre-sample mode (raw data, only in DataConfig),
        as well as train/test/val splits. If mode is not provided, uses self.score_mode or defaults to 'pre-sample'.
        """
        self._run_plugin_hook("before_score")
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"DataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        scorer_mode = str(
            mode or getattr(self, "score_mode", None) or "pre-sample",
        ).lower()
        allowed = _supported_data_score_modes()
        if scorer_mode not in allowed:
            raise ValueError(f"DataConfig score_mode '{scorer_mode}' not in {allowed}")
        if scorer_mode == "pre-sample":
            y_true = getattr(self, "_y", None)
            y_pred = getattr(self, "_X", None)
        elif scorer_mode == "train":
            y_true = getattr(self, "y_train", None)
            y_pred = getattr(self, "X_train", None)
        elif scorer_mode == "test":
            y_true = getattr(self, "y_test", None)
            y_pred = getattr(self, "X_test", None)
        elif scorer_mode == "val":
            y_true = getattr(self, "y_val", None)
            y_pred = getattr(self, "X_val", None)
        else:
            raise ValueError(f"Mode must be in {allowed}")
        if y_true is None or y_pred is None:
            raise ValueError(
                f"Data scoring mode '{scorer_mode}' requested but required data split is unavailable.",
            )
        result_dict = self.scorer(
            *args,
            y_true=y_true,
            y_pred=y_pred,
            mode=scorer_mode,
            data=self,
            **kwargs,
        )
        # Keep backward compatibility for DataConfig callers expecting a flat dict.
        if (
            isinstance(result_dict, dict)
            and scorer_mode in result_dict
            and isinstance(result_dict.get(scorer_mode), dict)
        ):
            result_dict = dict(result_dict[scorer_mode])
        plugin_scores = self._run_plugin_hook("after_score", scores=result_dict)
        for plugin_score in plugin_scores:
            if isinstance(plugin_score, dict):
                result_dict.update(plugin_score)
        return result_dict

    def apply_pipeline(self, pipeline) -> "DataConfig":
        """Attach a pipeline-like plugin object to this data config."""
        if pipeline is None:
            return self
        pipeline_plugins = (
            [pipeline] if not isinstance(pipeline, list) else list(pipeline)
        )
        existing_plugins = list(self.plugins or [])
        self.plugins = [*pipeline_plugins, *existing_plugins]
        return self

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

    def prepare_data_file(self, data_file: str | None) -> bool:
        """Public wrapper for data-file load/save preparation behavior."""
        return self._prepare_data_file(data_file=data_file)

    def ensure_data_loaded(self) -> None:
        """Ensure raw data is loaded for downstream orchestration."""
        if not hasattr(self, "data_load_time") or self.data_load_time is None:
            self.load_raw_data()

    def ensure_data_sampled(self) -> None:
        """Ensure train/test/(optional val) splits are materialized."""
        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
            self.split_data()

    def build_data_time_dict(self) -> dict:
        """Build timing/count metadata dictionary for data runtime outputs."""
        time_dict = {"data_load_time": self.data_load_time}
        time_dict["data_sample_time"] = self.data_sample_time
        time_dict["train_n"] = self.train_n
        time_dict["test_n"] = self.test_n
        if self.val_n is not None:
            time_dict["val_n"] = self.val_n
        return time_dict

    def __call__(
        self,
        *args,
        data_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
        **kwargs,
    ) -> dict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns timing and scoring information.
        Strictly validates that all output values are flat and serializable.
        """

        save_flag = self.prepare_data_file(data_file=data_file)
        scores = dict(getattr(self, "score_dict", {}) or {})
        self.ensure_data_loaded()
        logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
        self.ensure_data_sampled()
        time_dict = self.build_data_time_dict()
        if self.X_val is not None:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}, "
                f"Val set size: {len(self.X_val)}",
            )
        else:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        data_scores = self.score(*args, **kwargs)
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        assert hasattr(self, "score_dict"), "score_dict must be set"
        all_scores = self.merge_and_persist_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict


@dataclass
class DataPipelineStep:
    """
    Represents a step in a data pipeline with optional metadata.

    This dataclass normalizes and documents the optional parameters that
    configure how a step integrates with the pipeline runtime.

    Attributes
    ----------
    name : str
        Fully-qualified class name or file path (e.g., "sklearn.preprocessing.StandardScaler"
        or "path/to/custom.py:CustomTransformer").
    fit_y : bool, default=False
        If True, this step is applied to the target (y) rather than features (X).
        Only one of fit_y or fit_xy can be True.
    dtype : str, optional
        Column selector hint for ColumnTransformer. Supported values: "numeric",
        "num", "float", "int", "object", "string", "category". When specified,
        only columns of matching dtype are passed to this step.
    plugin_hook : str | list[str], optional
        Hook name(s) that trigger when this step runs. Supported hooks:
        - "before_sample": Run this step's pre_sample_fit before KFold sampling.
    """

    name: str
    fit_X: bool = True
    fit_y: bool = False
    fit_Xy: bool = False
    fit_pre_sample: bool = False
    fit_post_sample: bool = True
    dtype: Optional[str] = None
    plugin_hook: Union[str, list, None] = None
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, step_name: str, step_config: dict) -> "DataPipelineStep":
        """
        Extract DataPipelineStep from a pipeline step config dict.

        Parameters
        ----------
        step_name : str
            Name of the step (key in pipeline.steps dict).
        step_config : dict
            Configuration dict for the step.

        Returns
        -------
        DataPipelineStep
            Parsed step metadata.

        Raises
        ------
        ValueError
            If "name" key is missing or if both fit_y and fit_xy are True.
        """
        step_class = step_config.get("name")
        if step_class is None:
            raise ValueError(f"Step {step_name} missing required 'name' key")

        fit_y = bool(step_config.get("fit_y", False))
        fit_xy = bool(step_config.get("fit_Xy", step_config.get("fit_xy", False)))
        if fit_y and fit_xy:
            raise ValueError(
                f"Step {step_name} cannot enable both fit_y and fit_Xy/fit_xy",
            )
        fit_pre_sample = bool(
            step_config.get(
                "fit_pre-sample",
                step_config.get("fit_pre_sample", step_config.get("fit_presample", False)),
            ),
        )
        fit_post_sample = bool(
            step_config.get(
                "fit_post-sample",
                step_config.get(
                    "fit_post_sample",
                    step_config.get("fit_postsample", True),
                ),
            ),
        )
        fit_x = bool(step_config.get("fit_X", True))

        return cls(
            name=step_class,
            fit_X=fit_x,
            fit_y=fit_y,
            fit_Xy=fit_xy,
            fit_pre_sample=fit_pre_sample,
            fit_post_sample=fit_post_sample,
            dtype=step_config.get("dtype", None),
            plugin_hook=step_config.get("plugin_hook", None),
            args=list(step_config.get("args", []) or []),
            kwargs=dict(step_config.get("kwargs", {}) or {}),
        )

    def stripped_config(self, step_config: dict) -> dict:
        """
        Remove DataPipelineStep metadata from a step config dict.

        Parameters
        ----------
        step_config : dict
            Original step configuration.

        Returns
        -------
        dict
            New dict with metadata keys removed, ready for transformer instantiation.
        """
        config = dict(step_config)
        for key in {
            "name",
            "fit_X",
            "fit_y",
            "fit_Xy",
            "fit_xy",
            "fit_pre-sample",
            "fit_pre_sample",
            "fit_presample",
            "fit_post-sample",
            "fit_post_sample",
            "fit_postsample",
            "dtype",
            "plugin_hook",
            "args",
            "kwargs",
        }:
            config.pop(key, None)
        return config


@dataclass(eq=False, kw_only=True)
class DataPipelineConfig(DataConfig):
    """Initializes a data pipeline configuration and fits it to the data in the call() method."""

    pipeline: dict = field(default_factory=dict)
    pre_sample_transform: bool = False

    def __post_init__(self):
        self._validate_init()
        # Allow a list of step-dicts: merge them in order (later wins on key conflict)

        if isinstance(self.pipeline, (list, ListConfig)):
            self.pipeline = merge_list_of_dicts(coerce_to_list(self.pipeline))
        self._pipeline_step_hooks = {}
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
            # Validate step using DataPipelineStep.from_config (also validates "name" is present)
            try:
                DataPipelineStep.from_config(k, v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid pipeline step '{k}': {e}") from e
        if self.classifier in ["classifier", True]:
            self.classifier = True
        elif self.classifier in ["regressor", False]:
            self.classifier = False
        else:
            raise ValueError(
                f"classifier must be boolean or one of ['classifier', 'regressor'], got {self.classifier}",
            )
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                (
                    "deckard.score.data.DefaultDataClassificationConfig"
                    if self.classifier
                    else "deckard.score.data.DefaultDataRegressionConfig"
                ),
            ),
        )

    def _normalize_step_hooks(self, raw_hooks: Any) -> list[str]:
        if raw_hooks is None:
            return []
        if isinstance(raw_hooks, str):
            hooks = [raw_hooks]
        elif isinstance(raw_hooks, (list, tuple, set)):
            hooks = list(raw_hooks)
        else:
            raise TypeError(
                f"plugin_hook must be None, str, or list-like. Got {type(raw_hooks)}",
            )

        normalized = []
        for hook in hooks:
            text = str(hook).strip().lower()
            if text:
                normalized.append(text)
        return normalized

    def _pipeline_declares_hook(self, hook_name: str) -> bool:
        target_hook = str(hook_name).strip().lower()
        for _, step_config in self.pipeline.items():
            hooks = self._normalize_step_hooks(step_config.get("plugin_hook", None))
            if target_hook in hooks:
                return True
        return False

    def _resolve_step_config(self, step_class: str, step_config: dict) -> dict:
        """Resolve fold-specific paths in step config for precomputed matrices.

        For StringDistanceTransformer, resolves ${data.split} placeholders in
        distance_matrix_train and distance_matrix_test paths using self.split.
        """
        resolved_config = {**step_config}

        # Only process StringDistanceTransformer configs
        if "StringDistanceTransformer" not in step_class:
            return resolved_config

        # Resolve distance_matrix_full path if present
        if "distance_matrix_full" in resolved_config:
            path = resolved_config["distance_matrix_full"]
            if isinstance(path, str) and "${data.split}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_full"] = path.replace(
                        "${data.split}",
                        str(self.split),
                    )
            elif isinstance(path, str) and "${item.fold}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_full"] = path.replace(
                        "${item.fold}",
                        str(self.split),
                    )

        # Resolve distance_matrix_train path if present
        if "distance_matrix_train" in resolved_config:
            path = resolved_config["distance_matrix_train"]
            if isinstance(path, str) and "${data.split}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_train"] = path.replace(
                        "${data.split}",
                        str(self.split),
                    )
            elif isinstance(path, str) and "${item.fold}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_train"] = path.replace(
                        "${item.fold}",
                        str(self.split),
                    )

        # Resolve distance_matrix_test path if present
        if "distance_matrix_test" in resolved_config:
            path = resolved_config["distance_matrix_test"]
            if isinstance(path, str) and "${data.split}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_test"] = path.replace(
                        "${data.split}",
                        str(self.split),
                    )
            elif isinstance(path, str) and "${item.fold}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_test"] = path.replace(
                        "${item.fold}",
                        str(self.split),
                    )

        return resolved_config

    def _run_pre_sample_transformations(self, pipeline: Pipeline, force: bool = False):
        if not (self.pre_sample_transform or force):
            return
        # Compute budget: cap the data passed to pre_sample_fit to the configured split sizes
        train_size = getattr(self, "train_size", None) or 0
        test_size = getattr(self, "test_size", None) or 0
        val_size = getattr(self, "val_size", None) or 0
        # Only cap when sizes are integers (fractional sizes are not capped)
        if (
            isinstance(train_size, int)
            and isinstance(test_size, int)
            and isinstance(val_size, int)
        ):
            budget = train_size + test_size + val_size
        else:
            budget = 0
        n_samples = len(self._y) if self._y is not None else 0
        if budget > 0 and n_samples > budget:
            rng = np.random.default_rng(getattr(self, "random_state", None) or 0)
            keep_idx = np.sort(rng.choice(n_samples, size=budget, replace=False))
            if hasattr(self._X, "iloc"):
                X_hook = self._X.iloc[keep_idx].reset_index(drop=True)
            else:
                X_hook = self._X[keep_idx]
            if hasattr(self._y, "iloc"):
                y_hook = self._y.iloc[keep_idx].reset_index(drop=True)
            else:
                y_hook = self._y[keep_idx]
            logger.info(
                f"Subsetting data from {n_samples} to {budget} samples before pre_sample_fit.",
            )
        else:
            X_hook = self._X
            y_hook = self._y
        for step_name, step in pipeline.steps:
            configured_hooks = self._pipeline_step_hooks.get(step_name, [])
            if configured_hooks and "before_sample" not in configured_hooks:
                continue
            pre_sample_fit = getattr(step, "pre_sample_fit", None)
            if callable(pre_sample_fit):
                pre_sample_fit(X_hook, y=y_hook, data=self)

        # Apply declarative pre-sample fit/transform stages from pipeline step flags.
        X_hook_t, y_hook_t = self.fit_presample(X_hook, y_hook)
        self._X = X_hook_t
        self._y = y_hook_t

    def _inject_sample_indices(self, pipeline: Pipeline):
        if not hasattr(self, "train_indices") or self.train_indices is None:
            return
        for _, step in pipeline.steps:
            set_split_indices = getattr(step, "set_split_indices", None)
            if callable(set_split_indices):
                set_split_indices(
                    train_indices=self.train_indices,
                    test_indices=self.test_indices,
                    val_indices=self.val_indices,
                )

    def _init_pipeline(self):
        if not isinstance(self.pipeline, (dict, DictConfig)):
            raise ValueError(f"Invalid pipeline configuration: {self.pipeline}")
        self._pipeline_step_hooks = {}
        X_pipeline_steps = []
        y_pipeline_steps = []
        dtypes = []
        for step_name, step_config in self.pipeline.items():
            # Parse step metadata (name, fit_y, dtype, plugin_hook)
            step = DataPipelineStep.from_config(step_name, step_config)

            # Register hooks if present
            step_hooks = self._normalize_step_hooks(step.plugin_hook)
            if step_hooks:
                self._pipeline_step_hooks[step_name] = step_hooks

            # Get clean config (metadata removed, ready for instantiation)
            step_config_clean = step.stripped_config(step_config)

            # Merge explicit kwargs object after stripped config (kwargs wins).
            step_config_clean.update(step.kwargs)

            # Resolve fold-specific paths before instantiation
            step_config_clean = self._resolve_step_config(step.name, step_config_clean)

            # Instantiate the transformer
            step_instance = load_class(step.name, *step.args, **step_config_clean)

            if not step.fit_post_sample:
                continue
            if step.fit_X or step.fit_Xy:
                X_pipeline_steps.append((step_name, step_instance))
                dtypes.append(step.dtype)
            if step.fit_y:
                y_pipeline_steps.append((step_name, step_instance))

        if dtypes is not None and any(x is not None for x in dtypes):

            string_dtypes = {"object", "string", "category"}
            num_dtypes = {"num", "numeric", "float", "int"}

            transformers = []
            passthrough_steps = []

            for (name, transformer), dtype in zip(X_pipeline_steps, dtypes):
                if dtype is None:
                    passthrough_steps.append((name, transformer))
                    continue

                dtype_text = str(dtype).strip().lower()

                if dtype_text in num_dtypes:
                    selector = make_column_selector(dtype_include=np.number)

                elif dtype_text in string_dtypes:
                    selector = make_column_selector(
                        dtype_include=object,
                    )  # or "string" depending on your data

                else:
                    passthrough_steps.append((name, transformer))
                    continue

                transformers.append((transformer, selector))

            if transformers:
                pipeline_steps = [
                    (
                        "preprocess",
                        make_column_transformer(
                            *transformers,
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                ]
                pipeline_steps.extend(passthrough_steps)
                X_pipeline = Pipeline(steps=pipeline_steps)
            else:
                X_pipeline = Pipeline(steps=X_pipeline_steps)

        else:
            X_pipeline = Pipeline(steps=X_pipeline_steps)
        if len(y_pipeline_steps) > 0:
            y_pipeline = y_pipeline_steps
        else:
            y_pipeline = None
        return X_pipeline, y_pipeline

    def compose_pipeline_behavior(self):
        """Compose feature/target pipeline runtime behavior as needed for this config."""
        return self.create_pipeline()

    def should_run_pre_sample_pipeline(self) -> bool:
        """Determine whether pre-sample pipeline behavior is composed for this run."""
        return self.pre_sample_transform or self._pipeline_declares_hook(
            "before_sample",
        )

    def run_sampling_with_pipeline_hooks(self) -> None:
        """Compose sampling behavior with optional pre-sample pipeline hooks."""
        run_before_sample_pipeline = self.should_run_pre_sample_pipeline()
        if run_before_sample_pipeline:
            self._run_plugin_hook("before_sample")
            pre_sample_pipeline, _ = self.compose_pipeline_behavior()
            self._run_pre_sample_transformations(
                pre_sample_pipeline,
                force=run_before_sample_pipeline,
            )
            self._sample(run_hooks=False)
            self._run_plugin_hook("after_sample")
            return
        self._sample()

    def _transform_validation_with_pipeline(self, X_pipeline: Pipeline) -> None:
        """Apply the fitted feature pipeline to validation inputs when present."""
        if getattr(self, "X_val", None) is None or not X_pipeline.steps:
            return
        if isinstance(self.X_val, pd.DataFrame):
            val_cols = self.X_val.columns
        elif isinstance(self.X_val, pd.Series):
            val_cols = [self.X_val.name]
        else:
            val_cols = [f"feature_{i}" for i in range(self.X_val.shape[1])]
        X_val_transformed = X_pipeline.transform(self.X_val)
        from scipy.sparse import issparse

        if issparse(X_val_transformed):
            X_val_transformed = X_val_transformed.toarray()
        try:
            val_cols_out = list(X_pipeline.get_feature_names_out(val_cols))
        except AttributeError:
            n_features = X_val_transformed.shape[1]
            val_cols_out = (
                list(val_cols)
                if len(val_cols) == n_features
                else [f"feature_{i}" for i in range(n_features)]
            )
        self.X_val = pd.DataFrame(X_val_transformed, columns=val_cols_out)

    def apply_pipeline_behavior(self) -> None:
        """Compose and apply feature/target pipeline behavior to sampled splits."""
        X_pipeline, y_pipeline = self.compose_pipeline_behavior()
        self._inject_sample_indices(X_pipeline)

        start_fit = time.process_time()
        self.X_train, self.y_train = self.fit_X(self.X_train, self.y_train)
        self.X_train, self.y_train = self.fit_Xy(self.X_train, self.y_train)
        self.X_train, self.y_train = self.fit_y(self.X_train, self.y_train)
        end_fit = time.process_time()
        self.pipeline_fit_time = end_fit - start_fit
        self.pipeline_fit_n = len(self.X_train) if hasattr(self.X_train, "__len__") else None

        start_transform = time.process_time()
        self.X_test, self.y_test = self.run_pipeline((self.X_test, self.y_test))
        if getattr(self, "X_val", None) is not None:
            self.X_val, self.y_val = self.run_pipeline((self.X_val, self.y_val))
        end_transform = time.process_time()
        self.pipeline_transform_time = end_transform - start_transform
        self.pipeline_transform_n = len(self.X_test) if hasattr(self.X_test, "__len__") else None

        # Keep legacy path active for explicit y-only sklearn pipelines from create_pipeline().
        if y_pipeline is not None and not getattr(self, "_fitted_pipeline_y", None):
            self.X_train, self.X_test, self.y_train, self.y_test = self._fit_transform_y(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                y_pipeline,
            )

    def create_pipeline(self):
        """Public entry-point for pipeline initialisation. Delegates to `_init_pipeline()`."""
        return self._init_pipeline()

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

        # Nothing to do for an empty pipeline — skip fit/transform entirely.
        if not pipeline.steps:
            self.pipeline_fit_time = 0.0
            self.pipeline_fit_n = X_train.shape[0]
            self.pipeline_transform_time = 0.0
            self.pipeline_transform_n = X_test.shape[0]
            return X_train, X_test, y_train, y_test

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
            y_test.to_frame()
            if isinstance(y_test, pd.Series)
            else pd.DataFrame(y_test)
        )

        if (
            not hasattr(self, "pipeline_y_fit_time")
            or self.pipeline_y_fit_time is None
        ):
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
        *args,
        data_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
        mode: str = None,
        **kwargs,
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
        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
            self.run_sampling_with_pipeline_hooks()
        time_dict["data_sample_time"] = (self.data_sample_time,)

        self.apply_pipeline_behavior()
        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
            "pipeline_fit_time": self.pipeline_fit_time,
            "pipeline_fit_n": self.pipeline_fit_n,
            "pipeline_transform_time": self.pipeline_transform_time,
            "pipeline_transform_n": self.pipeline_transform_n,
        }
        if getattr(self, "val_n", None) is not None:
            time_dict["val_n"] = self.val_n
        if getattr(self, "X_val", None) is not None:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}, "
                f"Val set size: {len(self.X_val)}",
            )
        else:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        self.score_dict.update(**time_dict)
        # Respect explicit mode first, then configured score_mode, then _score fallback.
        resolved_mode = mode if mode is not None else getattr(self, "score_mode", None)
        data_scores = self._score(*args, mode=resolved_mode, **kwargs)
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        assert hasattr(self, "score_dict"), "score_dict must be set"

        if score_file is not None and not Path(score_file).exists():
            self.save_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict
