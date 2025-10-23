# Imports
import pandas as pd
import time
import logging
import importlib
import sys
from pathlib import Path

from dataclasses import dataclass, field
from typing import Tuple, Union
from omegaconf import DictConfig, OmegaConf

# Scikit-learn
from sklearn.datasets import (
    fetch_openml,
    make_classification,
    make_regression,
    load_digits,
    load_diabetes,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    f_classif,
    f_regression,
    r_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# deckard
from ..utils import ConfigBase, data_supported_filetypes

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DataPipelineConfig(ConfigBase):
    """Initializes a data pipeline configuration and fits it to the data in the call() method."""

    pipeline: dict = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(
            self.pipeline, dict
        ), f"pipeline must be a dictionary, got {type(self.pipeline)}"
        self.pipeline_fit_n = None
        self.pipeline_transform_n = None
        self.pipeline_fit_time = None
        self.pipeline_transform_time = None
        # Validate the pipeline configuration
        for k, v in self.pipeline.items():
            assert isinstance(
                v, dict
            ), f"Each step in pipeline must be a dictionary, got {type(v)} for step {k}"
            assert (
                "name" in v
            ), f"Each step in pipeline must have a 'name' key, missing in step {k}"

        return super().__post_init__()

    def _init_pipeline(self):
        if not isinstance(self.pipeline, (dict, DictConfig)):
            raise ValueError(f"Invalid pipeline configuration: {self.pipeline}")
        pipeline_steps = []
        for step_name, step_config in self.pipeline.items():
            step_class = step_config.get(
                "name", ValueError(f"Step {step_name} missing 'name' key")
            )
            step_config_without_name = {**step_config}
            del step_config_without_name["name"]
            module_name, class_name = step_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = module.__dict__[class_name]
            step_instance = cls(**step_config_without_name)
            pipeline_steps.append((step_name, step_instance))
        pipeline = Pipeline(pipeline_steps)
        return pipeline

    def __call__(
        self, X_train, X_test, y_train, y_test
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
        pipeline = self._init_pipeline()
        if not hasattr(self, "pipeline_fit_time") or self.pipeline_fit_time is None:
            logger.info("Fitting data pipeline to training data")
            # Fit and transform the training data
            start = time.process_time()
            pipeline.fit(X_train, y_train)
            end = time.process_time()
            before_shape = X_train.shape
            X_train = pipeline.transform(X_train)
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
            X_test = pipeline.transform(X_test.values)
            after_shape = X_test.shape
            assert (
                before_shape[0] == after_shape[0]
            ), f"Number of samples changed during transform from {before_shape[0]} to {after_shape[0]}"
            end = time.process_time()
            self.pipeline_transform_time = end - start
            self.pipeline_transform_n = X_test.shape[0]
        return X_train, X_test, y_train, y_test


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
    random_state : int
        Seed for random number generation to ensure reproducibility.
    stratify : Union[None, str, bool]
        Specifies stratification for sampling; can be None, True (use target), or a column name.
    classifier: bool
        Whether the task is classification (True) or regression (False).
    pipeline: DataPipelineConfig
        Optional data pipeline configuration for preprocessing steps.
    drop: list
        List of columns to drop from the dataset.
    target: Union[str, None]
        Name of the target column in the dataset (if applicable).
    keep: list
        List of columns to keep in the dataset.
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
    _train_indices : list
        Indices for training samples.
    _test_indices : list
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
    _load_diabetes_data()
        Loads and preprocesses the diabetes dataset from scikit-learn.
    _load_digits_data()
        Loads and preprocesses the digits dataset from scikit-learn.
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
    config = DataConfig(dataset_name="adult", **kwargs)
    config()
    X_train = config.X_train
    y_train = config.y_train
    X_test = config.X_test
    y_test = config.y_test
    score_dict = config.score_dict
    """

    dataset_name: str = "adult"
    data_params: dict = None
    test_size: Union[float, int, None] = None
    train_size: Union[float, int, None] = None
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    classifier: bool = True
    pipeline: Union[DataPipelineConfig, None] = None
    target: Union[str, None] = None
    drop: list = None
    keep: list = None

    def __post_init__(self):
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
        self.data_load_time = None
        self.data_sample_time = None
        self.data_params = self.data_params if self.data_params is not None else {}
        self.target = self.target
        self.drop = [] if not hasattr(self, "drop") or self.drop is None else self.drop
        self.keep = [] if not hasattr(self, "keep") or self.keep is None else self.keep
        self._X = None
        self._y = None
        self._train_indices = None
        self._test_indices = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_n = None
        self.test_n = None
        self.data_load_time = None
        self.data_sample_time = None
        self._train_indices = None
        self._test_indices = None
        if self.pipeline is not None:
            if isinstance(self.pipeline, dict):
                self.pipeline = DataPipelineConfig(pipeline=self.pipeline)
            elif isinstance(self.pipeline, DictConfig):
                self.pipeline = DataPipelineConfig(
                    pipeline=OmegaConf.to_container(self.pipeline)
                )
            elif isinstance(self.pipeline, DataPipelineConfig):
                pass
            else:
                raise ValueError(
                    f"pipeline must be a dict, DictConfig, or Pipeline instance, got {type(self.pipeline)}",
                )
            assert isinstance(
                self.pipeline,
                (DataPipelineConfig),
            ), f"pipeline must be a DataPipelineConfig instance, got {type(self.pipeline)}"
        assert self.classifier in [True, False], "classifier must be a boolean value"

        if self._target_ is None:
            self._target_ = "DataConfig"

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
        return self

    def _load_diabetes_data(self):
        """
        Loads the diabetes dataset using scikit-learn, separates features and target,
        and stores them as instance variables. Also records the time taken to load the data.

        Returns
        -------
        self : DataConfig
            The instance of the class with loaded data.
        """
        start_time = time.process_time()
        diabetes = load_diabetes(as_frame=True)
        X = diabetes.frame.drop(columns="target")
        y = diabetes.frame["target"]
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = X
        self._y = pd.Series(y)
        return self

    def _load_digits_data(self):
        """
        Loads the scikit-learn digits dataset into the instance variables.

        Loads the digits dataset as a pandas DataFrame, separates the features (X) and target labels (y),
        records the time taken to load the data, and stores the results in instance variables.

        Returns
        -------
        self : DataConfig
            The instance with loaded data and updated attributes.
        """
        start_time = time.process_time()
        digits = load_digits(as_frame=True)
        X = digits.frame.drop(columns="target")
        y = digits.frame["target"]
        end_time = time.process_time()
        self.data_load_time = end_time - start_time
        self._X = X
        self._y = pd.Series(y)
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
        Sets ``self._train_indices``, ``self._test_indices``, and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        if not hasattr(self, "_X") or self._X is None:
            raise ValueError("Data not loaded. Cannot sample.")
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
        self._train_indices = train_idx
        self._test_indices = test_idx
        self.X_train = self._X.iloc[self._train_indices].reset_index(drop=True)
        self.y_train = self._y.iloc[self._train_indices].reset_index(drop=True)
        self.X_test = self._X.iloc[self._test_indices].reset_index(drop=True)
        self.y_test = self._y.iloc[self._test_indices].reset_index(drop=True)
        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)
        assert isinstance(
            self.X_train, (pd.DataFrame, pd.Series)
        ), "X_train must be a DataFrame"
        assert isinstance(self.y_train, pd.Series), "y_train must be a Series"
        assert isinstance(
            self.X_test, (pd.DataFrame, pd.Series)
        ), "X_test must be a DataFrame"
        assert isinstance(self.y_test, pd.Series), "y_test must be a Series"

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
        filetype = Path(self.dataset_name).suffix
        supported_filetypes = data_supported_filetypes
        suppoted_datasets = [
            "adult",
            "make_classification",
            "make_regression",
            "diabetes",
            "digits",
        ]
        if (
            filetype not in supported_filetypes
            and self.dataset_name not in suppoted_datasets
        ):
            raise NotImplementedError(
                f"Currently only {supported_filetypes} filetypes are supported for loading data. Cannot load {self.dataset_name}",
            )
        match self.dataset_name:
            case "adult":
                self._load_adult_income_data(**self.data_params)
            case "make_classification":
                self._make_classification_data(**self.data_params)
            case "make_regression":
                self._make_regression_data(**self.data_params)
            case "diabetes":
                self._load_diabetes_data(**self.data_params)
            case "digits":
                self._load_digits_data(**self.data_params)
            case _ if filetype in supported_filetypes:
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
                    assert (
                        len(self.keep) == 0
                    ), "Cannot specify both keep and drop columns"
                    if del_col in data.columns:
                        data = data.drop(columns=del_col)
                self._X = data
                self._y = y
                end_time = time.process_time()
                self.data_load_time = end_time - time.process_time()
                logger.info(
                    f"Data loaded from {self.dataset_name} in {self.data_load_time:.2f} seconds",
                )
            case _:
                raise NotImplementedError(
                    f"Dataset {self.dataset_name} not implemented",
                )
        assert isinstance(
            self._X, (pd.DataFrame, pd.Series)
        ), "_X must be a DataFrame after loading data"
        assert isinstance(self._y, pd.Series), "_y must be a Series after loading data"

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
        if self.classifier:
            return self._classification_feature_scores()
        else:
            return self._regression_feature_scores()

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
            scores["mutual_info_classif"] = mutual_info_classif(
                self.X_train,
                self.y_train,
                random_state=self.random_state,
            ).tolist()
            try:
                scores["chi2"] = chi2(self.X_train, self.y_train)[0].tolist()
            except ValueError as e:
                logger.warning(
                    f"Chi-squared test could not be computed: {e}. Skipping chi2 scoring.",
                )
            scores["f_classif"] = f_classif(self.X_train, self.y_train)[0].tolist()
        else:
            logger.warning(
                "Only one class present in y_train; skipping classification feature scoring.",
            )
        # Class counts
        class_counts = self.y_train.value_counts().to_dict()
        scores["class_counts"] = class_counts
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

        # Load existing data if filepath is provided and file exists, else create directory
        if data_file is not None and Path(data_file).exists():
            # Load existing data
            logger.info(f"Loading existing DataConfig from {data_file}")
            self = self.load(data_file)
            save_flag = False
        elif data_file is not None and not Path(data_file).exists():
            # Ensure directory exists
            logger.debug(f"Creating directory for DataConfig at {data_file}")
            Path(data_file).parent.mkdir(parents=True, exist_ok=True)
            save_flag = True
        else:
            logger.debug("No data_file provided, data will not be saved")
            save_flag = False

        # Load scores if filepath is provided and file exists, else create directory
        if score_file is not None and Path(score_file).exists():
            # Load existing scores
            logger.info(f"Loading existing scores from {score_file}")
            scores = self.load_scores(score_file)
        elif score_file is not None:
            # Ensure directory exists
            logger.debug(f"Creating directory for scores at {score_file}")
            Path(score_file).parent.mkdir(parents=True, exist_ok=True)
            scores = {}
        else:
            logger.debug("No score_file provided, scores will not be saved")
            scores = {}
        # Load data if not already loaded
        if not hasattr(self, "_data_load_time") or self.data_load_time is None:
            start_time = time.process_time()
            self._load_data()
            end_time = time.process_time()
            self.data_load_time = end_time - start_time
            logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
        # Sample data if not already sampled
        if not hasattr(self, "_data_sample_time") or self.data_sample_time is None:
            # Sample data
            self._sample()
            assert (
                hasattr(self, "_train_indices") and self._train_indices is not None
            ), "Train indices must be set after sampling"
            assert (
                hasattr(self, "_test_indices") and self._test_indices is not None
            ), "Test indices must be set after sampling"
        if self.pipeline is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = self.pipeline(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            )
            time_dict = {
                "data_load_time": self.data_load_time,
                "data_sample_time": self.data_sample_time,
                "pipeline_fit_time": self.pipeline.pipeline_fit_time,
                "pipeline_fit_n": self.pipeline.pipeline_fit_n,
                "pipeline_transform_time": self.pipeline.pipeline_transform_time,
                "pipeline_transform_n": self.pipeline.pipeline_transform_n,
            }
        else:
            # Prepare return dictionary
            time_dict = {
                "data_load_time": self.data_load_time,
                "data_sample_time": self.data_sample_time,
            }
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        data_scores = self._score()
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        if score_file is not None:
            self.save_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict
