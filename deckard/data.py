# Imports
import pandas as pd
import time
import logging
import argparse

from pathlib import Path
from hashlib import md5
from dataclasses import dataclass
from typing import Union

# Scikit-learn
from sklearn.datasets import (
    fetch_openml,
    make_classification,
    make_regression,
    load_digits,
    load_diabetes,
)
import sklearn.model_selection


# deckard
from .utils import initialize_config

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
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
    _X : pd.DataFrame
        Loaded feature matrix.
    _y : pd.Series
        Loaded target vector.
    _data_load_time : float
        Time taken to load the data.
    _data_sample_time : float
        Time taken to sample/split the data.
    _train_indices : list
        Indices for training samples.
    _test_indices : list
        Indices for testing samples.
    _X_train : pd.DataFrame
        Training feature matrix.
    _y_train : pd.Series
        Training target vector.
    _X_test : pd.DataFrame
        Testing feature matrix.
    _y_test : pd.Series
        Testing target vector.
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

    Raises
    ------
    ValueError
        For invalid parameter values or missing data.
    NotImplementedError
        For unsupported datasets or file types.

    Examples
    --------
    config = DataConfig(dataset_name="adult", **kwargs)
    X_train, y_train, X_test, y_test = config()
    """

    dataset_name: str = "adult"
    data_params: dict = None
    test_size: float = 0.2
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    _X: pd.DataFrame = None
    _y: pd.Series = None
    _data_load_time: float = None
    _data_sample_time: float = None
    _train_indices: list = None
    _test_indices: list = None
    _X_train: pd.DataFrame = None
    _y_train: pd.Series = None
    _X_test: pd.DataFrame = None
    _y_test: pd.Series = None
    _target_: str = "DataConfig"

    def __post_init__(self):
        """
        Post-initialization method for setting up data-related attributes.

        Validates that `test_size` is between 0 and 1, then initializes training size and internal attributes
        for data loading, sampling, parameters, and train/test splits.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.
        """
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        self.train_size = 1 - self.test_size
        self._data_load_time = None
        self._data_sample_time = None
        self._data_params = self.data_params if self.data_params is not None else {}
        self._X = None
        self._y = None
        self._train_indices = None
        self._test_indices = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

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
            f"{k}:{v}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return int(md5(hash_input.encode()).hexdigest(), 16)

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
        start_time = time.time()
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
        end_time = time.time()
        self._data_load_time = end_time - start_time
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
        start_time = time.time()
        diabetes = load_diabetes(as_frame=True)
        X = diabetes.frame.drop(columns="target")
        y = diabetes.frame["target"]
        end_time = time.time()
        self._data_load_time = end_time - start_time
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
        start_time = time.time()
        digits = load_digits(as_frame=True)
        X = digits.frame.drop(columns="target")
        y = digits.frame["target"]
        end_time = time.time()
        self._data_load_time = end_time - start_time
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
        Sets self._data_load_time (float): Time taken to generate the data.
        """
        start_time = time.time()
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            random_state=random_state,
        )
        self._X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self._y = pd.Series(y)
        end_time = time.time()
        self._data_load_time = end_time - start_time
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
            The instance with generated data stored in self._X (DataFrame), self._y (Series), and self._data_load_time (float).
        """
        start_time = time.time()
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state,
        )
        self._X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self._y = pd.Series(y)
        end_time = time.time()
        self._data_load_time = end_time - start_time
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
        Sets ``self._train_indices``, ``self._test_indices``, and ``self._data_sample_time``.
        Logs the time taken for sampling.
        """
        if not hasattr(self, "_X") or self._X is None:
            raise ValueError("Data not loaded. Cannot sample.")
        train_n = int(self.train_size * len(self._X))
        test_n = len(self._X) - train_n
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
                        f"Stratify column {self.stratify} not found in data columns"
                    )
            else:
                raise ValueError("stratify must be None, True, or a column name")
        indices = range(len(self._X))
        start_time = time.time()
        train_idx, test_idx = sklearn.model_selection.train_test_split(
            indices,
            train_size=train_n,
            test_size=test_n,
            random_state=self.random_state,
            stratify=stratify_col if self.stratify is not None else None,
        )
        end_time = time.time()
        self._data_sample_time = end_time - start_time
        logger.info(f"Data sampled in {self._data_sample_time:.2f} seconds")
        self._train_indices = train_idx
        self._test_indices = test_idx

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
        Updates ``self._X``, ``self._y``, and ``self._data_load_time`` with loaded data and timing information.

        Raises
        ------
        NotImplementedError
            If the dataset or file type is not supported.
        ValueError
            If a CSV file does not contain a 'target' column.
        """
        filetype = Path(self.dataset_name).suffix
        supported_filetypes = [".csv"]
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
                f"Currently only {supported_filetypes} filetypes are supported for loading data"
            )
        match self.dataset_name:
            case "adult":
                self._load_adult_income_data(**self._data_params)
            case "make_classification":
                self._make_classification_data(**self._data_params)
            case "make_regression":
                self._make_regression_data(**self._data_params)
            case "diabetes":
                self._load_diabetes_data(**self._data_params)
            case "digits":
                self._load_digits_data(**self._data_params)
            case _ if filetype == ".csv":
                data = pd.read_csv(self.dataset_name)
                if "target" not in data.columns:
                    raise ValueError("CSV file must contain 'target' column")
                y = data.pop("target")
                X = data
                self._X = X
                self._y = y
                end_time = time.time()
                self._data_load_time = end_time - time.time()
                logger.info(
                    f"Data loaded from {self.dataset_name} in {self._data_load_time:.2f} seconds"
                )
            case _ if filetype == ".npz":
                raise NotImplementedError("Loading from .npz files not yet implemented")
            case _:
                raise NotImplementedError(
                    f"Dataset {self.dataset_name} not implemented"
                )

    def __call__(self, filepath=None) -> dict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns the corresponding features and labels.

        Parameters
        ----------
        filepath : str, optional
            Path to the data file. If None, uses the default data source.

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
        start_time = time.time()
        self._load_data()
        end_time = time.time()
        self._data_load_time = end_time - start_time
        logger.info(f"Data loaded in {self._data_load_time:.2f} seconds")
        # Sample data
        self._sample()
        assert (
            hasattr(self, "_train_indices") and self._train_indices is not None
        ), "Train indices must be set after sampling"
        assert (
            hasattr(self, "_test_indices") and self._test_indices is not None
        ), "Test indices must be set after sampling"
        train_indices = self._train_indices
        test_indices = self._test_indices
        # Create train and test sets
        self._X_train = self._X.iloc[train_indices].reset_index(drop=True)
        self._y_train = self._y.iloc[train_indices].reset_index(drop=True)
        self._X_test = self._X.iloc[test_indices].reset_index(drop=True)
        self._y_test = self._y.iloc[test_indices].reset_index(drop=True)
        time_dict = {
            "data_load_time": self._data_load_time,
            "data_sample_time": self._data_sample_time,
        }
        logger.info(f"Train set size: {len(self._X_train)}, Test set size: {len(self._X_test)}")
        ## TODO: Add Scores for dataset
        
        scores = {}
        all_scores = {**time_dict, **scores}
        return all_scores


# Argument parsing
data_parser = argparse.ArgumentParser(
    description="DataConfig parameters",
    add_help=False,
)
data_parser.add_argument(
    "--data_config_file", type=str, help="Path to YAML config file"
)
data_parser.add_argument(
    "--data_filepath", type=str, help="Path to save loaded data as CSV"
)
# data_params should be a dotlist of key=value pairs
data_parser.add_argument(
    "--data_params",
    type=str,
    nargs="*",
    help="Override configuration parameters as key=value pairs",
)


def initialize_data_config():
    """
    Initializes the data configuration using command-line arguments.

    Parses known arguments for data configuration file and parameters,
    then initializes the DataConfig instance using ``initialize_config``.

    Returns
    -------
    DataConfig
        An instance of DataConfig initialized with the specified configuration.
    """
    args = data_parser.parse_known_args()[0]
    config_file = args.data_config_file
    params = args.data_params if args.data_params is not None else []
    target = "deckard.DataConfig"
    data = initialize_config(config_file, params, target)
    assert isinstance(data, DataConfig), "Config must be an instance of DataConfig"
    return data


def data_main(args: argparse.Namespace = None):

    """
    Parameters
    ----------
    args : argparse.Namespace, optional
        Parsed command-line arguments. If None, arguments are parsed from sys.argv.
    
    Main function for data initialization and validation.

    Parses command-line arguments, sets up logging, loads data configuration,
    and validates the consistency of training and testing datasets.

    Steps
    -----
    1. Parses known arguments for data file path.
    2. Sets up logging at INFO level.
    3. Loads data configuration using ``initialize_data_config``.
    4. Loads data from the specified file path.
    5. Extracts training and testing features, labels, and optional attributes.
    6. Asserts that feature and label arrays have matching lengths for both train and test sets.
    7. Logs the sizes of the training and testing sets.

    Raises
    ------
    AssertionError
        If any of the data arrays have mismatched lengths.

    Logs
    ----
    Train and test set sizes.
    """
    if args is None:
        args = data_parser.parse_known_args()[0]
    else:
        assert isinstance(args, argparse.Namespace), "args must be an argparse.Namespace"
    # setup logging
    logging.basicConfig(level=logging.INFO)
    # Load configuration from YAML file if provided
    data = initialize_data_config()
    data(filepath=args.data_filepath)
    X_train = data._X_train
    y_train = data._y_train
    X_test = data._X_test
    y_test = data._y_test
    assert len(X_train) == len(y_train), "X_train and y_train must have the same length"
    assert len(X_test) == len(y_test), "X_test and y_test must have the same length"


if __name__ == "__main__":
    data_main()
