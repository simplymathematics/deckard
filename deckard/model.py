import pandas as pd
import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from dataclasses import dataclass
from typing import Union
import logging
import yaml
from hydra.utils import instantiate
from hydra import initialize, compose
import argparse

import importlib
import numpy as np
from pathlib import Path
from hashlib import md5 

from .data import initialize_data_config, data_parser
from .utils import initialize_config

logger = logging.getLogger(__name__)

supported_sklearn_libraries = ["sklearn"]

@dataclass
class ModelConfig:
    """
    A configuration and utility class for managing scikit-learn model instantiation, training, prediction, scoring, and persistence.

    Attributes:
    -------
    
    model_type : str
        The fully qualified class name of the scikit-learn model to instantiate (e.g., "sklearn.svm.SVC").
    classifier : bool
        Indicates whether the model is a classifier (True) or a regressor (False).
    model_params : dict or None
        Dictionary of parameters to initialize the model with. If None, default parameters are used.
    _model : object or None
        The instantiated scikit-learn model object.
    probability : bool
        If True, enables probability prediction (requires model support).
    _training_time : float or None
        Time taken to train the model (in seconds).
    _prediction_time : float or None
        Time taken to make predictions (in seconds).
    _score_time : float or None
        Time taken to compute scoring metrics (in seconds).
    _score_dict : dict or None
        Dictionary containing the latest computed scores and timing information.
    _target_ : str
        Internal identifier for the class.
    
    Methods:
    -------
    __post_init__(): Initializes the model based on the provided type and parameters.
    __hash__(): Computes a hash value for the instance based on its attributes.
    _train(X, y): Trains the model using the provided feature matrix and target vector.
    _predict(X): Generates predictions for the input data.
    _predict_proba(X): Predicts class probabilities for the input data (if supported).
    _classification_scores(y_true, y_pred): Computes classification metrics.
    _regression_scores(y_true, y_pred): Computes regression metrics.
    _score(y_true, y_pred, train): Computes and logs performance scores.
    __call__(X, y, train, score, filepath): Executes the model workflow including training, prediction, scoring, and model persistence.
    _save_model(filepath): Saves the model to the specified filepath.
    _load_model(filepath): Loads the model from the specified filepath.
    
    Raises:
    -------
    AssertionError:
        If the specified model type is not supported.
    ValueError:
        If the model is not initialized, not trained, or if prediction is attempted without a trained model.
    NotImplementedError:
        If model saving/loading is attempted for unsupported model types.
    
    Examples
    -------
    data_config = DataConfig()
    data = data_config()
    model_config = ModelConfig(model_type="sklearn.ensemble.RandomForestClassifier", classifier=True, model_params={"n_estimators": 100})
    train_scores = model_config(data, train=True, score=True)
    test_scores = model_config(data, train=False, score=True)
    """
    model_type: str = "sklearn.svm.SVC"
    classifier: bool = True
    model_params: dict = None
    _model = None
    probability: bool = False
    _training_time: float = None
    _prediction_time: float = None
    _score_time: float = None
    _score_dict: dict = None
    _target_: str = "ModelConfig"

    def __post_init__(self):
        """
        Initializes the scikit-learn model specified by `self.model_type` using the provided parameters.

        This method:
            - Ensures that only scikit-learn models are supported by checking the prefix of `self.model_type`.
            - Dynamically imports the specified scikit-learn model class.
            - Instantiates the model with `self.model_params` if provided, otherwise with default parameters.
            - Updates `self.model_params` with the parameters of the instantiated model.
            - Initializes an empty dictionary for storing model scores.

        Raises:
            AssertionError: If `self.model_type` does not start with "sklearn.".
        """
        # Import the model class from sklearn
        library = self.model_type.split(".")[0]
        
        assert library in supported_sklearn_libraries, f"Only {supported_sklearn_libraries} models are supported"
         # Dynamically import the model class
        module_name, class_name = self.model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        # initialize model with parameters if provided
        if self.model_params is not None:
            self._model = model_class(**self.model_params)
        else:
            self._model = model_class()
        self.model_params = self._model.get_params()
        self._score_dict = {}
    
    def __hash__(self):
        """
        Computes a hash value for the instance by concatenating all non-private attribute names and values,
        then applying the MD5 hash function. The resulting hash is returned as an integer.

        Returns:
            int: The hash value representing the current state of the instance.

        Note:
            Only attributes whose names do not start with an underscore ('_') are included in the hash computation.
        """
        # Hash all fields that do not start with an underscore
        hash_input = "".join(f"{k}:{v},\n" for k, v in self.__dict__.items() if not k.startswith("_"))
        return int(md5(hash_input.encode()).hexdigest(), 16)


    def _train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the internal model using the provided feature matrix and target vector.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target vector for training.

        Raises:
            ValueError: If the internal model is not initialized.

        Side Effects:
            - Fits the internal model to the data.
            - Records the training time in seconds.
            - Logs the training duration.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        start_time = time.process_time()
        self._model.fit(X, y)
        end_time = time.process_time()
        self._training_time = end_time - start_time
        logger.info(f"Model trained in {self._training_time:.2f} seconds")

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions for the input data using the initialized model.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If the model has not been initialized.

        Logs:
            - The type and shape of the input data.
            - The time taken to make predictions.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        logger.debug(f"Type of X: {type(X)}, shape of X: {X.shape}")
        start_time = time.process_time()
        y_pred = self._model.predict(X)
        end_time = time.process_time()
        self._prediction_time = end_time - start_time
        logger.info(f"Prediction made in {self._prediction_time:.2f} seconds")
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities for the input data using the trained model.

        Args:
            X (pd.DataFrame): Input features for which to predict probabilities.

        Returns:
            pd.DataFrame: Predicted class probabilities for each sample in X.

        Raises:
            ValueError: If the model is not initialized or does not support probability predictions.

        Side Effects:
            Updates self._prediction_time with the time taken for prediction.
            Logs the prediction time using the logger.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        if not self.probability:
            raise ValueError("Model does not support probability predictions")
        start_time = time.process_time()
        y_proba = self._model.predict_proba(X)
        end_time = time.process_time()
        self._prediction_time = end_time - start_time
        logger.info(f"Probability prediction made in {self._prediction_time:.2f} seconds")
        return y_proba


    def _classification_scores(self, y_true:pd.Series, y_pred:pd.Series) -> dict:
        """
        Computes classification metrics including accuracy, precision, recall, and F1-score.

        Args:
            y_true (pd.Series): True labels of the classification task.
            y_pred (pd.Series): Predicted labels from the classifier.

        Returns:
            dict: A dictionary containing the following metrics:
                - "accuracy": Accuracy score.
                - "precision": Precision score.
                - "recall": Recall score.
                - "f1-score": F1 score.

        Raises:
            AssertionError: If y_true and y_pred do not have the same length.
        """
        # Ensure that y_true and y_pred have the same length
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }
        return scores

    def _regression_scores(self, y_true:pd.Series, y_pred:pd.Series) -> dict:
        """
        Calculate regression error metrics between true and predicted values.

        Args:
            y_true (pd.Series): Series of true target values.
            y_pred (pd.Series): Series of predicted target values.

        Returns:
            dict: Dictionary containing the following regression metrics:
                - 'mse': Mean Squared Error
                - 'rmse': Root Mean Squared Error
                - 'mae': Mean Absolute Error

        Raises:
            AssertionError: If y_true and y_pred do not have the same length.
        """
        # Ensure that y_true and y_pred have the same length
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        mse = ((y_true - y_pred) ** 2).mean()
        rmse = mse ** 0.5
        mae = (y_true - y_pred).abs().mean()
        scores = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }
        return scores
    
    def _score(self, y_true:pd.Series, y_pred:pd.Series, train: bool) -> dict:
        
        """
        Compute and log performance scores for classification or regression.

        Args:
            y_true (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values.
            train (bool): If True, prefix score keys with 'train_'.

        Returns:
            dict: Dictionary of rounded performance scores.

        Side Effects:
            - Uses classification or regression scoring based on `self.classifier`.
            - Measures and logs scoring time.
            - Rounds scores based on the size of `y_true`.
            - Logs each rounded score.
            - Updates `self._score_time`.
        """
        if self.classifier:
            start_time = time.process_time()
            scores = self._classification_scores(y_true, y_pred)
        
        else:
            start_time = time.process_time()
            scores =  self._regression_scores(y_true, y_pred)
        end_time = time.process_time()
        # prepend train_ to each score if train is True
        if train:
            scores = {f"train_{k}": v for k, v in scores.items()}
        self._score_time = end_time - start_time
        logger.info(f"Scoring done in {self._score_time:.2f} seconds")
        sig_figs = np.log10(len(y_true)) + 1
        if sig_figs < 1:
            sig_figs = 1
        logger.info(f"Rounding scores to {int(sig_figs)} significant figures")
        logger.info("Scores:")
        for score in scores:
            rounded = round(scores[score], int(sig_figs))
            logger.info(f"{score}: {rounded}")
            scores[score] = rounded
        self._score_time = end_time - start_time
        return scores
    
    def _save_model(self, filepath: str):
        """
        Saves the trained model to the specified filepath using pickle.

        Args:
            filepath (str): The path where the model should be saved.

        Raises:
            ValueError: If the model is not initialized.

        Side Effects:
            - Serializes the model and writes it to the specified file.
            - Logs the save operation.
        """
        if type(self._model).__module__.split(".")[0] in supported_sklearn_libraries:
            try:
                check_is_fitted(self._model)
            except NotFittedError:
                raise ValueError("Model is not fitted yet. Train the model before saving.")
        else:
            raise NotImplementedError("Model saving is only implemented for sklearn models.")
        if self._training_time is None:
            raise ValueError("Model not trained")
        if filepath is not None:
            # Ensure the directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            # Touch the file to ensure it exists
            Path(filepath).touch()
            with open(filepath, "wb") as f:
                pickle.dump(self._model, f)
            logger.info(f"Model saved to {filepath}")
            assert Path(filepath).exists(), f"Model file {filepath} was not created"    
    
    def _load_model(self, filepath: str):
        """
        Loads a trained model from the specified filepath using pickle.

        Args:
            filepath (str): The path from which the model should be loaded.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded object is not a valid model instance.
            Exception: For any other issues during the loading process.
        Side Effects:
            - Deserializes the model from the specified file and assigns it to self._model.
            - Logs the load operation.
            - Updates self.model_params with the parameters of the loaded model.
            - Updates self._model to the loaded model instance.
            - Updates self.model_type to the class name of the loaded model.
        """
        try:
            with open(filepath, "rb") as f:
                loaded_model = pickle.load(f)
            if not hasattr(loaded_model, "predict"):
                raise ValueError("Loaded object is not a valid model instance")
            self._model = loaded_model
            self.model_params = self._model.get_params()
            self.model_type = f"{self._model.__class__.__module__}.{self._model.__class__.__name__}"
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"File {filepath} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


    def __call__(self, X: pd.DataFrame, y: pd.Series, train: bool = True, score = False, filepath: Union[str, None] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Executes the model workflow: training, prediction, scoring, and model persistence.
        Parameters
        ----------
        X : pd.DataFrame
            Feature data for training or prediction.
        y : pd.Series
            Target labels for training or scoring.
        train : bool, optional (default=True)
            If True, trains the model; otherwise, performs prediction using a loaded or previously trained model.
        score : bool, optional (default=False)
            If True, computes and returns scoring metrics.
        filepath : str or None, optional
            Path to save or load the model. If provided, the model will be loaded from or saved to this path.
        Returns
        -------
        dict
            Dictionary containing scores and timing information for training, prediction, and scoring.
        Raises
        ------
        ValueError
            If prediction is requested without a trained or loaded model.
        """
        if filepath is not None:
            if self._model is None:
                start_time = time.process_time()
                self._load_model(filepath)
                end_time = time.process_time()
                logger.debug(f"Model loaded from {filepath} in {end_time - start_time:.2f} seconds")
            else:
                logger.debug(f"Model already loaded, skipping loading from {filepath}")
        if train:
            if filepath is None or not Path(filepath).exists():
                self._train(X, y)
                self._save_model(filepath)
                times = {
                    "training_time": self._training_time,
                    "training_prediction_time": self._prediction_time,
                    "training_score_time": self._score_time,
                }
            else:
                logger.warning(f"Model file {filepath} already exists. Skipping training to avoid overwriting.")
                self._load_model(filepath)
                # TODO: Save/Load training times/scores
                times = {
                    "training_time": None,
                    "training_prediction_time": None,
                    "training_score_time": None,
                }
        else:
            if filepath is not None:
                self._load_model(filepath)
            else:
                if self._model is None:
                    raise ValueError("Model not trained or loaded. Cannot predict.")
            times = {
                "training_time" : self._training_time,
                "training_prediction_time": self._prediction_time,
                "training_score_time": self._score_time,
            }
        if self.probability:
            preds = self._predict_proba(X)
        else:
            preds = self._predict(X)
        if score is True:
            scores = self._score(y, preds, train = train)
            
        else:
            scores = {}
        times["prediction_time"] = self._prediction_time
        times["score_time"] = self._score_time
        logger.info("Timing Information:")
        for time in times:
            if times[time] is not None:
                logger.info(f"{time}: {times[time]:.3f} seconds")
        score_dict = {**scores, **times}
        self._score_dict = score_dict
        return self._score_dict



    
# Argument parsing
model_parser = argparse.ArgumentParser(description="DataConfig parameters", add_help=False,)
model_parser.add_argument('--probability', action="store_true", help='Whether the model will output probabilities (True/False)')
model_parser.add_argument('--model_config_file', type=str, help='Path to YAML config file')
model_parser.add_argument('--model_filepath', type=str, help='Path to save loaded data as CSV')
model_parser.add_argument('--model_params', type=str, nargs='*', help='Override configuration parameters as key=value pairs')


def initialize_model_config() -> ModelConfig:
    """
    Initializes a ModelConfig instance using command-line arguments and configuration files.

    This function:
        - Parses command-line arguments for model configuration.
        - Loads a YAML configuration file if specified.
        - Applies any parameter overrides provided via command-line arguments.
        - Instantiates and returns a ModelConfig object based on the composed configuration.

    Returns:
        ModelConfig: An instance of ModelConfig initialized with the specified parameters.
    """
    args = model_parser.parse_known_args()[0]
    config_file = args.model_config_file
    params = args.model_params if args.model_params is not None else []
    target = "deckard.ModelConfig"
    if not config_file and len(params) == 0:
        logger.info("No config file or parameters provided, using default ModelConfig")
        return ModelConfig()
    model = initialize_config(config_file, params, target)
    assert isinstance(model, ModelConfig), "Config must be an instance of ModelConfig"
    return model

def train_and_evaluate(args, train = True, score = True, data = None) -> tuple[dict, dict, object]:
    """
    Trains and evaluates a machine learning model using provided arguments and data.

    Args:
        args: An object containing configuration parameters such as data and model filepaths.
        train (bool, optional): If True, trains the model on the training data. Defaults to True.
        score (bool, optional): If True, evaluates the model on the test data. Defaults to True.
        data (optional): An optional data configuration object. If None, a new data configuration is initialized.

    Returns:
        tuple[dict, dict, object]: 
            - train_scores (dict): Scores or metrics from training data evaluation.
            - test_scores (dict): Scores or metrics from test data evaluation.
            - model._model (object): The trained model instance.
    """
    # Initialize DataConfig and load data
    if data is None:
        data = initialize_data_config()
    if not hasattr(data, "_X_train") or not hasattr(data, "_y_train"):
        # Load and sample data
        data(filepath=args.data_filepath)
    # Initialize model configuration
    model = initialize_model_config()   
    # Train and score model on the training and test sets
    if train:
        train_scores = model(data._X_train, data._y_train, train=True, score=True, filepath=args.model_filepath)
    else:
        train_scores = {}
    if score:
        test_scores = model(data._X_test, data._y_test, train=False, score=True, filepath=args.model_filepath)
    else:
        test_scores = {}
    return train_scores, test_scores, model._model

def model_main():
    """
    Initializes logging, parses command-line arguments for model training and evaluation, and executes the training and evaluation process.

    This function sets up logging at the INFO level, constructs an argument parser with model and data-specific options, parses the provided arguments, and calls the train_and_evaluate function with those arguments.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Model Training and Evaluation", parents=[model_parser, data_parser], conflict_handler="resolve")
    args = parser.parse_args()
    train_and_evaluate(args)
    
if __name__ == "__main__":
    model_main()

    