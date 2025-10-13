import pandas as pd
import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from dataclasses import dataclass
from typing import Union
import logging
import argparse

import importlib
import numpy as np
from pathlib import Path

from ..data import data_parser, DataConfig, data_main
from ..utils import initialize_config, ConfigBase, create_parser_from_function

logger = logging.getLogger(__name__)

supported_sklearn_libraries = ["sklearn"]


@dataclass
class ModelConfig(ConfigBase):
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
    training_time : float or None
        Time taken to train the model (in seconds).
    prediction_time : float or None
        Time taken to make predictions (in seconds).
    training_prediction_time : float or None
        Time taken to make predictions during training (in seconds).
    training_score_time : float or None
        Time taken to compute training scoring metrics (in seconds).
    prediction_score_time : float or None
        Time taken to compute prediction scoring metrics (in seconds).
    score_dict : dict or None
        Dictionary containing the latest computed scores and timing information.
    _training_n : int or None
        Number of training samples.
    _prediction_n : int or None
        Number of prediction samples.
    training_predictions : pd.Series, pd.DataFrame, np.ndarray, list, or None
        Predictions made on the training data.
    predictions : pd.Series, pd.DataFrame, np.ndarray, list, or None
        Predictions made on the prediction data.
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

    model_type: str = "sklearn.ensemble.RandomForestClassifier"
    classifier: bool = False
    model_params: dict = None
    probability: bool = False

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

        assert (
            library in supported_sklearn_libraries
        ), f"Only {supported_sklearn_libraries} models are supported"
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
        self.score_dict = {}
        self.training_time = None
        self.prediction_time = None
        self.training_prediction_time = None
        self.training_score_time = None
        self.prediction_score_time = None
        self.training_n = None
        self.prediction_n = None
        self.training_predictions = None
        self.predictions = None
        if self._target_ is None:
            self._target_ = "deckard.ModelConfig"

    def __hash__(self):
        return super().__hash__()

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
        assert hasattr(self._model, "fit"), "Model does not have a fit method"
        self._model.fit(X, y)
        end_time = time.process_time()
        self.training_time = end_time - start_time
        self._training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions for the input data using the initialized model.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If the model has not been initialized.

        """
        if self._model is None:
            raise ValueError("Model not initialized")
        logger.debug(f"Type of X: {type(X)}, shape of X: {X.shape}")

        y_pred = self._model.predict(X)

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

        """
        if self._model is None:
            raise ValueError("Model not initialized")
        if not self.probability:
            raise ValueError("Model does not support probability predictions")
        y_proba = self._model.predict_proba(X)

        return y_proba

    def _classification_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
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
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }
        return scores

    def _regression_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
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
        rmse = mse**0.5
        mae = (y_true - y_pred).abs().mean()
        scores = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }
        return scores

    def _score(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Compute and log performance scores for classification or regression.

        Args:
            y_true (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values.

        Returns:
            dict: Dictionary of rounded performance scores.

        Side Effects:
            - Uses classification or regression scoring based on `self.classifier`.
            - Measures and logs scoring time.
            - Rounds scores based on the size of `y_true`.
            - Logs each rounded score.
        """
        if self.classifier:
            scores = self._classification_scores(y_true, y_pred)

        else:
            scores = self._regression_scores(y_true, y_pred)
        sig_figs = np.log10(len(y_true)) + 1
        if sig_figs < 1:
            sig_figs = 1
        logger.info(f"Rounding scores to {int(sig_figs)} significant figures")
        logger.info("Scores:")
        for score in scores:
            rounded = round(scores[score], int(sig_figs))
            logger.info(f"{score}: {rounded}")
            scores[score] = rounded
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
                raise ValueError(
                    "Model is not fitted yet. Train the model before saving.",
                )
        else:
            raise NotImplementedError(
                "Model saving is only implemented for sklearn models.",
            )
        if self.training_time is None:
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
            logger.info(f"Loading model from {filepath}")
            with open(filepath, "rb") as f:
                loaded_model = pickle.load(f)
            if not hasattr(loaded_model, "predict"):
                raise ValueError("Loaded object is not a valid model instance")
            self._model = loaded_model
            self.model_params = self._model.get_params()
            self.model_type = (
                f"{self._model.__class__.__module__}.{self._model.__class__.__name__}"
            )
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"File {filepath} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_predictions(self, filepath: str):
        """
        Loads predictions from a specified CSV file.

        Args:
            filepath (str): The path to the CSV file containing predictions.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded predictions are not in a valid format.
            Exception: For any other issues during the loading process.
        Side Effects:
            - Reads predictions from the specified CSV file and assigns them to self.predictions.
            - Logs the load operation.
        """
        try:
            predictions = self.load_data(filepath)
            if not isinstance(predictions, (pd.Series, pd.DataFrame, np.ndarray, list)):
                raise ValueError("Loaded predictions are not in a valid format")
            logger.info(f"Predictions loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"File {filepath} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            raise e
        return predictions

    def _load_all_predictions(
        self,
        training_predictions_filepath,
        predictions_filepath,
        times,
    ):
        """
        Loads training and prediction data from the specified file paths and updates the provided times dictionary
        with relevant metadata.
        Parameters
        ----------
        training_predictions_filepath : str or Path or None
            File path to the training predictions. If None or the file does not exist, training predictions are not loaded.
        predictions_filepath : str or Path or None
            File path to the predictions. If None or the file does not exist, predictions are not loaded.
        times : dict
            Dictionary to be updated with timing and count information for training and prediction data.
        Updates
        -------
        self.training_predictions : object
            Loaded training predictions, if available.
        self.training_prediction_time : object
            Time associated with training predictions, must be set if training predictions are loaded.
        self.predictions : object
            Loaded predictions, if available.
        self.prediction_time : object
            Time associated with predictions, must be set if predictions are loaded.
        times["training_prediction_time"] : object
            Updated with training prediction time.
        times["training_n"] : int
            Updated with the number of training predictions.
        times["prediction_time"] : object
            Updated with prediction time.
        times["prediction_n"] : int
            Updated with the number of predictions.
        Returns
        -------
        dict
            The updated times dictionary.
        Raises
        ------
        AssertionError
            If training or prediction time is not set when corresponding predictions are loaded.
        """
        # Load the training predictions if provided
        if (
            training_predictions_filepath is not None
            and Path(training_predictions_filepath).exists()
        ):
            self.training_predictions = self._load_predictions(
                training_predictions_filepath,
            )
            assert (
                self.training_prediction_time is not None
            ), "Training prediction time must be set if training predictions are loaded"
            times["training_prediction_time"] = self.training_prediction_time
            times["training_n"] = len(self.training_predictions)

        # Load the predictions if provided
        if predictions_filepath is not None and Path(predictions_filepath).exists():
            self.predictions = self._load_predictions(predictions_filepath)
            assert (
                self.prediction_time is not None
            ), "Prediction time must be set if predictions are loaded"
            times["prediction_time"] = self.prediction_time
            times["prediction_n"] = len(self.predictions)
        return times

    def _load_score_file(self, model_score_filepath):
        """
        Loads score data from the specified file, merges it with existing scores, and extracts timing and count metrics.

        Parameters
        ----------
        model_score_filepath : str or Path
            Path to the score file to load.

        Returns
        -------
        dict
            A dictionary containing timing and count metrics (keys ending with '_time' or '_n') extracted from the score data.

        Side Effects
        -----------
        Updates instance attributes with timing and count metrics, prefixed with an underscore.
        Merges new score data with existing score data in `self.score_dict`.
        """
        times = {}
        if model_score_filepath is not None and Path(model_score_filepath).exists():
            new_score_dict = self.load_scores(model_score_filepath)
            old_score_dict = self.score_dict if self.score_dict is not None else {}
            # Update old_score_dict with new_score_dict
            score_dict = {**old_score_dict, **new_score_dict}
            # pop keys ending with _time and add to times dict
            for key in list(score_dict.keys()):
                if key.endswith("_time") or key.endswith("_n"):
                    times[key] = score_dict.pop(key)
        # Update all attributes in times dict
        for key in times:
            setattr(self, f"{key}", times[key])
        return times

    def __call__(
        self,
        data: DataConfig,
        model_filepath: Union[str, None] = None,
        predictions_filepath: Union[str, None] = None,
        training_predictions_filepath: Union[str, None] = None,
        model_score_filepath: Union[str, None] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Executes the model workflow: training, prediction, scoring, and model persistence.

        Parameters
        ----------
        data : DataConfig
            An instance of DataConfig containing training and test data.
        model_filepath : str or None, optional
            Path to save or load the model. If provided, the model will be loaded from or saved to this path.
        predictions_filepath : str or None, optional
            Path to save the predictions. If provided, the predictions will be saved to this path.
        model_score_filepath : str or None, optional
            Path to load existing scores. If provided, scores will be loaded from this path.

        Returns
        -------
        dict
            Dictionary containing scores and timing information for training, prediction, and scoring.
        Raises
        ------
        ValueError
            If prediction is requested without a trained or loaded model.

        """
        # Ensure data is loaded
        if data.X_train is None or data.y_train is None:
            raise ValueError(
                "Data not loaded. Please load data before calling the model.",
            )

        # Load the model_score_filepath if provided
        times = self._load_score_file(model_score_filepath)

        # Load predictions from filepaths and update times
        times = self._load_all_predictions(
            training_predictions_filepath,
            predictions_filepath,
            times,
        )

        # Train the model if training data is provided and model is not already trained
        times = self._load_or_train_model(data, model_filepath, times)
        self._evaluate_and_score(data, times)
        self.save(
            training_predictions_filepath=training_predictions_filepath,
            predictions_filepath=predictions_filepath,
            model_filepath=model_filepath,
            model_score_filepath=model_score_filepath,
        )
        return self.score_dict

    def _evaluate_and_score(self, data: DataConfig, times: dict = None):
        """
        Evaluates the model by making predictions and scoring them on both training and test data.

        This method performs the following steps:
        1. Makes predictions on the training data if not already available, and records the prediction time.
        2. Scores the training predictions if true labels are available and scores have not already been computed.
        3. Makes predictions on the test data if not already available, and records the prediction time.
        4. Scores the test predictions if true labels are available and scores have not already been computed.
        5. Updates the internal score dictionary with timing and scoring information.

        Parameters
        ----------
        data : DataConfig
            The data configuration object containing training and test data (X_train, y_train, X_test, y_test).
        times : dict, optional
            A dictionary to store timing information for predictions and scoring.

        Raises
        ------
        ValueError
            If training predictions are not available when attempting to score them.

        Notes
        -----
        - Timing information for predictions and scoring is logged and stored in the `times` dictionary.
        - Score metrics are prefixed with 'train_' for training scores.
        - The method updates `self.score_dict` with all computed scores and timing information.
        """
        # Make predictions on training data if not already done
        if self.training_predictions is None or self.training_prediction_time is None:
            start_time = time.process_time()
            self.training_predictions = self._predict(data.X_train)
            end_time = time.process_time()
            self.training_prediction_time = end_time - start_time
            times["training_prediction_time"] = self.training_prediction_time
            logger.info(
                f"Training predictions made in {self.training_prediction_time:.2f} seconds",
            )
            times["training_n"] = len(self.training_predictions)
            times["training_prediction_time"] = self.training_prediction_time
        else:
            logger.info(
                "Training predictions already available, skipping prediction step."
            )

        # Score training predictions if true labels are available and scoring not already done
        if self.training_score_time is None or self.score_dict is None:
            if self.training_predictions is not None:
                start = time.process_time()
                train_scores = self._score(data.y_train, self.training_predictions)
                self.training_score_time = time.process_time() - start
                # Prefix training scores with 'train_'
                train_scores = {
                    f"train_{key}": value for key, value in train_scores.items()
                }
                if self.score_dict is None:
                    self.score_dict = {}
                self.score_dict.update(train_scores)
                times["training_score_time"] = self.training_score_time
                logger.info(
                    f"Training scores computed in {self.training_score_time:.2f} seconds",
                )
            else:
                raise ValueError("Training predictions not available for scoring.")
        else:
            pass
        # Make predictions on test data if not already done
        if self.predictions is None or self.prediction_time is None:
            if data.X_test is not None:
                start_time = time.process_time()
                self.predictions = self._predict(data.X_test)
                end_time = time.process_time()
                self.prediction_time = end_time - start_time
                times["prediction_time"] = self.prediction_time
                logger.info(f"Predictions made in {self.prediction_time:.2f} seconds")
                times["prediction_n"] = len(self.predictions)
            else:
                raise ValueError("No test data available for prediction.")
        # Score test predictions if true labels are available and scoring not already done
        if self.prediction_score_time is None or self.score_dict is None:
            if data.y_test is not None and self.predictions is not None:
                test_scores = self._score(data.y_test, self.predictions)
                if self.score_dict is None:
                    self.score_dict = {}
                self.score_dict.update(test_scores)
                self.prediction_score_time = time.process_time() - start
                times["prediction_score_time"] = self.prediction_score_time
                logger.info(
                    f"Prediction scores computed in {self.prediction_score_time:.2f} seconds",
                )
            else:
                raise ValueError("No test labels available for scoring.")
        else:
            pass
        self.score_dict.update(times)

    def save(
        self,
        training_predictions_filepath,
        predictions_filepath,
        model_filepath,
        model_score_filepath,
    ):
        """
        Saves model-related outputs to specified filepaths.

        Parameters
        ----------
        training_predictions_filepath : str or None
            Filepath to save training predictions. If None, training predictions are not saved.
        predictions_filepath : str or None
            Filepath to save predictions. If None, predictions are not saved.
        model_filepath : str or None
            Filepath to save the trained model. If None, model is not saved.
        model_score_filepath : str or None
            Filepath to save model scores. If None or file does not exist, scores are not saved.

        Returns
        -------
        dict
            Dictionary containing model scores.
        """
        # Save training predictions if filepath provided
        if training_predictions_filepath is not None:
            self.save_data(
                filepath=training_predictions_filepath,
                data=self.training_predictions,
            )
            logger.info(
                f"Training predictions saved to {training_predictions_filepath}",
            )
        # Save predictions if filepath provided
        if predictions_filepath is not None and self.predictions is not None:
            self.save_data(filepath=predictions_filepath, data=self.predictions)
            logger.info(f"Predictions saved to {predictions_filepath}")
        # Save model if filepath provided
        if model_filepath is not None:
            self._save_model(model_filepath)
        # Save scores if filepath provided
        all_scores = self.score_dict if self.score_dict is not None else {}
        if model_score_filepath is not None and Path(model_score_filepath).exists():
            self.save_scores(all_scores, model_score_filepath)
            logger.info(f"Scores saved to {model_score_filepath}")
        return all_scores

    def _load_or_train_model(self, data, model_filepath, times):
        match self._model, model_filepath:
            case None, None:  # Neither model nor filepath provided
                raise ValueError(
                    "Model not trained or loaded. Please train or load a model before prediction.",
                )
            case _, _:  # Model and/or filepath provided
                if model_filepath is not None and Path(model_filepath).exists():
                    logger.info(f"Model file {model_filepath} exists. Loading model.")
                    self._load_model(model_filepath)
                    assert isinstance(self._model, object)
                    try:  # validate that the  loaded model is trained
                        check_is_fitted(self._model)
                        logger.info("Model is already trained.")
                    except NotFittedError:
                        # train the model if it is not fitted
                        self._train(data.X_train, data.y_train)
                        times["training_time"] = self.training_time
                        times["training_n"] = self._training_n
                else:
                    # train the model if no model exists at the filepath
                    self._train(data.X_train, data.y_train)
                    times["training_time"] = self.training_time
                    times["training_n"] = self._training_n
                    if model_filepath is not None:
                        self._save_model(model_filepath)

        # Validate model is trained
        if self._model is None:
            raise NotFittedError("Model is not initialized")
        try:
            check_is_fitted(self._model)
        except NotFittedError:
            raise ValueError(
                "Model is not trained. Please train the model before prediction.",
            )
        return times


# Argument parsing
model_init_parser = argparse.ArgumentParser(
    description="ModelConfig initialization parameters",
    add_help=False,
    conflict_handler="resolve",
)
model_init_parser.add_argument(
    "--model_config_file",
    type=str,
    help="Path to YAML config file",
)
model_init_parser.add_argument(
    "--model_config_params",
    type=str,
    nargs="*",
    help="Override configuration parameters as key=value pairs",
)
model_call_parser = create_parser_from_function(
    ModelConfig.__call__,
    add_help=False,
    exclude=["data"],
    parser=None,
)

model_parser = argparse.ArgumentParser(
    description="ModelConfig parameters",
    parents=[model_init_parser, model_call_parser],
    add_help=False,
    conflict_handler="resolve",
)


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
    args = model_init_parser.parse_known_args()[0]
    config_file = args.model_config_file
    params = args.model_config_params if args.model_config_params is not None else []
    target = "deckard.ModelConfig"
    model = initialize_config(config_file, params, target)
    assert isinstance(model, ModelConfig), "Config must be an instance of ModelConfig"
    return model


def model_main(args: argparse.Namespace = None) -> None:
    """
    Main entry point for initializing and running the model pipeline.
    This function sets up logging, parses command-line arguments or uses the provided
    argparse.Namespace, loads data, prepares model parameters, initializes the model,
    and executes the model with the given parameters and data.
    Args:
        args (argparse.Namespace, optional): Namespace containing parsed arguments.
            If None, arguments are parsed from the command line.
    Returns:
        tuple: A tuple containing the loaded data and the initialized model instance.
    """

    logging.basicConfig(level=logging.INFO)
    if args is None:
        parser = argparse.ArgumentParser(
            description="ModelConfig parameters",
            parents=[data_parser, model_call_parser],
        )
        args = parser.parse_known_args()[0]
    else:
        assert isinstance(
            args,
            argparse.Namespace,
        ), "args must be an argparse.Namespace"

    data_args = data_parser.parse_known_args(args=vars(args))[0]
    data = data_main(data_args)

    model = initialize_model_config()
    model_args = model_call_parser.parse_known_args()[0]
    model_params = dict(vars(model_args))
    model(data, **model_params)

    return data, model
