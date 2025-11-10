import time
import importlib
import logging
from typing import Union
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from art.estimators.classification.scikitlearn import (
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnRandomForestClassifier,
    ScikitlearnSVC,
)
from art.estimators.regression.scikitlearn import (
    ScikitlearnDecisionTreeRegressor,
    ScikitlearnRegressor,
)

from ..data import DataConfig
from ..utils import ConfigBase

logger = logging.getLogger(__name__)

__all__ = ["ModelConfig"]


classifier_dict = {
    "SVC": ScikitlearnSVC,
    "LogisticRegression": ScikitlearnLogisticRegression,
    "RandomForestClassifier": ScikitlearnRandomForestClassifier,
    "GradientBoostingClassifier": ScikitlearnGradientBoostingClassifier,
    "ExtraTreesClassifier": ScikitlearnExtraTreesClassifier,
    "AdaBoostClassifier": ScikitlearnAdaBoostClassifier,
    "BaggingClassifier": ScikitlearnBaggingClassifier,
    "DecisionTreeClassifier": ScikitlearnDecisionTreeClassifier,
    "sklearn-classifier": ScikitlearnClassifier,
}

regressor_dict = {
    "DecisionTreeRegressor": ScikitlearnDecisionTreeRegressor,
    "sklearn-regressor": ScikitlearnRegressor,
}

sklearn_dict = {**classifier_dict, **regressor_dict}
sklearn_models = list(sklearn_dict.keys())

supported_defense_types = [
    "detector",
    "preprocessor",
    "postprocessor",
    "trainer",
    "regularizer",
    "transformer",
    None,
]


@dataclass
class DefenseConfig(ConfigBase):
    """Stores the name and parameters of a defense to be applied to a model."""

    defense_type: Union[str, None] = None
    defense_params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self._target_ = "deckard.model.DefenseConfig"
        return super().__post_init__()


@dataclass
class ModelConfig(ConfigBase):
    """
    A configuration and utility class for managing scikit-learn model instantiation, training, prediction, scoring, and persistence.

    Attributes:
    -----------

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
    alias : str or None
        An optional alias for the model configuration.
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
    classifier: bool = True
    model_params: dict = None
    probability: bool = False
    alias: Union[str, None] = None
    defense: Union[DefenseConfig, None] = None

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
        # Dynamically import the model class
        self._initialize_model()
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

    def _initialize_model(self):
        module_name, class_name = self.model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        # initialize model with parameters if provided
        if self.model_params is not None:
            self._model = model_class(**self.model_params)
        else:
            self._model = model_class()
        if hasattr(self._model, "get_params"):
            self.model_params = self._model.get_params()
        else:
            assert isinstance(
                self.model_params,
                (dict, DictConfig),
            ), f"model_params must be a dict if model does not have get_params method. Got {type(self.model_params)}"

    def __hash__(self):
        return super().__hash__()

    def parse_defense_name(self):
        defense_name = self.defense.defense_type if self.defense is not None else None
        assert defense_name is not None, "defense_type must be provided in ModelConfig"
        if defense_name is not None and len(defense_name) > 0:
            module_name, class_name = defense_name.rsplit(".", 1)
        else:
            module_name = None
            class_name = None
        if module_name is None or class_name is None:
            defense_type = None
        else:
            try:
                defense_type = module_name.split(".")[2]  # e.g., 'preprocessor'
            except IndexError:
                raise ImportError(
                    f"Could not parse defense type from defense name {self.defense_name}",
                )
        if module_name is not None and len(module_name.split(".")) >= 4:
            defense_subtype = module_name.split(".")[3]  # e.g., 'FeatureSqueezing'
        else:
            defense_subtype = None
        if defense_type is not None:
            try:
                module = importlib.import_module(module_name)
                defense_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not import defense class {class_name} from module {module_name}",
                ) from e
        else:
            defense_class = None
            module = None
        assert (
            defense_type in supported_defense_types
        ), f"Unsupported defense type: {defense_type}. Supported types are: {supported_defense_types}"

        return defense_type, defense_subtype, defense_class

    def get_art_class(self, data):
        
        art_class = (
            classifier_dict[self.model_type.split(".")[-1]]
            if self.classifier
            else regressor_dict[self.model_type.split(".")[-1]]
        )
        if art_class in sklearn_dict.values():
            init_params = {}
        else:
            init_params = {
                "input_shape": data.X_train.shape[1:],
                "nb_classes": len(set(data.y_train)) if self.classifier else None,
            }
        return art_class, init_params

    def get_model(self) -> BaseEstimator:
        """Get the model's estimator.

        Returns
        -------
        BaseEstimator
            The model's estimator.
        """
        if self._model is None:
            raise ValueError("Model is not fitted yet.")
        return self._model

    def _apply_defense(self, data) -> BaseEstimator:
        """Apply the specified defense to the model's estimator.

        Returns
        -------
        BaseEstimator
            The estimator wrapped with the specified defense.
        Raises
        ------
        ValueError
            If the model is not fitted before applying the defense.
        """

        if self._model is None:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            )
        # Dynamically import the defense class with defense_params as kwargs
        defense_type, defense_subtype, defense_class = self.parse_defense_name()
        art_class, init_params = self.get_art_class(data)
        start = time.process_time()
        match defense_type:  # Note: only one defense can be applied at a time
            case "preprocessor":
                defense = defense_class(**(self.defense.defense_params or {}))
                defended_estimator = art_class(
                    self.get_model(),
                    preprocessing_defences=[defense],
                    **init_params,
                )
            case "postprocessor":
                defense = defense_class(**(self.defense.defense_params or {}))
                defended_estimator = art_class(
                    self.get_model(),
                    postprocessing_defences=[defense],
                    **init_params,
                )
            case "detector":
                match defense_subtype:
                    case "evasion":
                        defense = defense_class(**(self.defense.defense_params or {}))
                        defended_estimator = defense(self.get_model(), **init_params)
                    case "poison":
                        defense = defense_class(**(self.defense.defense_params or {}))
                        defended_estimator = defense(self.get_model(), **init_params)
                    case _:
                        raise NotImplementedError(
                            f"Detector subtype '{defense_subtype}' is not implemented yet.",
                        )
                # Overwrite the _score method to handle each
            case "trainer":
                defense = defense_class(**(self.defense.defense_params or {}))
                defended_estimator = defense(self._model, **init_params)
            case "transformer":
                defense = defense_class(**(self.defense.defense_params or {}))
                defended_estimator = defense(
                    self._model,
                    input_transformations=[defense],
                    **init_params,
                )
            case "regularizer":
                raise NotImplementedError(
                    "Regularizer defenses are not implemented yet.",
                )
            case None:
                defense = None
                defense_params = {**self.defense.defense_params, **init_params}
                defended_estimator = art_class(
                    self.get_model(),
                    **defense_params,
                )
            case "_":
                raise NotImplementedError(
                    f"Defense type '{defense_type}' is not implemented yet.",
                )
        # Some defences can optionally be applied during training or prediction
        end = time.process_time()
        self._apply_fit = getattr(defense, "_apply_fit", True)

        self.defense_application_time = end - start
        end = time.process_time()
        return defended_estimator

    def _train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the internal model using the provided feature matrix and target vector.

        Args
        -------
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target vector for training.

        Raises
        -------
            ValueError: If the internal model is not initialized.

        Side Effects
        -------
            - Fits the internal model to the data.
            - Records the training time in seconds.
            - Logs the training duration.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        start_time = time.process_time()
        assert hasattr(self._model, "fit"), "Model does not have a fit method"
        fit_params = {} if not hasattr(self, "fit_params") else self.fit_params
        self._model.fit(X, y, **fit_params)
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
        y_pred = self._model.predict(X)

        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities for the input data using the trained model.

        Args
        -------
            X (pd.DataFrame): Input features for which to predict probabilities.

        Returns
        -------
            pd.DataFrame: Predicted class probabilities for each sample in X.

        Raises
        -------
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

        Args
        -------
            y_true (pd.Series): True labels of the classification task.
            y_pred (pd.Series): Predicted labels from the classifier.

        Returns
        -------
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
        try:
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
        except ValueError as ve:
            if "mix of binary and continuous" in str(ve):
                new_y_pred = np.argmax(y_pred, axis=1)
                return self._classification_scores(y_true, new_y_pred)
            elif "mix of multiclass and continuous-multioutput" in str(ve):
                new_y_pred = np.argmax(y_pred, axis=1)
                return self._classification_scores(y_true, new_y_pred)
            elif "y_prob contains values greater than 1" in str(ve):
                # Convert class labels to one-hot encoding
                classes = np.unique(y_true)
                y_pred_one_hot = np.zeros((len(y_pred), len(classes)))
                for i, class_label in enumerate(classes):
                    y_pred_one_hot[:, i] = np.array(y_pred == class_label).astype(int)
                    logloss = log_loss(y_true=y_true, y_pred=y_pred_one_hot)
            elif "mix of multiclass and multilabel-indicator" in str(ve):
                new_y_pred = np.argmax(y_pred, axis=1)
                return self._classification_scores(y_true, new_y_pred)
            else:
                logger.error(f"Error computing classification scores: {ve}")
                raise ve
        except Exception as e:
            logger.error(f"Error computing classification scores: {e}")
            raise e
        try:
            logloss = log_loss(y_true=y_true, y_pred=y_pred)
        except ValueError as e:
            if "y_prob contains values greater than 1" in str(e):
                y_true = pd.get_dummies(y_true).values
                y_pred = pd.get_dummies(y_pred).values
                logloss = log_loss(y_true=y_true, y_pred=y_pred)
            else:
                raise e
        scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "log_loss": logloss,
        }
        return scores

    def _regression_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Calculate regression error metrics between true and predicted values.

        Args
        -------
            y_true (pd.Series): Series of true target values.
            y_pred (pd.Series): Series of predicted target values.

        Returns
        -------
            dict: Dictionary containing the following regression metrics:
                - 'mse': Mean Squared Error
                - 'rmse': Root Mean Squared Error
                - 'mae': Mean Absolute Error

        Raises
        -------
            AssertionError: If y_true and y_pred do not have the same length.
        """
        # Ensure that y_true and y_pred have the same length
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        mse = ((y_true - y_pred) ** 2).mean()
        rmse = mse**0.5
        mae = np.abs(y_true - y_pred).mean()
        try:
            logloss = log_loss(y_true=y_true, y_pred=y_pred)
        except ValueError as e:
            if "y_prob contains values greater than 1" in str(e):
                y_true = pd.get_dummies(y_true).values
                y_pred = pd.get_dummies(y_pred).values
                logloss = log_loss(y_true=y_true, y_pred=y_pred)
            else:
                raise e
        scores = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "log_loss": logloss,
        }
        return scores

    def _score(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Compute and log performance scores for classification or regression.

        -----
        Args
            y_true (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values.

        -----
        Returns
            dict: Dictionary of rounded performance scores.

        -----
        Side Effects
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

    def _load_predictions(self, filepath: str):
        """
        Loads predictions from a specified CSV file.

        Args
        -------
            filepath (str): The path to the CSV file containing predictions.
        Raises
        -------
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded predictions are not in a valid format.
            Exception: For any other issues during the loading process.
        Side Effects
        -------
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
        train_predictions_file,
        test_predictions_file,
        times,
    ):
        """
        Loads training and prediction data from the specified file paths and updates the provided times dictionary
        with relevant metadata.

        Parameters
        ----------
        train_predictions_file : str or Path or None
            File path to the training predictions. If None or the file does not exist, training predictions are not loaded.
        test_predictions_file : str or Path or None
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
        if train_predictions_file is not None and Path(train_predictions_file).exists():
            self.training_predictions = self._load_predictions(
                train_predictions_file,
            )
            assert (
                self.training_prediction_time is not None
            ), "Training prediction time must be set if training predictions are loaded"
            times["training_prediction_time"] = self.training_prediction_time
            times["training_n"] = len(self.training_predictions)

        # Load the predictions if provided
        if test_predictions_file is not None and Path(test_predictions_file).exists():
            self.predictions = self._load_predictions(test_predictions_file)
            assert (
                self.prediction_time is not None
            ), "Prediction time must be set if predictions are loaded"
            times["prediction_time"] = self.prediction_time
            times["prediction_n"] = len(self.predictions)
        return times

    def _load_score_file(self, score_file):
        """
        Loads score data from the specified file, merges it with existing scores, and extracts timing and count metrics.

        Parameters
        ----------
        score_file : str or Path
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
        if score_file is not None and Path(score_file).exists():
            new_score_dict = self.load_scores(score_file)
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
        model_file: Union[str, None] = None,
        test_predictions_file: Union[str, None] = None,
        train_predictions_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Executes the model workflow: training, prediction, scoring, and model persistence.

        Parameters
        ----------
        data : DataConfig
            An instance of DataConfig containing training and test data.
        model_file : str or None, optional
            Path to save or load the model. If provided, the model will be loaded from or saved to this path.
        test_predictions_file : str or None, optional
            Path to save the predictions. If provided, the predictions will be saved to this path.
        score_file : str or None, optional
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

        # Load the score_file if provided
        times = self._load_score_file(score_file)

        # Load predictions from filepaths and update times
        times = self._load_all_predictions(
            train_predictions_file,
            test_predictions_file,
            times,
        )

        # Train the model if training data is provided and model is not already trained
        times = self._load_or_train_model(data, model_file, times)
        # Apply defense if specified

        self._evaluate_and_score(data, times)
        if train_predictions_file is not None:
            self.save_data(
                self.training_predictions,
                train_predictions_file,
            )
        if test_predictions_file is not None:
            self.save_data(self.predictions, test_predictions_file)
        if score_file is not None:
            self.save_scores(self.score_dict, score_file)
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
                "Training predictions already available, skipping prediction step.",
            )

        # Score training predictions if true labels are available and scoring not already done
        if self.training_score_time is None or self.score_dict is None:
            if self.training_predictions is not None:
                start = time.process_time()
                train_scores = self._score(data.y_train, self.training_predictions)
                self.training_score_time = time.process_time() - start
                # Prefix training scores with 'train_'
                train_scores = {
                    f"training_{key}": value for key, value in train_scores.items()
                }
                if "training_loss_curve" in train_scores:
                    del train_scores["training_loss_curve"]
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
        if (
            self.training_score_time is None
            or self.prediction_score_time is None
            or self.score_dict is None
        ):
            if data.y_test is not None and self.predictions is not None:
                test_scores = self._score(data.y_test, self.predictions)
                if self.score_dict is None:
                    self.score_dict = {}
                self.score_dict = {**self.score_dict, **test_scores}
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

    def _load_or_train_model(self, data, model_file, times):
        """
        Loads a model from the specified filepath if it exists and is trained, or trains a new model using the provided data.
        If a model file exists at `model_file`, attempts to load and validate that the model is fitted.
        If the loaded model is not fitted, or if no model file exists, trains a new model using `data.X_train` and `data.y_train`.
        Updates the `times` dictionary with training time and number of training samples.
        Saves the trained model to `model_file` if provided and a new model was trained.
        Raises:
            ValueError: If neither a model nor a filepath is provided, or if the model is not trained after loading/training.
            NotFittedError: If the model is not initialized.
        Args:
            data: An object containing training data (`X_train`, `y_train`).
            model_file (str or Path or None): Path to the model file to load or save.
            times (dict): Dictionary to store training time and number of training samples.
        Returns:
            dict: Updated `times` dictionary with training metadata.
        """
        match self._model, model_file:
            case None, None:  # Neither model nor filepath provided
                raise ValueError(
                    "Model not trained or loaded. Please train or load a model before prediction.",
                )
            case _, _:  # Model and/or filepath provided
                if model_file is not None and Path(model_file).exists():
                    logger.info(f"Model file {model_file} exists. Loading model.")
                    self = self.load(model_file)
                    try:
                        # Validate that the loaded model is fitted
                        check_is_fitted(self._model)
                        logger.info("Model loaded and is fitted.")
                    except NotFittedError:
                        logger.warning(
                            "Loaded model is not fitted. Training a new model.",
                        )
                        self._train(data.X_train, data.y_train)
                        try:
                            check_is_fitted(self._model)
                        except NotFittedError:
                            raise ValueError(
                                "Model is not trained. Please train the model before prediction.",
                            )
                        assert hasattr(
                            self,
                            "_model",
                        ), "Model not initialized after training"
                        if self.defense is not None:
                            self._model = self._apply_defense(data)
                        times["training_time"] = self.training_time
                        times["training_n"] = self._training_n
                        # Save the newly trained mode
                else:
                    # train the model if no model exists at the filepath
                    self._train(data.X_train, data.y_train)
                    try:
                        check_is_fitted(self._model)
                    except NotFittedError:
                        raise ValueError(
                            "Model is not trained. Please train the model before prediction.",
                        )
                    times["training_time"] = self.training_time
                    times["training_n"] = self._training_n
                    if self.defense is not None:
                        self._model = self._apply_defense(data)
                    if model_file is not None:
                        self.save(filepath=model_file)
        # Validate model is trained
        if self._model is None:
            raise NotFittedError("Model is not initialized")

        return times
