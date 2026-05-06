import time
import logging
from typing import Any, Union
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
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor
from art.config import ART_NUMPY_DTYPE

from ..data import DataConfig
from ..score.base import ScorerDictConfig, coerce_scorer_config as _coerce_scorer_config
from ..utils import ConfigBase, load_class, round_scores

art_model_types = tuple(
    [
        ScikitlearnClassifier,
        ScikitlearnRegressor,
        PyTorchClassifier,
        PyTorchRegressor,
    ],
)


logger = logging.getLogger(__name__)

AUTO_SCORER = "auto"

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
sklearn_models = list(sklearn_dict.values())


@dataclass(eq=False)
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
    defense : dict or None
        Optional defense configuration applied after base model training.
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
    defense_application_time : float or None
        Time taken to apply the configured defense (in seconds).
    alias : str or None
        An optional alias for the model configuration.
    score_dict : dict or None
        Dictionary containing the latest computed scores and timing information.
    training_n : int or None
        Number of training samples.
    prediction_n : int or None
        Number of prediction samples.
    training_predictions : pd.Series, pd.DataFrame, np.ndarray, list, or None
        Predictions made on the training data.
    predictions : pd.Series, pd.DataFrame, np.ndarray, list, or None
        Predictions made on the prediction data.
    _target_ : str
        Internal identifier for the class.

    Methods
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

    # Configuration fields
    model_type: Union[str, None] = None
    classifier: Union[bool, None, str] = True
    model_params: dict = None
    probability: bool = False
    alias: Union[str, None] = None
    defense: Any = None
    plugins: Union[list, None] = None
    scorer: Any = AUTO_SCORER

    # Runtime/model state fields
    _model: Any = None
    score_dict: dict = None
    training_time: Union[float, None] = None
    prediction_time: Union[float, None] = None
    training_prediction_time: Union[float, None] = None
    training_score_time: Union[float, None] = None
    prediction_score_time: Union[float, None] = None
    defense_application_time: Union[float, None] = None
    training_n: Union[int, None] = None
    prediction_n: Union[int, None] = None
    training_predictions: Any = None
    predictions: Any = None
    training_probabilities: Any = None
    probabilities: Any = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _defense_pipeline: Any = field(default=None, repr=False, compare=False)

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
        # Dynamically import the model class only when not already initialized.
        if not hasattr(self, "_model") or self._model is None:
            self._initialize_model()
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        for attr in [
            "training_time",
            "prediction_time",
            "training_prediction_time",
            "training_score_time",
            "prediction_score_time",
            "defense_application_time",
            "training_n",
            "prediction_n",
            "training_predictions",
            "predictions",
            "training_probabilities",
            "probabilities",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.model.ModelConfig"
        if hasattr(self, "defense") and hasattr(self.defense, "defense_name"):
            if self.defense.defense_name in [
                "",
                None,
                "None",
                "null",
                "Null",
                "NULL",
                "none",
                "N/A",
                "n/a",
            ]:
                self.defense = None
        if self.classifier in ["classifier", True]:
            self.classifier = True
        elif self.classifier in ["regressor", False]:
            self.classifier = False
        else:
            self.classifier = None
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                "deckard.score.base.DefaultClassifierConfig"
                if self.classifier
                else "deckard.score.base.DefaultRegressorConfig"
            ),
        )
        if self.plugins is None:
            self.plugins = []
        if self.defense is not None:
            from .defend import DefensePipelineConfig

            self.defense = DefensePipelineConfig.coerce(self.defense)

    def _initialize_model(self):
        # Initialize model through the shared loader used by config objects.
        if self.model_params is not None:
            self._model = load_class(self.model_type, **self.model_params)
        else:
            self._model = load_class(self.model_type)
        if hasattr(self._model, "get_params"):
            self.model_params = self._model.get_params()
        else:
            assert isinstance(
                self.model_params,
                (dict, DictConfig),
            ), f"model_params must be a dict if model does not have get_params method. Got {type(self.model_params)}"
        if hasattr(self._model, "predict_proba"):
            self.probability = True

    def __hash__(self):
        return super().__hash__()

    def _instantiate_plugin(self, plugin_spec: Any):
        if isinstance(plugin_spec, dict):
            spec = dict(plugin_spec)
            class_path = spec.pop("name", spec.pop("_target_", None))
            if class_path is None:
                raise ValueError(
                    "Plugin dict must include 'name' or '_target_'",
                )
            return load_class(class_path, **spec)

        if isinstance(plugin_spec, str):
            return load_class(plugin_spec)

        if isinstance(plugin_spec, type):
            return plugin_spec()

        return plugin_spec

    def _get_plugins(self) -> list:
        if self._plugin_objects is None:
            plugin_specs = self.plugins if self.plugins is not None else []
            if not isinstance(plugin_specs, list):
                raise TypeError(
                    f"plugins must be a list, got {type(plugin_specs)}",
                )
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

    def _merge_plugin_scores(self, hook_outputs):
        if self.score_dict is None:
            self.score_dict = {}
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(output)

    def _copy_runtime_state_to(self, target) -> None:
        runtime_fields = [
            "_model",
            "score_dict",
            "training_predictions",
            "predictions",
            "training_probabilities",
            "probabilities",
            "training_time",
            "prediction_time",
            "training_prediction_time",
            "training_score_time",
            "prediction_score_time",
            "defense_application_time",
            "training_n",
            "prediction_n",
        ]
        for attr in runtime_fields:
            setattr(target, attr, getattr(self, attr, None))

    def _require_defense_pipeline(self):
        """Return configured defense pipeline or raise on invalid type."""
        if self.defense is None:
            self._defense_pipeline = None
            return None

        from .defend import DefensePipelineConfig

        self.defense = DefensePipelineConfig.coerce(self.defense)
        self._defense_pipeline = self.defense
        return self._defense_pipeline

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
                "nb_classes": (len(set(data.y_train)) if self.classifier else None),
            }
        if "preprocessing" not in init_params:
            init_params["preprocessing"] = None
        return art_class, init_params

    def get_art_model(self, data: DataConfig):
        """Get the ART model estimator.

        Parameters
        ----------
        data : DataConfig
            The data configuration containing training data.

        Returns
        -------
        BaseEstimator
            The ART model estimator.
        """
        if self.defense is None:
            art_class, init_params = self.get_art_class(data)
            art_model = art_class(self._model, **init_params)
        else:
            art_model = self._apply_defense(data)
        return art_model

    def get_model(self) -> BaseEstimator:
        """Get the model's estimator.

        Returns
        -------
        BaseEstimator
            The model's estimator.
        """
        if self._model is None:
            raise ValueError("Model is not fitted yet.")
        if isinstance(self._model, art_model_types):
            return self._model.model
        else:
            return self._model

    def _apply_defense(self, data) -> BaseEstimator:
        """Delegate defense application to DefensePipelineConfig."""

        if self.defense is None:
            return self._model
        if self._model is None:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            )

        defense_pipeline = self._require_defense_pipeline()
        if defense_pipeline is None:
            return self._model
        stage = defense_pipeline.resolve_stage(
            default_stage="post_fit_pre_predict",
            model=self,
            data=data,
        )
        defended_estimator = defense_pipeline.apply(
            estimator=self._model,
            data=data,
            stage=stage,
        )
        self.defense_application_time = getattr(
            defense_pipeline,
            "defense_application_time",
            None,
        )
        if getattr(defense_pipeline, "score_dict", None):
            if self.score_dict is None:
                self.score_dict = {}
            self.score_dict.update(defense_pipeline.score_dict)
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
        self.training_n = len(y)
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
        try:
            y_pred = self._model.predict(X)
        except TypeError as e:
            if "loop of ufunc does not support argument" in str(e):
                X_array = np.array(X, dtype=ART_NUMPY_DTYPE)
                y_pred = self._model.predict(X_array)
            elif "can't convert" in str(e):
                X_array = np.array(X, dtype=ART_NUMPY_DTYPE)
                y_pred = self._model.predict(X_array)
            else:
                raise e

        # Some postprocessors can emit invalid multi-class matrices (e.g., all ones).
        # If detected, fall back to the underlying estimator predictions.
        if self.classifier and isinstance(y_pred, (pd.DataFrame, np.ndarray)):
            y_pred_array = np.asarray(y_pred)
            if y_pred_array.ndim == 2 and y_pred_array.shape[1] > 1:
                row_sums = np.sum(y_pred_array, axis=1)
                invalid_matrix = (
                    np.isfinite(y_pred_array).all() and np.allclose(y_pred_array, 1.0)
                ) or (np.isfinite(row_sums).all() and np.all(row_sums > 1.0 + 1e-8))
                if invalid_matrix:
                    base_model = getattr(self._model, "model", None)
                    if base_model is not None and hasattr(
                        base_model,
                        "predict",
                    ):
                        logger.warning(
                            "Detected invalid classifier prediction matrix from wrapped model; "
                            "falling back to underlying estimator predictions.",
                        )
                        y_pred = base_model.predict(X)
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

    def _classification_scores(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> dict:
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
        assert len(y_true) == len(
            y_pred,
        ), "y_true and y_pred must have the same length"
        # Ensure that y_true.shape and y_pred.shape are compatible
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true.ndim > 1 and y_pred.ndim == 1:
            y_pred = pd.get_dummies(y_pred).values
            y_prob = y_pred
        elif y_true.ndim == 1 and y_pred.ndim > 1:
            y_prob = y_pred.copy()
            if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == 1:
                # Handle binary models that emit a single score/probability column.
                binary_scores = y_pred_arr.reshape(-1)
                threshold = 0.5
                if np.nanmin(binary_scores) < 0.0 or np.nanmax(binary_scores) > 1.0:
                    threshold = 0.0
                classes = np.unique(y_true_arr[~pd.isna(y_true_arr)])
                if len(classes) == 2 and np.issubdtype(
                    np.asarray(classes).dtype,
                    np.number,
                ):
                    sorted_classes = np.sort(np.asarray(classes, dtype=float))
                    low_label = sorted_classes[0]
                    high_label = sorted_classes[1]
                    y_pred = np.where(
                        binary_scores >= threshold,
                        high_label,
                        low_label,
                    )
                else:
                    y_pred = (binary_scores >= threshold).astype(int)
            else:
                y_pred = np.argmax(y_prob, axis=1)
        elif y_true.ndim > 1 and y_pred.ndim > 1:
            assert (
                y_true.shape == y_pred.shape
            ), "y_true and y_pred must have the same shape"
            y_prob = y_pred.copy()
            y_prob = np.argmax(y_prob, axis=1)
        else:
            y_prob = y_pred.copy()
        assert (
            y_true.shape[0] == y_pred.shape[0]
        ), "y_true and y_pred must have the same number of samples"
        if y_true.ndim > 1:
            assert (
                y_true.shape[1] == y_pred.shape[1]
            ), "y_true and y_pred must have the same number of classes"
        try:
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        except Exception as e:
            logger.error(f"Error computing classification scores: {e}")
            raise e
        try:
            logloss = log_loss(y_true=y_true, y_pred=y_prob)
        except ValueError as e:
            if "y_prob contains values greater than 1" in str(e):
                y_true = pd.get_dummies(y_true).values
                y_pred = pd.get_dummies(y_pred).values
                logloss = log_loss(y_true=y_true, y_pred=y_pred)
            else:
                logloss = np.nan
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
        assert len(y_true) == len(
            y_pred,
        ), "y_true and y_pred must have the same length"
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

    def _score(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        mode: str = "test",
        **kwargs,
    ) -> dict:
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
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"ModelConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        scores = self.scorer(y_true=y_true, y_pred=y_pred, mode=mode, **kwargs)
        return round_scores(
            scores=scores,
            n_samples=len(y_true),
            logger_obj=logger,
        )

    def _decode_predictions_for_persistence(self, y_pred, y_true=None):
        """Persist classifier outputs with explicit probability-vs-label behavior."""
        if not self.classifier:
            return y_pred
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            return y_pred
        if y_pred_arr.ndim != 2:
            return y_pred

        if y_pred_arr.shape[1] == 1:
            binary_scores = y_pred_arr.reshape(-1)
            threshold = 0.5
            if np.nanmin(binary_scores) < 0.0 or np.nanmax(binary_scores) > 1.0:
                threshold = 0.0
            if y_true is not None:
                y_true_arr = np.asarray(y_true)
                classes = np.unique(y_true_arr[~pd.isna(y_true_arr)])
                if len(classes) == 2 and np.issubdtype(
                    np.asarray(classes).dtype,
                    np.number,
                ):
                    sorted_classes = np.sort(np.asarray(classes, dtype=float))
                    low_label = sorted_classes[0]
                    high_label = sorted_classes[1]
                    return np.where(
                        binary_scores >= threshold,
                        high_label,
                        low_label,
                    )
            return (binary_scores >= threshold).astype(int)

        return np.argmax(y_pred_arr, axis=1)

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
            if not isinstance(
                predictions,
                (pd.Series, pd.DataFrame, np.ndarray, list),
            ):
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
        if (
            train_predictions_file is not None
            and Path(train_predictions_file).exists()
        ):
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
        training_probabilities_file: Union[str, None] = None,
        test_probabilities_file: Union[str, None] = None,
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

        self._run_plugin_hook(
            "before_load_score",
            data=data,
            score_file=score_file,
        )

        # Load the score_file if provided
        times = self._load_score_file(score_file)
        self._run_plugin_hook(
            "after_load_score",
            data=data,
            score_file=score_file,
            times=times,
        )

        # Load predictions from filepaths and update times
        self._run_plugin_hook(
            "before_load_predictions",
            data=data,
            train_predictions_file=train_predictions_file,
            test_predictions_file=test_predictions_file,
        )
        times = self._load_all_predictions(
            train_predictions_file,
            test_predictions_file,
            times,
        )
        self._run_plugin_hook(
            "after_load_predictions",
            data=data,
            train_predictions_file=train_predictions_file,
            test_predictions_file=test_predictions_file,
            times=times,
        )

        # Train the model if training data is provided and model is not already trained
        self._run_plugin_hook(
            "before_train_or_load_model",
            data=data,
            model_file=model_file,
            times=times,
        )
        times = self._load_or_train_model(data, model_file, times)
        self._run_plugin_hook(
            "after_train_or_load_model",
            data=data,
            model_file=model_file,
            times=times,
        )
        # Apply defense if specified

        self._run_plugin_hook(
            "before_evaluate",
            data=data,
            times=times,
        )
        times = self._evaluate_and_score(
            data,
            times,
            persist_training_predictions=train_predictions_file is not None,
            persist_test_predictions=test_predictions_file is not None,
            persist_training_probabilities=training_probabilities_file is not None,
            persist_test_probabilities=test_probabilities_file is not None,
        )
        hook_outputs = self._run_plugin_hook(
            "after_evaluate",
            data=data,
            times=times,
        )
        self._merge_plugin_scores(hook_outputs)

        self._run_plugin_hook(
            "before_persist",
            data=data,
            times=times,
            model_file=model_file,
            test_predictions_file=test_predictions_file,
            train_predictions_file=train_predictions_file,
            training_probabilities_file=training_probabilities_file,
            test_probabilities_file=test_probabilities_file,
            score_file=score_file,
        )
        if (
            train_predictions_file is not None
            and self.training_predictions is not None
        ):
            self.save_data(
                self.training_predictions,
                train_predictions_file,
            )
        if test_predictions_file is not None and self.predictions is not None:
            self.save_data(self.predictions, test_predictions_file)
        if (
            training_probabilities_file is not None
            and self.training_probabilities is not None
        ):
            self.save_data(
                self.training_probabilities,
                training_probabilities_file,
            )
        if test_probabilities_file is not None and self.probabilities is not None:
            self.save_data(self.probabilities, test_probabilities_file)
        if score_file is not None:
            self.save_scores(self.score_dict, score_file)
        hook_outputs = self._run_plugin_hook(
            "after_persist",
            data=data,
            times=times,
            model_file=model_file,
            test_predictions_file=test_predictions_file,
            train_predictions_file=train_predictions_file,
            training_probabilities_file=training_probabilities_file,
            test_probabilities_file=test_probabilities_file,
            score_file=score_file,
        )
        self._merge_plugin_scores(hook_outputs)
        return self.score_dict

    def _evaluate_and_score(
        self,
        data: DataConfig,
        times: dict = None,
        persist_training_predictions: bool = False,
        persist_test_predictions: bool = False,
        persist_training_probabilities: bool = False,
        persist_test_probabilities: bool = False,
    ):
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
        if self.defense is not None and self._model is not None:
            defense_pipeline = self._require_defense_pipeline()
            stage = defense_pipeline.resolve_stage(
                default_stage="post_fit_pre_predict",
                model=self,
                data=data,
            )
            if stage == "before_predict":
                self._model = defense_pipeline.apply(
                    estimator=self._model,
                    data=data,
                    stage=stage,
                )
                self.defense_application_time = getattr(
                    defense_pipeline,
                    "defense_application_time",
                    None,
                )
                if self.defense_application_time is not None:
                    times["defense_application_time"] = self.defense_application_time
                if getattr(defense_pipeline, "score_dict", None):
                    if self.score_dict is None:
                        self.score_dict = {}
                    self.score_dict.update(defense_pipeline.score_dict)

        # Compute predictions transiently for scoring unless already provided.
        if self.training_predictions is not None:
            train_predictions = self.training_predictions
            times["training_n"] = len(train_predictions)
        else:
            start_time = time.process_time()
            train_predictions = self._predict(data.X_train)
            end_time = time.process_time()
            self.training_prediction_time = end_time - start_time
            times["training_prediction_time"] = self.training_prediction_time
            times["training_n"] = len(train_predictions)
            if persist_training_predictions:
                self.training_predictions = self._decode_predictions_for_persistence(
                    train_predictions,
                    y_true=data.y_train,
                )
            if persist_training_probabilities and self.classifier:
                try:
                    if hasattr(self._model, "predict_proba"):
                        try:
                            self.training_probabilities = self._model.predict_proba(
                                data.X_train,
                            )
                        except TypeError as e:
                            if "loop of ufunc does not support argument" in str(
                                e,
                            ) or "can't convert" in str(e):
                                X_array = np.array(
                                    data.X_train,
                                    dtype=ART_NUMPY_DTYPE,
                                )
                                self.training_probabilities = (
                                    self._model.predict_proba(
                                        X_array,
                                    )
                                )
                            else:
                                raise e
                    else:
                        self.training_probabilities = self._predict(
                            data.X_train,
                        )
                except ValueError as e:
                    logger.warning(
                        "Skipping training probability persistence: %s",
                        e,
                    )
                    self.training_probabilities = None

        # Score training predictions from current run.
        if train_predictions is not None:
            if self.scorer is not None:
                start = time.process_time()
                train_scores = self._score(
                    data.y_train,
                    train_predictions,
                    mode="train",
                    data=data,
                    model=self,
                )
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
        if self.predictions is not None:
            test_predictions = self.predictions
            times["prediction_n"] = len(test_predictions)
        else:
            if data.X_test is not None:
                start_time = time.process_time()
                test_predictions = self._predict(data.X_test)
                end_time = time.process_time()
                self.prediction_time = end_time - start_time
                times["prediction_time"] = self.prediction_time
                times["prediction_n"] = len(test_predictions)
                if persist_test_predictions:
                    self.predictions = self._decode_predictions_for_persistence(
                        test_predictions,
                        y_true=data.y_test,
                    )
                if persist_test_probabilities and self.classifier:
                    try:
                        if hasattr(self._model, "predict_proba"):
                            try:
                                self.probabilities = self._model.predict_proba(
                                    data.X_test,
                                )
                            except TypeError as e:
                                if "loop of ufunc does not support argument" in str(
                                    e,
                                ) or "can't convert" in str(e):
                                    X_array = np.array(
                                        data.X_test,
                                        dtype=ART_NUMPY_DTYPE,
                                    )
                                    self.probabilities = self._model.predict_proba(
                                        X_array,
                                    )
                                else:
                                    raise e
                        else:
                            self.probabilities = self._predict(data.X_test)
                    except ValueError as e:
                        logger.warning(
                            "Skipping test probability persistence: %s",
                            e,
                        )
                        self.probabilities = None
            else:
                raise ValueError("No test data available for prediction.")
        # Score test predictions from current run.
        if data.y_test is not None and test_predictions is not None:
            if self.scorer is not None:
                start = time.process_time()
                test_scores = self._score(
                    data.y_test,
                    test_predictions,
                    mode="test",
                    data=data,
                    model=self,
                )
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

        def _is_model_fitted(estimator, X_sample=None, depth: int = 0) -> bool:
            if estimator is None:
                return False
            if depth > 2:
                return False

            try:
                check_is_fitted(estimator)
                return True
            except Exception:
                pass

            # Support wrapped estimators (e.g., ART wrappers exposing underlying model)
            for attr in ["model", "_model", "estimator"]:
                wrapped = getattr(estimator, attr, None)
                if wrapped is None or wrapped is estimator:
                    continue
                if _is_model_fitted(
                    wrapped,
                    X_sample=X_sample,
                    depth=depth + 1,
                ):
                    return True

            for attr in ["is_fitted_", "fitted", "_is_fitted"]:
                if hasattr(estimator, attr):
                    try:
                        return bool(getattr(estimator, attr))
                    except Exception:
                        continue

            # Framework-agnostic fallback: if prediction works on one sample, treat as fitted.
            if X_sample is not None and hasattr(estimator, "predict"):
                try:
                    if isinstance(X_sample, (pd.DataFrame, pd.Series)):
                        sample = X_sample.iloc[:1]
                    else:
                        sample = X_sample[:1]
                    _ = estimator.predict(sample)
                    return True
                except Exception:
                    pass

            return False

        match self._model, model_file:
            case None, None:  # Neither model nor filepath provided
                raise ValueError(
                    "Model not trained or loaded. Please train or load a model before prediction.",
                )
            case _, _:  # Model and/or filepath provided
                if model_file is not None and Path(model_file).exists():
                    logger.info(
                        f"Model file {model_file} exists. Loading model.",
                    )
                    self = self.load(model_file)
                    if _is_model_fitted(self._model, X_sample=data.X_train):
                        logger.info("Model loaded and is fitted.")
                    else:
                        logger.warning(
                            "Loaded model is not fitted. Training a new model.",
                        )
                        logger.info(
                            f"Training model on {len(data.y_train)} samples...",
                        )
                        self._train(data.X_train, data.y_train)
                        assert hasattr(
                            self,
                            "_model",
                        ), "Model not initialized after training"
                        if self.defense is not None:
                            self._model = self._apply_defense(data)
                        times["training_time"] = self.training_time
                        times["training_n"] = self.training_n
                        # Save the newly trained mode
                else:
                    # train the model if no model exists at the filepath
                    logger.info(
                        f"Training model on {len(data.y_train)} samples...",
                    )
                    model_is_fitted = _is_model_fitted(
                        self._model,
                        X_sample=data.X_train,
                    )

                    # Do not trust timing metadata from loaded score files as proof of fitted state.
                    if not model_is_fitted:
                        self._train(data.X_train, data.y_train)
                    times["training_time"] = self.training_time
                    times["training_n"] = self.training_n

                    if model_file is not None:
                        self.save(filepath=model_file)
        # Validate model is trained
        if self._model is None:
            raise NotFittedError("Model is not initialized")
        if self.defense is not None:
            defense_pipeline = self._require_defense_pipeline()
            stage = defense_pipeline.resolve_stage(
                default_stage="post_fit_pre_predict",
                model=self,
                data=data,
            )
            if stage == "post_fit_pre_predict":
                self._model = self._apply_defense(data)
                if getattr(self, "defense_application_time", None) is not None:
                    times["defense_application_time"] = self.defense_application_time
        return times
