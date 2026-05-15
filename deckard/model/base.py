import time
import logging
from typing import Any, Literal, Union
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
from omegaconf import DictConfig

import numpy as np
import pandas as pd


from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from ..data import DataConfig
from ..score.base import (  # noqa: F401
    ScorerDictConfig,
    coerce_scorer_config as _coerce_scorer_config,
)
from ..utils import (
    ConfigBase,
    load_class,
    probabilities_from_model_outputs,
    round_scores,
    normalize_plugin_specs,
    is_null_config_value,
)
from ..frameworks import ModelContractMixin, FrameworkModelConfig
from ._mixins import (
    ModelHookRuntimeMixin,
    ModelPrunerMixin,
    ModelTrainingMixin,
    PretrainedModelMixin,
)
from ..frameworks.core import ArrayLike, EstimatorLike, MatrixLike

logger = logging.getLogger(__name__)

AUTO_SCORER = "auto"

__all__ = ["ModelConfig"]


@lru_cache(maxsize=1)
def _get_art_symbols() -> dict[str, Any]:
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
    art_model_types = (
        ScikitlearnClassifier,
        ScikitlearnRegressor,
        PyTorchClassifier,
        PyTorchRegressor,
    )
    return {
        "classifier_dict": classifier_dict,
        "regressor_dict": regressor_dict,
        "sklearn_dict": sklearn_dict,
        "sklearn_models": list(sklearn_dict.values()),
        "art_model_types": art_model_types,
    }


def _art_numpy_dtype():
    try:
        from art.config import ART_NUMPY_DTYPE

        return ART_NUMPY_DTYPE
    except Exception:
        return np.float32


def _is_art_model_instance(model_obj: Any) -> bool:
    try:
        art_model_types = _get_art_symbols()["art_model_types"]
    except Exception:
        return False
    return isinstance(model_obj, art_model_types)


@dataclass(eq=False, kw_only=True)
class ModelConfig(
    ModelTrainingMixin,
    PretrainedModelMixin,
    ModelPrunerMixin,
    ModelHookRuntimeMixin, #Allows for user-configured plugins
    ModelContractMixin, #Ensures that the final object has necessary components, according to the Hook
    ConfigBase, # Persistence, Hashing, 
    FrameworkModelConfig, # Defines order of operations
):
    """Runtime model configuration with plugin-aware training/evaluation orchestration.

    Model behavior is resolved from ``model_type`` and runtime context. This
    class owns model instantiation, training/load flow, prediction, scoring,
    persistence, and optional defense-pipeline integration.

    Plugin hooks
    ------------
    before_load_score(self, *, data, score_file)
        Runs before reading persisted score/timing artifacts.
    after_load_score(self, *, data, score_file, times)
        Runs after reading persisted score/timing artifacts.
    before_load_predictions(self, *, data, train_predictions_file, test_predictions_file)
        Runs before loading persisted predictions/probabilities.
    after_load_predictions(self, *, data, train_predictions_file, test_predictions_file, times)
        Runs after loading persisted predictions/probabilities.
    before_train_or_load_model(self, *, data, model_file, times)
        Runs before model load-or-train resolution.
    after_train_or_load_model(self, *, data, model_file, times)
        Runs after model load-or-train resolution.
    before_evaluate(self, *, data, times)
        Runs before evaluation/scoring.
    after_evaluate(self, *, data, times)
        Runs after evaluation/scoring. Dict returns are merged into
        ``score_dict``.
    before_persist(self, *, data, times, model_file, test_predictions_file, train_predictions_file, training_probabilities_file, test_probabilities_file, score_file)
        Runs before persistence of model artifacts and score outputs.
    after_persist(self, *, data, times, model_file, test_predictions_file, train_predictions_file, training_probabilities_file, test_probabilities_file, score_file)
        Runs after persistence. Dict returns are merged into ``score_dict``.

    Parameter layers
    ----------------
    model_params : dict
        Model-constructor kwargs passed to the resolved estimator/class.
    defense : Any
        Optional defense pipeline/config applied after model training or load.
    plugins : list
        Plugin specs resolved at runtime and invoked through hook names.

    Family-specific parameter semantics
    ----------------------------------
    sklearn estimators
        ``model_params`` are forwarded directly to estimator constructors.
    wrapped or custom model classes
        ``model_params`` may be split between wrapper setup and underlying
        model kwargs.
    defense-enabled runs
        ``defense`` controls post-training estimator wrapping/application.

    Plugin hook runtime params
    --------------------------
    Hooks are orchestrated by ``_run_plugin_hook(hook_name, **kwargs)``.
    Core hook names used by ModelConfig runtime are:
    ``before_load_score``, ``after_load_score``, ``before_load_predictions``,
    ``after_load_predictions``, ``before_train_or_load_model``,
    ``after_train_or_load_model``, ``before_evaluate``, ``after_evaluate``,
    ``before_persist``, and ``after_persist``.
    Hook kwargs are phase-specific runtime objects supplied by model
    orchestration.
    """

    # Configuration fields
    model_type: Union[str, None] = None
    classifier: Union[bool, str] = True
    model_params: dict = None
    probability: bool = False
    alias: Union[str, None] = None
    defense: Any = None
    plugins: Union[list, None] = None
    scorer: Any = AUTO_SCORER
    score_mode: Literal["train", "test", "val"] = "test"

    # Runtime/model state fields
    _model: Any = None
    score_dict: dict = None
    training_time: Union[float, None] = None
    prediction_time: Union[float, None] = None
    val_prediction_time: Union[float, None] = None
    training_prediction_time: Union[float, None] = None
    training_score_time: Union[float, None] = None
    prediction_score_time: Union[float, None] = None
    val_score_time: Union[float, None] = None
    defense_application_time: Union[float, None] = None
    training_n: Union[int, None] = None
    prediction_n: Union[int, None] = None
    val_n: Union[int, None] = None
    training_predictions: Any = None
    predictions: Any = None
    val_predictions: Any = None
    training_probabilities: Any = None
    probabilities: Any = None
    val_probabilities: Any = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _defense_pipeline: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Initialize runtime defaults and normalize model-config state."""
        self._initialize_runtime_defaults()
        self._initialize_target_reference()
        self._normalize_classifier_flag()
        self._initialize_default_scorer()
        self._normalize_plugins()
        self._coerce_defense_config()

    def _initialize_runtime_defaults(self) -> None:
        """Initialize runtime attributes that must exist after construction."""
        if not hasattr(self, "_model") or self._model is None:
            self._initialize_model()
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        for attr in [
            "training_time",
            "prediction_time",
            "val_prediction_time",
            "training_prediction_time",
            "training_score_time",
            "prediction_score_time",
            "val_score_time",
            "defense_application_time",
            "training_n",
            "prediction_n",
            "val_n",
            "training_predictions",
            "predictions",
            "val_predictions",
            "training_probabilities",
            "probabilities",
            "val_probabilities",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)

    def _initialize_target_reference(self) -> None:
        """Ensure the canonical runtime target path is set."""
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.model.ModelConfig"

    def _normalize_classifier_flag(self) -> None:
        """Normalize classifier/regressor selector to a strict boolean."""
        if hasattr(self, "defense") and hasattr(self.defense, "defense_name"):
            if is_null_config_value(self.defense.defense_name):
                self.defense = None
        if self.classifier in ["classifier", True]:
            self.classifier = True
        elif self.classifier in ["regressor", False]:
            self.classifier = False
        else:
            raise ValueError(
                f"classifier must be boolean or one of ['classifier', 'regressor'], got {self.classifier}",
            )

    def _initialize_default_scorer(self) -> None:
        """Resolve model scorer defaults based on classifier mode."""
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                (
                    "deckard.score.base.DefaultClassifierConfig"
                    if self.classifier
                    else "deckard.score.base.DefaultRegressorConfig"
                ),
            ),
        )

    def _normalize_plugins(self) -> None:
        """Normalize configured plugin specs."""
        self.plugins = normalize_plugin_specs(self.plugins)

    def _coerce_defense_config(self) -> None:
        """Coerce defense configuration into the runtime pipeline wrapper."""
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

    def initialize_model(self) -> None:
        """Public entry-point for model initialisation. Idempotent."""
        if not hasattr(self, "_model") or self._model is None:
            self._initialize_model()

    def set_estimator(self, estimator: EstimatorLike) -> None:
        """Set the internal fitted estimator directly."""
        self._model = estimator

    @property
    def model(self) -> EstimatorLike | None:
        """Public accessor for the runtime estimator payload."""
        return self._model

    @model.setter
    def model(self, value: EstimatorLike | None) -> None:
        """Set the runtime estimator payload."""
        self._model = value

    def __hash__(self) -> int:
        return super().__hash__()

    def _require_defense_pipeline(self):
        """Return configured defense pipeline or raise on invalid type."""
        if self.defense is None:
            self.defense_pipeline = None
            return None

        from .defend import DefensePipelineConfig

        self.defense = DefensePipelineConfig.coerce(self.defense)
        self.defense_pipeline = self.defense
        return self.defense_pipeline

    @property
    def defense_pipeline(self) -> Any:
        """Public accessor for the resolved defense pipeline runtime object."""
        return self._defense_pipeline

    @defense_pipeline.setter
    def defense_pipeline(self, value: Any) -> None:
        """Set the resolved defense pipeline runtime object."""
        self._defense_pipeline = value

    def compose_defense_pipeline(self):
        """Compose defense pipeline behavior only when defense config is present."""
        return self._require_defense_pipeline()

    def compose_defense_behavior(
        self,
        data: "DataConfig",
        default_stage: str = "post_fit_pre_predict",
    ):
        """Compose a defense application callable and resolved stage for runtime use."""
        defense_pipeline = self.compose_defense_pipeline()
        if defense_pipeline is None:
            return None, None, None
        stage = defense_pipeline.resolve_stage(
            default_stage=default_stage,
            model=self,
            data=data,
        )

        def _apply(estimator: EstimatorLike) -> EstimatorLike:
            return defense_pipeline.apply(
                estimator=estimator,
                data=data,
                stage=stage,
            )

        return _apply, defense_pipeline, stage

    def get_art_class(self, data: "DataConfig"):
        try:
            art_symbols = _get_art_symbols()
        except Exception as exc:
            raise ImportError(
                "ART estimators are required for wrapped model access. Install optional dependencies that include ART.",
            ) from exc

        if self.model_type is None:
            raise ValueError(
                "model_type must be set before creating an ART model wrapper",
            )

        art_class = (
            art_symbols["classifier_dict"][self.model_type.split(".")[-1]]
            if self.classifier
            else art_symbols["regressor_dict"][self.model_type.split(".")[-1]]
        )
        if art_class in art_symbols["sklearn_dict"].values():
            init_params = {}
        else:
            init_params = {
                "input_shape": data.X_train.shape[1:],
                "nb_classes": (len(set(data.y_train)) if self.classifier else None),
            }
        if "preprocessing" not in init_params:
            init_params["preprocessing"] = None
        return art_class, init_params

    def get_art_model(self, data: "DataConfig") -> EstimatorLike:
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
        if _is_art_model_instance(self._model):
            return self._model.model
        else:
            return self._model

    @property
    def fitted_estimator(self) -> EstimatorLike | None:
        """Public accessor for the trained runtime estimator payload."""
        return self.model

    @fitted_estimator.setter
    def fitted_estimator(self, value: EstimatorLike | None) -> None:
        """Set the trained runtime estimator payload."""
        self.model = value

    @property
    def test_predictions(self) -> Any:
        """Compatibility alias for test split predictions."""
        return self.predictions

    @test_predictions.setter
    def test_predictions(self, value: Any) -> None:
        """Compatibility alias setter for test split predictions."""
        self.predictions = value

    @property
    def test_probabilities(self) -> Any:
        """Compatibility alias for test split probabilities."""
        return self.probabilities

    @test_probabilities.setter
    def test_probabilities(self, value: Any) -> None:
        """Compatibility alias setter for test split probabilities."""
        self.probabilities = value

    def _apply_defense(self, data: "DataConfig") -> EstimatorLike:
        """Delegate defense application to DefensePipelineConfig."""

        if self.defense is None:
            return self._model
        if self._model is None:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            )

        apply_defense, defense_pipeline, _ = self.compose_defense_behavior(data)
        if apply_defense is None or defense_pipeline is None:
            return self._model
        defended_estimator = apply_defense(self._model)
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
        self.train_model(X, y)
        end_time = time.process_time()
        self.training_time = end_time - start_time
        self.training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def train(self, X: MatrixLike, y: ArrayLike) -> None:
        """Public entry-point for model training. Delegates to _train()."""
        return self._train(X, y)

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
                X_array = np.array(X, dtype=_art_numpy_dtype())
                y_pred = self._model.predict(X_array)
            elif "can't convert" in str(e):
                X_array = np.array(X, dtype=_art_numpy_dtype())
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

        # Try predict_proba or _predict_proba on the wrapped model
        for proba_method in ("predict_proba", "_predict_proba"):
            predict_proba = getattr(self._model, proba_method, None)
            if callable(predict_proba):
                return predict_proba(X)
        # Try underlying estimator if available
        estimator = getattr(self._model, "model", None)
        if estimator is not None:
            for proba_method in ("predict_proba", "_predict_proba"):
                predict_proba = getattr(estimator, proba_method, None)
                if callable(predict_proba):
                    return predict_proba(X)
        raise AttributeError(
            f"Wrapped model of type {type(self._model)} does not have a predict_proba or _predict_proba method, nor does its underlying estimator.",
        )

    def _score(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        *args,
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
        y_proba = kwargs.pop("y_proba", None)
        scores = self.scorer(
            *args,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            mode=mode,
            **kwargs,
        )
        return round_scores(
            scores=scores,
            n_samples=len(y_true),
            logger_obj=logger,
        )

    def compute_score(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *args: Any,
        mode: str = "test",
        **kwargs: Any,
    ) -> dict:
        """Public entry-point for model scoring. Delegates to _score()."""
        return self._score(y_true, y_pred, *args, mode=mode, **kwargs)

    def _canonical_score_mode(self) -> Literal["train", "test", "val"]:
        mode = str(getattr(self, "score_mode", "test") or "test").lower()
        if mode not in {"train", "test", "val"}:
            raise ValueError(
                f"Unsupported ModelConfig score_mode '{self.score_mode}'. Expected one of: train, test, val.",
            )
        return mode

    @staticmethod
    def _mode_score_prefix(mode: str) -> str:
        if mode == "train":
            return "training_"
        if mode == "val":
            return "validation_"
        return ""

    def _mode_runtime_names(self, mode: str) -> dict:
        if mode == "train":
            return {
                "predictions_attr": "training_predictions",
                "probabilities_attr": "training_probabilities",
                "prediction_time_attr": "training_prediction_time",
                "score_time_attr": "training_score_time",
                "n_attr": "training_n",
                "time_key": "training_prediction_time",
                "score_time_key": "training_score_time",
                "n_key": "training_n",
            }
        if mode == "val":
            return {
                "predictions_attr": "val_predictions",
                "probabilities_attr": "val_probabilities",
                "prediction_time_attr": "val_prediction_time",
                "score_time_attr": "val_score_time",
                "n_attr": "val_n",
                "time_key": "validation_prediction_time",
                "score_time_key": "validation_score_time",
                "n_key": "validation_n",
            }
        return {
            "predictions_attr": "predictions",
            "probabilities_attr": "probabilities",
            "prediction_time_attr": "prediction_time",
            "score_time_attr": "prediction_score_time",
            "n_attr": "prediction_n",
            "time_key": "prediction_time",
            "score_time_key": "prediction_score_time",
            "n_key": "prediction_n",
        }

    def _mode_split_data(self, data: DataConfig, mode: str):
        if mode == "train":
            return data.X_train, data.y_train
        if mode == "val":
            if data.X_val is None or data.y_val is None:
                can_resample = (
                    hasattr(data, "_sample")
                    and getattr(data, "_X", None) is not None
                    and getattr(data, "_y", None) is not None
                )
                if can_resample:
                    data.data_sample_time = None
                    for attr in (
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
                    ):
                        setattr(data, attr, None)
                    data._sample()
            if data.X_val is None or data.y_val is None:
                raise ValueError(
                    "ModelConfig score_mode='val' requires data.X_val and data.y_val.",
                )
            return data.X_val, data.y_val
        if data.X_test is None or data.y_test is None:
            raise ValueError(
                "ModelConfig score_mode='test' requires data.X_test and data.y_test.",
            )
        return data.X_test, data.y_test

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

    def load_score_times(self, score_file: str | None) -> dict:
        """Public wrapper for loading persisted score/timing metadata."""
        return self._load_score_file(score_file)

    def load_cached_predictions(
        self,
        train_predictions_file: str | None,
        test_predictions_file: str | None,
        times: dict,
    ) -> dict:
        """Public wrapper for loading cached prediction artifacts."""
        return self._load_all_predictions(
            train_predictions_file,
            test_predictions_file,
            times,
        )

    def train_or_load_model(
        self,
        data: DataConfig,
        model_file: str | None,
        times: dict,
    ) -> dict:
        """Public wrapper for model load-or-train orchestration."""
        return self._load_or_train_model(data, model_file, times)

    def evaluate_model(
        self,
        data: DataConfig,
        times: dict,
    ) -> dict:
        """Public wrapper for evaluation and scoring orchestration."""
        return self._evaluate_and_score(
            data,
            times,
            persist_training_predictions=True,
            persist_test_predictions=True,
            persist_training_probabilities=True,
            persist_test_probabilities=True,
        )

    def persist_runtime_artifacts(
        self,
        test_predictions_file: str | None,
        train_predictions_file: str | None,
        training_probabilities_file: str | None,
        test_probabilities_file: str | None,
        score_file: str | None,
    ) -> None:
        """Persist runtime predictions/probabilities/scores to configured outputs."""
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
        self.score_dict = self.merge_and_persist_scores(self.score_dict, score_file)

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
        times = self.load_score_times(score_file)
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
        times = self.load_cached_predictions(
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
        times = self.train_or_load_model(data, model_file, times)
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
        times = self.evaluate_model(data, times)
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
        self.persist_runtime_artifacts(
            test_predictions_file=test_predictions_file,
            train_predictions_file=train_predictions_file,
            training_probabilities_file=training_probabilities_file,
            test_probabilities_file=test_probabilities_file,
            score_file=score_file,
        )
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
        if times is None:
            times = {}
        if self.defense is not None and self._model is not None:
            apply_defense, defense_pipeline, stage = self.compose_defense_behavior(
                data,
                default_stage="post_fit_pre_predict",
            )
            if (
                stage == "before_predict"
                and apply_defense is not None
                and defense_pipeline is not None
            ):
                self._model = apply_defense(self._model)
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

        if persist_training_predictions and self.training_predictions is None:
            train_predictions = self._predict(data.X_train)
            self.training_predictions = self._decode_predictions_for_persistence(
                train_predictions,
                y_true=data.y_train,
            )
            self.training_n = len(train_predictions)
            times.setdefault("training_n", self.training_n)

        if (
            persist_training_probabilities
            and self.classifier
            and self.training_probabilities is None
        ):
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
                                dtype=_art_numpy_dtype(),
                            )
                            self.training_probabilities = self._model.predict_proba(
                                X_array,
                            )
                        else:
                            raise e
                else:
                    self.training_probabilities = self._predict(data.X_train)
            except ValueError as e:
                logger.warning(
                    "Skipping training probability persistence: %s",
                    e,
                )
                self.training_probabilities = None

        score_mode = self._canonical_score_mode()

        X_mode, y_mode = self._mode_split_data(data, score_mode)
        names = self._mode_runtime_names(score_mode)
        predictions_attr = names["predictions_attr"]
        probabilities_attr = names["probabilities_attr"]
        prediction_time_attr = names["prediction_time_attr"]
        score_time_attr = names["score_time_attr"]
        n_attr = names["n_attr"]

        cached_predictions = getattr(self, predictions_attr, None)
        if cached_predictions is not None:
            mode_predictions = cached_predictions
            times[names["n_key"]] = len(mode_predictions)
        else:
            start_time = time.process_time()
            mode_predictions = self._predict(X_mode)
            end_time = time.process_time()
            prediction_time = end_time - start_time
            setattr(self, prediction_time_attr, prediction_time)
            setattr(self, n_attr, len(mode_predictions))
            times[names["time_key"]] = prediction_time
            times[names["n_key"]] = len(mode_predictions)

            if score_mode == "train":
                should_persist_predictions = persist_training_predictions
                should_persist_probabilities = persist_training_probabilities
            else:
                should_persist_predictions = persist_test_predictions
                should_persist_probabilities = persist_test_probabilities

            if should_persist_predictions:
                setattr(
                    self,
                    predictions_attr,
                    self._decode_predictions_for_persistence(
                        mode_predictions,
                        y_true=y_mode,
                    ),
                )

            if should_persist_probabilities and self.classifier:
                try:
                    if hasattr(self._model, "predict_proba"):
                        try:
                            probabilities = self._model.predict_proba(X_mode)
                        except TypeError as e:
                            if "loop of ufunc does not support argument" in str(
                                e,
                            ) or "can't convert" in str(e):
                                X_array = np.array(
                                    X_mode,
                                    dtype=_art_numpy_dtype(),
                                )
                                probabilities = self._model.predict_proba(X_array)
                            else:
                                raise e
                    else:
                        probabilities = self._predict(X_mode)
                    setattr(self, probabilities_attr, probabilities)
                except ValueError as e:
                    logger.warning(
                        "Skipping %s probability persistence: %s",
                        score_mode,
                        e,
                    )
                    setattr(self, probabilities_attr, None)

        if y_mode is None or mode_predictions is None:
            raise ValueError(
                f"No labels or predictions available for {score_mode} scoring.",
            )

        if self.scorer is not None:
            mode_probabilities = getattr(self, probabilities_attr, None)
            if mode_probabilities is None and self.classifier:
                try:
                    mode_probabilities = self._predict_proba(X_mode)
                except Exception:
                    try:
                        mode_probabilities = probabilities_from_model_outputs(
                            mode_predictions,
                        )
                    except Exception:
                        mode_probabilities = None

            start = time.process_time()
            mode_scores = self._score(
                y_mode,
                mode_predictions,
                y_proba=mode_probabilities,
                mode=score_mode,
                data=data,
                model=self,
            )
            score_time = time.process_time() - start
            setattr(self, score_time_attr, score_time)
            times[names["score_time_key"]] = score_time
            prefix = self._mode_score_prefix(score_mode)
            if prefix:
                mode_scores = {
                    f"{prefix}{key}": value for key, value in mode_scores.items()
                }
                loss_curve_key = f"{prefix}loss_curve"
                if loss_curve_key in mode_scores:
                    del mode_scores[loss_curve_key]
            if self.score_dict is None:
                self.score_dict = {}
            self.score_dict.update(mode_scores)
            logger.info(
                "%s scores computed in %.2f seconds",
                score_mode.title(),
                score_time,
            )

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
