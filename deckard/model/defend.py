# A BaseConfig class for Configuration of Models using adversarial-robustness-toolbox (ART)
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest

import pandas as pd
import time
import logging
import warnings
from sklearn.base import BaseEstimator
from dataclasses import dataclass, field
from typing import Any, Union
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

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
from . import ModelConfig
from ..utils import ConfigBase, resolve_class

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

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


class _DefenseBehaviorMixin:
    """Reusable defense workflow behavior mixed into concrete config dataclasses."""

    # Declared for static analyzers; concrete dataclass provides these fields.
    model_type: Union[str, None]
    classifier: Union[bool, str, None]
    model_params: dict
    probability: bool
    alias: str
    defense_name: Union[str, None]
    defense_params: dict
    _model: Union[BaseEstimator, None]
    score_dict: dict
    _target_: Union[str, None]
    _model_config: Union[ModelConfig, None]

    def _get_model_config(self) -> ModelConfig:
        if getattr(self, "_model_config", None) is None:
            self._model_config = ModelConfig(
                model_type=self.model_type,
                classifier=self.classifier,
                model_params=self.model_params,
                probability=self.probability,
                alias=self.alias,
            )
            self._model_config.defense = None
        assert self._model_config is not None
        return self._model_config

    def __post_init__(self):
        if self.model_type not in [None, "", "None", "null", "Null", "NULL"]:
            model_cfg = self._get_model_config()
            self.classifier = model_cfg.classifier
            self.model_params = model_cfg.model_params
            self._model = model_cfg._model
        elif not hasattr(self, "_model"):
            self._model = None

        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.DefenseConfig"

        # Initialize times, scores, and defended model
        self.defense_training_time = None
        self.defense_application_time = None
        self.defense_prediction_time = None
        self.defense_scoring_time = None
        self.defense_params = self.defense_params or {}
        self._apply_fit = True  # Whether to apply fit during defense application

    def __hash__(self):
        return super().__hash__()

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

    def apply_to(self, estimator: Union[BaseEstimator, None], data) -> BaseEstimator:
        """Apply this defense to a pre-fitted estimator."""
        if estimator is None:
            raise ValueError("estimator must be provided before applying defense")
        self._model = estimator
        model_cfg = getattr(self, "_model_config", None)
        if model_cfg is not None:
            model_cfg._model = estimator
        return self.apply_defense(data)

    def apply_defense(self, data) -> BaseEstimator:
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
        else:
            assert isinstance(
                self._model,
                BaseEstimator,
            ), "ModelConfig's _model must be a scikit-learn BaseEstimator"

        # Dynamically import the defense class with defense_params as kwargs
        defense_type, defense_subtype, defense_class = self.parse_defense_name()
        art_class, init_params = self.get_art_class(data)
        try:
            check_is_fitted(self._model)
        except NotFittedError as e:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            ) from e
        start = time.process_time()
        defense = None
        defended_estimator = None
        match defense_type:  # Note: only one defense can be applied at a time
            case "preprocessor":
                assert defense_class is not None
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = art_class(
                    self.get_model(),
                    preprocessor=defense,
                    preprocessing_defences=[defense],
                    **init_params,
                )
            case "postprocessor":
                assert defense_class is not None
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = art_class(
                    self.get_model(),
                    postprocessing_defences=[defense],
                    **init_params,
                )
            case "detector":
                assert defense_class is not None
                match defense_subtype:
                    case "evasion":
                        defense = defense_class(**(self.defense_params or {}))
                        defended_estimator = defense(self.get_model(), **init_params)
                    case "poison":
                        defense = defense_class(**(self.defense_params or {}))
                        defended_estimator = defense(self.get_model(), **init_params)
                    case _:
                        raise NotImplementedError(
                            f"Detector subtype '{defense_subtype}' is not implemented yet.",
                        )
                # Overwrite the _score method to handle each
            case "trainer":
                assert defense_class is not None
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = defense(self._model, **init_params)
            case "transformer":
                assert defense_class is not None
                defense = defense_class(**(self.defense_params or {}))
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
                defense_params = {**self.defense_params, **init_params}
                defended_estimator = art_class(
                    self.get_model(),
                    **defense_params,
                )
            case "_":
                raise NotImplementedError(
                    f"Defense type '{defense_type}' is not implemented yet.",
                )
        if defended_estimator is None:
            raise RuntimeError("Defense application did not produce an estimator")
        # Some defences can optionally be applied during training or prediction
        end = time.process_time()
        self._apply_fit = getattr(defense, "_apply_fit", True)

        self.defense_application_time = end - start
        model_cfg = getattr(self, "_model_config", None)
        if model_cfg is not None:
            model_cfg._model = defended_estimator
        return defended_estimator

    def parse_defense_name(self):
        if self.defense_name is not None and len(self.defense_name) > 0:
            module_name, class_name = self.defense_name.rsplit(".", 1)
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
                assert self.defense_name is not None
                defense_class = resolve_class(self.defense_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not import defense class {self.defense_name}",
                ) from e
        else:
            defense_class = None
        assert (
            defense_type in supported_defense_types
        ), f"Unsupported defense type: {defense_type}. Supported types are: {supported_defense_types}"

        return defense_type, defense_subtype, defense_class

    def get_art_class(self, data):
        if self.model_type in [None, "", "None", "null", "Null", "NULL"]:
            raise ValueError("model_type must be set before creating an ART defense estimator")
        assert self.model_type is not None
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

    def __call__(
        self,
        data: DataConfig,
        model_file: Union[str, None] = None,
        test_predictions_file: Union[str, None] = None,
        train_predictions_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> dict[str, Any]:
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

        model_cfg = self._get_model_config()
        model_cfg.defense = None
        model_cfg.classifier = self.classifier
        model_cfg.probability = self.probability
        model_cfg.model_params = self.model_params
        if self._model is not None:
            model_cfg._model = self._model

        # Load the score_file if provided
        times = model_cfg._load_score_file(score_file)

        # Load predictions from filepaths and update times
        times = model_cfg._load_all_predictions(
            train_predictions_file,
            test_predictions_file,
            times,
        )

        # Train the model if training data is provided and model is not already trained
        times = model_cfg._load_or_train_model(data, model_file, times)
        self._model = model_cfg._model
        model_cfg._model = self.apply_to(model_cfg._model, data)
        self._model = model_cfg._model

        model_cfg._evaluate_and_score(data, times)
        self.score_dict = model_cfg.score_dict
        self.training_predictions = model_cfg.training_predictions
        self.predictions = model_cfg.predictions
        self.training_time = model_cfg.training_time
        self.prediction_time = model_cfg.prediction_time
        self.training_score_time = model_cfg.training_score_time
        self.prediction_score_time = model_cfg.prediction_score_time
        self.training_n = model_cfg.training_n
        self.prediction_n = model_cfg.prediction_n

        if train_predictions_file is not None:
            model_cfg.save_data(
                self.training_predictions,
                train_predictions_file,
            )
        if test_predictions_file is not None:
            model_cfg.save_data(self.predictions, test_predictions_file)
        if score_file is not None:
            model_cfg.save_scores(self.score_dict, score_file)
        return self.score_dict


@dataclass(eq=False)
class DefenseConfig(_DefenseBehaviorMixin, ConfigBase):
    """Concrete defense config dataclass that uses shared defense behavior mixin."""

    model_type: Union[str, None] = None
    classifier: Union[bool, str, None] = True
    model_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the model."},
    )
    probability: bool = False
    clip_values: tuple | None = field(
        default=None,
        metadata={"help": "Tuple of the form (min, max) to clip input features."},
    )
    defense_name: Union[str, None] = field(
        default=None,
        metadata={"help": "Name of the defense to apply."},
    )
    defense_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the defense."},
    )
    alias: str = field(default_factory=str)
    _model: Union[BaseEstimator, None] = field(default=None, repr=False)
    score_dict: dict = field(default_factory=dict)
    _target_: Union[str, None] = field(default=None, repr=False)
    _model_config: Union[ModelConfig, None] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
