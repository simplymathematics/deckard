# A BaseConfig class for Configuration of Models using adversarial-robustness-toolbox (ART)
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest

import pandas as pd
import time
import logging
import warnings
import importlib
from sklearn.base import BaseEstimator
from dataclasses import dataclass, field
from typing import Union
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


@dataclass
class DefenseConfig(ModelConfig):
    model_type: str = "sklearn.linear_model.LogisticRegression"
    classifier: bool = True
    model_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the model."},
    )
    probability: bool = False
    clip_values: tuple = field(
        default=None,
        metadata={"help": "Tuple of the form (min, max) to clip input features."},
    )
    defense_name: str = field(
        default_factory=str,
        metadata={"help": "Name of the defense to apply."},
    )
    defense_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the defense."},
    )
    """
    Overview
    --------
    Configuration class for applying defenses to machine learning models using the Adversarial Robustness Toolbox (ART).
    Inherits from ModelConfig and extends it to include defense mechanisms.

    Parameters
    ----------
    model_type : str
        The type of model to be used (e.g., 'sklearn.linear_model.LogisticRegression').
    classifier : bool
        Indicates whether the model is a classifier (True) or regressor (False).
    model_params : dict
        Parameters to initialize the model.
    probability : bool
        Whether to use probability estimates (for classifiers).
    clip_values : tuple
        Tuple specifying the minimum and maximum values for input features.
    defense_name : str
        The full class path of the defense to apply (e.g., 'art.defences.preprocessor.JPEGCompression').
    defense_params : dict
        Parameters to initialize the defense.

    Attributes
    ----------
    _model : BaseEstimator
        The model's estimator after applying the defense.
    defense_training_time : float
        Time taken to train the model with the defense.
    defense_application_time : float
        Time taken to apply the defense to the model.
    defense_prediction_time : float
        Time taken to make predictions with the defended model.
    defense_scoring_time : float
        Time taken to score the defended model.
    score_dict : dict
        Dictionary to store scores and metrics related to the defense.

    Methods
    -------
    apply_defense(defense, estimator=None, defense_params) -> BaseEstimator
        Apply the specified defense to the model's estimator.
    __post_init__()
        Validate the configuration after initialization.
    __hash__()
        Generate a hash for the configuration instance.
    __call__(data, model_file=None, test_predictions_file=None, train_predictions_file=None, score_file=None) -> Union[pd.Series, pd.DataFrame]
        Execute the model workflow: training, prediction, scoring, and model persistence.
    save(model_file=None, test_predictions_file=None, train_predictions_file=None, score_file=None)
        Save the model, predictions, and scores to specified file paths.
    _load_or_train_model(data, model_file=None, times=None) -> dict
        Load a model from a file or train a new model if not available.
    _load_all_predictions(train_predictions_file=None, test_predictions_file=None, times=None) -> dict
        Load predictions from specified file paths and update timing information.
    _load_score_file(score_file=None) -> dict
        Load scores from a specified file path and update timing information.
    _evaluate_and_score(data, times=None) -> dict
        Evaluate the model on the provided data and update scores and timing information.
    """

    def __post_init__(self):
        super().__post_init__()
        # Initialize times, scores, and defended model
        self.defense_training_time = None
        self.defense_application_time = None
        self.defense_prediction_time = None
        self.defense_scoring_time = None
        self.defense_params = self.defense_params or {}
        self.score_dict = {}
        self._apply_fit = True  # Whether to apply fit during defense application
        if self._target_ is None:
            self._target_ = "deckard.DefenseConfig"

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
        match defense_type:  # Note: only one defense can be applied at a time
            case "preprocessor":
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = art_class(
                    self.get_model(),
                    preprocessor=defense,
                    preprocessing_defences=[defense],
                    **init_params,
                )
            case "postprocessor":
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = art_class(
                    self.get_model(),
                    postprocessing_defences=[defense],
                    **init_params,
                )
            case "detector":
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
                defense = defense_class(**(self.defense_params or {}))
                defended_estimator = defense(self._model, **init_params)
            case "transformer":
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
        # Some defences can optionally be applied during training or prediction
        end = time.process_time()
        self._apply_fit = getattr(defense, "_apply_fit", True)

        self.defense_application_time = end - start
        end = time.process_time()
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
                module = importlib.import_module(module_name)
                defense_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not import defense class {self.defense_name}",
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
        self._model = self.apply_defense(data)
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
