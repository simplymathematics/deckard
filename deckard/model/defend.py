# A BaseConfig class for Configuration of Models using adversarial-robustness-toolbox (ART)
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest

import pandas as pd
import time
import logging
import argparse
import warnings
import importlib
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from typing import Union


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
from .data import data_parser, DataConfig, initialize_data_config, data_call_parser
from .model import ModelConfig, initialize_model_config, model_call_parser, model_parser
from .attack import (
    attack_parser,
)
from .utils import initialize_config, create_parser_from_function

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

supported_attacks = [
    "blackbox_membership_inference",
    "blackbox_evasion",
    "whitebox_evasion",
    "blackbox_attribute_inference",
    "whitebox_attribute_inference",
]
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


@dataclass
class DefenseConfig(ModelConfig):
    model_type: str = "sklearn.linear_model.LogisticRegression"
    classifier: bool = True
    model_params: dict = None
    probability: bool = False
    clip_values: tuple = None
    defense_name: str = "art.defences.postprocessor.HighConfidence"
    defense_params: dict = None

    """"
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
    __call__(data, model_filepath=None, predictions_filepath=None, training_predictions_filepath=None, model_score_filepath=None) -> Union[pd.Series, pd.DataFrame]
        Execute the model workflow: training, prediction, scoring, and model persistence.
    save(model_filepath=None, predictions_filepath=None, training_predictions_filepath=None, model_score_filepath=None)
        Save the model, predictions, and scores to specified file paths.
    _load_or_train_model(data, model_filepath=None, times=None) -> dict
        Load a model from a file or train a new model if not available.
    _load_all_predictions(training_predictions_filepath=None, predictions_filepath=None, times=None) -> dict
        Load predictions from specified file paths and update timing information.
    _load_score_file(model_score_filepath=None) -> dict
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
        return hash(
            (
                self.defense_name,
                frozenset(self.defense_params.items()) if self.defense_params else None,
                hash(self.model),
                hash(self.data),
            ),
        )

    def apply_defense(self, *args, **kwargs) -> BaseEstimator:
        """Apply the specified defense to the model's estimator.

        Parameters
        ----------
        defense : str
            The full class path of the defense to apply (e.g., 'art.defences.preprocessor.JPEGCompression').
        estimator : BaseEstimator, optional
            The estimator to which the defense will be applied. If None, uses self.model._model.
        defense_params : dict, optional
            Parameters to initialize the defense.

        Returns
        -------
        BaseEstimator
            The estimator wrapped with the specified defense.
        """
        supported_defense_types = [
            "detector",
            "preprocessor",
            "postprocessor",
            "trainer",
            "regularizer",
            "transformer",
        ]
        if self._model is None:
            raise ValueError(
                "ModelConfig must have a fitted estimator before applying defense",
            )

        # Dynamically import the defense class with defense_params as kwargs
        module_name, class_name = self.defense_name.rsplit(".", 1)
        defense_type = module_name.split(".")[2]  # e.g., 'preprocessor'
        if len(module_name.split(".")) >= 4:
            defense_subtype = module_name.split(".")[3]  # e.g., 'FeatureSqueezing'
        else:
            defense_subtype = None
        try:
            module = importlib.import_module(module_name)
            defense_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import defense class {self.defense_name}",
            ) from e
        assert (
            defense_type in supported_defense_types
        ), f"Unsupported defense type: {defense_type}. Supported types are: {supported_defense_types}"
        start = time.process_time()
        art_class = (
            classifier_dict[self.model_type.split(".")[-1]]
            if self.classifier
            else regressor_dict[self.model_type.split(".")[-1]]
        )
        if isinstance(self._model, art_class):
            pass
        else:
            match defense_type:  # Note: only one defense can be applied at a time
                case "preprocessor":
                    defense = defense_class(**(self.defense_params or {}))
                    defended_estimator = art_class(
                        self._model,
                        preprocessor=defense,
                        preprocessing_defences=[defense],
                        clip_values=self.clip_values,
                        *args,
                        **kwargs,
                    )
                case "postprocessor":
                    defense = defense_class(**(self.defense_params or {}))
                    defended_estimator = art_class(
                        self._model,
                        postprocessing_defences=[defense],
                        clip_values=self.clip_values,
                        *args,
                        **kwargs,
                    )
                case "detector":
                    match defense_subtype:
                        case "evasion":
                            defense = defense_class(**(self.defense_params or {}))
                            defended_estimator = defense(self._model, *args, **kwargs)
                        case "poison":
                            defense = defense_class(**(self.defense_params or {}))
                            defended_estimator = defense(self._model, *args, **kwargs)
                        case _:
                            raise NotImplementedError(
                                f"Detector subtype '{defense_subtype}' is not implemented yet.",
                            )
                    # Overwrite the _score method to handle each
                case "trainer":
                    defense = defense_class(**(self.defense_params or {}))
                    defended_estimator = defense(self._model, *args, **kwargs)
                case "transformer":
                    defense = defense_class(**(self.defense_params or {}))
                    defended_estimator = defense(
                        self._model,
                        input_transformations=[defense],
                        clip_values=self.clip_values,
                        *args,
                        **kwargs,
                    )
                case "regularizer":
                    raise NotImplementedError(
                        "Regularizer defenses are not implemented yet.",
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


defense_init_parser = argparse.ArgumentParser(
    description="Initialize DefenseConfig from YAML file",
    add_help=False,
    conflict_handler="resolve",
)
defense_init_parser.add_argument(
    "--defense_config_file",
    type=str,
    default=None,
    help="Path to YAML file containing DefenseConfig parameters",
)
defense_init_parser.add_argument(
    "--defense_config_params",
    type=str,
    default=None,
    help="Additional DefenseConfig parameters as a comma-separated list of key=value pairs",
)
defense_call_parser = create_parser_from_function(
    DefenseConfig.__call__,
    exclude=["data"],
)

defense_parser = argparse.ArgumentParser(
    description="DefenseConfig parameters",
    parents=[data_parser, model_parser, defense_init_parser, attack_parser],
    add_help=False,
    conflict_handler="resolve",
)


def initialize_defense_config(args: argparse.Namespace) -> DefenseConfig:
    """
    Initializes and returns a DefenseConfig object using command-line arguments.

    Args:
        args (argparse.Namespace): Namespace object containing parsed command-line arguments.

    Returns:
        DefenseConfig: An initialized DefenseConfig object based on the provided arguments.

    Raises:
        AssertionError: If the initialized config is not an instance of DefenseConfig.
    """
    args = defense_parser.parse_args()
    config_file = args.defense_config_file
    params = args.defense_config_params if args.defense_config_params else ""
    target = "deckard.DefenseConfig"
    config = initialize_config(config_file, params, target)
    assert isinstance(
        config,
        DefenseConfig,
    ), "Initialized config is not an instance of DefenseConfig"
    return config


def defense_main(args: argparse.Namespace):
    assert isinstance(args, argparse.Namespace), "args must be an argparse.Namespace"
    defense_config = initialize_defense_config(args)
    data_config = initialize_data_config(args)
    model_config = initialize_model_config(args)

    data_call_args = data_call_parser.parse_known_args()[0]
    data_config(**vars(data_call_args))

    model_call_args = model_call_parser.parse_known_args()[0]
    model_config(data=data_config, **vars(model_call_args))

    defense_call_args = defense_call_parser.parse_known_args()[0]
    defense_config(data=data_config, **vars(defense_call_args))
    return defense_config.score_dict
