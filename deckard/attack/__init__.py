# Standard library imports
import pickle
import time
import logging
import warnings
import importlib
from pathlib import Path
import pandas as pd

# Typing imports
from dataclasses import dataclass, field
from typing import Union

# Sklearn, torch, numpy imports
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from torch import Tensor
import numpy as np

# ART imports
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor
from art.config import ART_NUMPY_DTYPE
from torch import int32 as torchint32

from omegaconf import DictConfig, OmegaConf, ListConfig

from ..model import ModelConfig
from ..model.pytorch import PytorchTemplateClassifier
from ..model.defend import sklearn_dict, sklearn_models
from ..utils import ConfigBase

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

supported_attacks = [
    "blackbox_membership_inference",
    "blackbox_evasion",
    "whitebox_evasion",
    "blackbox_attribute_inference",
    "whitebox_attribute_inference",
]

sklearn_supported_models = list(sklearn_dict.values())
pytorch_supported_models = [PyTorchRegressor, PyTorchClassifier]

supported_models = sklearn_supported_models + pytorch_supported_models


@dataclass
class AttackConfig(ConfigBase):
    """
    AttackConfig
    Configuration and execution class for adversarial attacks on machine learning models.
    This class provides a unified interface for configuring, executing, and scoring various types of adversarial attacks,
    including evasion, poisoning, extraction, and inference attacks. It supports integration with scikit-learn models
    and the Adversarial Robustness Toolbox (ART), and provides detailed logging and timing for attack operations.

    Attributes
    ----------

    attack_time : float, optional
        Time taken to execute the attack.
    attack_prediction_time : float, optional
        Time taken for adversarial prediction.
    attack_score_time : float, optional
        Time taken to score the attack.
    attack : object, optional
        Stores the result of the attack.
    attack_predictions : list, optional
        Stores the predictions made by the attack.
    score_dict : dict, optional
        Stores the computed scores and metrics for the attack.
    alias: str or None
        An optional alias for the attack configuration.

    Methods
    -------
    __hash__()
        Computes a hash value for the object based on its non-private attributes.
    __post_init__()
        Initializes post-construction attributes and sets defaults.
    __call__(data, model, train=False, **kwargs)
    _get_benign_preds(data, art_model, train=False)
        Generates benign predictions and corresponding labels for a subset of data.
    _get_feature_vector_preds(data, targeted_attribute, train=False)
        Extracts a subset of feature vectors, labels, and attributes from the provided data.
    _score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)
    _evade(data, art_model, attack, train=False)
    _infer_attribute(data, art_model, attack, targeted_attribute, train=False)
        Performs an attribute inference attack on a dataset using a specified attack model and model.
    _infer_membership(data, art_model, attack, train=False)
        Performs membership inference attack on the given dataset using the specified attack and model.
    _poison()
    _extract()
    _save(filepath)

    Raises
    ------
    ValueError
        If the attack type, subtype, or model type is unsupported, or if the model is not fitted.
    NotImplementedError
        If the attack type or subtype is not implemented.
    AssertionError
        If the output scores or timing variables are not of the expected types.
    TypeError
        If the attack model's fit method does not accept the expected arguments.

    Examples
    --------
    >>> config = AttackConfig(attack_type="art.attacks.evasion.FastGradientMethod", attack_params={"eps": 0.2})
    >>> results = config(data, model)
    >>> print(results)
    """

    attack_type: str = "art.attacks.evasion.HopSkipJump"
    attack_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the attack."},
    )
    attack_size: int = field(
        default=1000,
        metadata={"help": "Number of samples to use for the attack."},
    )
    targeted_attribute: str = field(
        default_factory=str,
        metadata={"help": "Targeted attribute for inference attacks."},
    )
    alias: Union[str, None] = None

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        """
        Initializes post-construction attributes for the class.

        Sets the internal attack attribute to None. If attack_params is not provided,
        initializes it as an empty dictionary.
        """
        self.attack_predictions = None
        self.attack = None
        self.score_dict = {}
        self.attack_time = None
        self.attack_prediction_time = None
        self.attack_score_time = None
        self._target_ = "deckard.attack.AttackConfig"

    def _initialize_attack(self, model, data):
        """
        Initialize an attack instance for a given model.

        This method determines the appropriate attack class and model wrapper based on the provided model and attack name.
        It validates the attack type and model compatibility, wraps the model if necessary, and instantiates the attack.
        If the attack cannot be initialized with the model (Whitebox), it falls back to a Blackbox attack.

        Parameters
        ----------
        model : object
            The model or configuration object to attack. Can be a fitted scikit-learn model or a ModelConfig instance.

        Returns
        -------
        attack : object
            The initialized attack instance.
        art_model : object
            The ART-wrapped model compatible with the attack.
        attack_type : str
            The type of attack (evasion, poisoning, extraction, inference).
        attack_subtype : str
            The subtype of the attack.

        Raises
        ------
        ValueError
            If the attack type or model type is unsupported, or if the model is not fitted.
        """
        if isinstance(model, ModelConfig):
            model = model._model
        else:
            check_is_fitted(model)
        module = importlib.import_module(self.attack_type.rsplit(".", 1)[0])
        attack_type = self.attack_type.split("attacks.")[-1].split(".")[0]
        attack_subtype = self.attack_type.split("attacks.")[-1].split(".")[1]

        # Validate attack type
        if attack_type not in ["evasion", "poisoning", "extraction", "inference"]:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        attack_class = getattr(module, self.attack_type.split(".")[-1])
        
        if isinstance(model, tuple(supported_models)):
            art_model = model
        elif isinstance(model, BaseEstimator):
            assert isinstance(model, ClassifierMixin), f"Model must be a ClassifierMixin, got {type(model)}"
            model_alias = type(model).__name__
            art_cls = sklearn_dict[model_alias]
            try:
                check_is_fitted(model)
            except NotFittedError as e:
                model.fit(data.X_train, data.y_train)            
            art_model = art_cls(model)
        elif isinstance(model, ModelConfig):
            art_model = model.get_art_model(data)
        elif isinstance(model, (PytorchTemplateClassifier,)):
            art_model = model.get_art_model(data)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        # Convert targeted attribute to index if necessary
        if len(self.targeted_attribute) > 0 and isinstance(
            self.targeted_attribute,
            str,
        ):
            feature_name = self.targeted_attribute
            assert isinstance(
                data.X_train,
                pd.DataFrame,
            ), f"Expected Dataframe got {type(data.X_train)}"
            if not hasattr(self, "target_index"):
                if feature_name not in data.X_train.columns:
                    cols = [col for col in data.X_train.columns if feature_name.split("_")[0] in col]
                    raise ValueError(f"{feature_name} not found. Did you mean one of these: {cols}?")
                self.target_index = data.X_train.columns.get_loc(feature_name)
                self.attack_params["attack_feature"] = self.target_index
                assert (
                    "attack_feature" in self.attack_params
                ), "attack_feature must be specified in attack_params for attribute inference attacks"
        # TODO: Set labels to distinguish targeted attacks from non-targeted attacks
        if "attack_model" in self.attack_params:
            attack_model = self.attack_params["attack_model"]
            if isinstance(attack_model, DictConfig):
                dict_ = OmegaConf.to_container(attack_model)
                cfg = ModelConfig(**dict_)
                cfg(data)
                attack_model = cfg.get_art_model(data)
            elif isinstance(attack_model, ModelConfig):
                attack_model._load_or_train_model(data)
                attack_model = attack_model.get_art_model(data)
            elif isinstance(attack_model, str):
                assert Path(attack_model).exists(), f"attack_model path {attack_model} does not exist"
                with open(attack_model, "rb") as f:
                    attack_model = pickle.load(f)
                    assert isinstance(attack_model, ModelConfig), "Loaded attack_model must be a ModelConfig instance"
                    attack_model = attack_model.get_art_model(data)
            else:
                raise ValueError(f"attack_model must be a ModelConfig instance. Got {type(attack_model)}")
            self.attack_params["attack_model"] = attack_model
        attack = attack_class(art_model, **self.attack_params)
        self._attack_type = attack_type
        self._attack_subtype = attack_subtype
        return attack, art_model, attack_type, attack_subtype

    def __call__(
        self,
        data,
        model,
        attack_file: Union[str, None] = None,
        attack_predictions_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ):
        """
        Executes the specified attack on the provided model using the given data.

        Parameters
        ----------
        data : Any
            The input data to be used for the attack.
        model : object
            The machine learning model to be attacked.
        attack_file : str or None, optional
            File path to save the attack object. If None, the attack object is not saved. Default is None.
        attack_predictions_file : str or None, optional
            File path to save the attack predictions. If None, predictions are not saved. Default is None.
        score_file : str or None, optional
            File path to save the attack scores. If None, scores are not saved. Default is None.
        **kwargs
            Additional keyword arguments for the attack.

        Returns
        -------
        dict
            A dictionary containing attack scores and timing information.

        Raises
        ------
        ValueError
            If the attack type, subtype, or model type is unsupported, or if the model is not fitted.
        NotImplementedError
            If the attack type or subtype is not implemented.
        AssertionError
            If the output scores or timing variables are not of the expected types.
        """
        if attack_file is not None and Path(attack_file).exists():
            self = self.load_object(attack_file)
        if (
            attack_predictions_file is not None
            and Path(attack_predictions_file).exists()
        ):
            self.attack_predictions = self.load_object(attack_predictions_file)
        if score_file is not None and Path(score_file).exists():
            self.score_dict = self.load_scores(score_file)
        attack, art_model, attack_type, attack_subtype = self._initialize_attack(
            model,
            data,
        )
        # Execute the attack based on type and subtype
        if attack_type == "evasion":
            scores = self._evade(data, art_model, attack)
        elif attack_type == "poisoning":
            raise NotImplementedError("Poisoning attack not implemented yet.")
        elif attack_type == "extraction":
            raise NotImplementedError("Extraction attack not implemented yet.")
        elif attack_type == "inference":
            match attack_subtype:
                case "membership_inference":
                    scores = self._infer_membership(
                        data=data,
                        attack=attack,
                    )
                case "attribute_inference":
                    assert (
                        self.targeted_attribute is not None
                    ), "targeted_attribute must be specified for inference attacks"
                    scores = self._infer_attribute(
                        data,
                        art_model,
                        attack,
                        targeted_attribute=self.targeted_attribute,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported inference attack subtype: {attack_subtype}",
                    )
        else:
            raise NotImplementedError(f"Attack type {attack_type} not implemented yet.")
        assert isinstance(scores, dict), "Scores should be a dictionary"
        assert isinstance(
            self.attack_time,
            float,
        ), f"Attack time should be a float, got {type(self.attack_time)}"
        assert isinstance(
            self.attack_prediction_time,
            float,
        ), "Attack prediction time should be a float"
        assert isinstance(
            self.attack_score_time,
            float,
        ), "Attack score time should be a float"
        times = {
            "attack_generation_time": self.attack_time,
            "attack_prediction_time": self.attack_prediction_time,
            "attack_score_time": self.attack_score_time,
        }
        score_dict = {**scores, **times}
        self.score_dict = score_dict

        # Save attack, predictions, and scores if file paths are provided
        if attack_file is not None and not Path(attack_file).exists():
            self.save_object(self, attack_file)
        if attack_predictions_file is not None:
            self.save_object(self.attack_predictions, attack_predictions_file)
        if score_file is not None:
            self.save_scores(self.score_dict, score_file)
        return score_dict

    def _get_benign_preds(self, data, art_model, train=False):
        """
        Generate benign predictions and corresponding labels for a subset of data.

        Depending on the `train` flag, selects either the training or test set, obtains predictions
        from the provided ART model, and returns the predicted labels along with the corresponding
        data subset and true labels.

        Parameters
        ----------
        data : callable
            A function that returns data splits. If `train` is True, should return
            (_, _, X_test, y_test). If `train` is False, should return (X_train, y_train, _, _).
        art_model : object
            An model object with a `predict` method that accepts numpy arrays.
        train : bool, optional
            If True, use the test set; otherwise, use the training set. Defaults to False.

        Returns
        -------
        tuple
            n (int): Number of samples in the subset (self.attack_size).
            ben_pred_labels (np.ndarray): Predicted labels for the benign samples.
            X_subset (pd.DataFrame): Subset of feature data used for prediction.
            y_subset (pd.Series or np.ndarray): True labels for the subset.
        """
        n = self.attack_size
        if train is True:
            X_train = data.X_train
            y_train = data.y_train
            X_test = data.X_test
            y_test = data.y_test
            ben_preds = art_model.predict(X_test)
            ben_pred_labels = ben_preds.argmax(axis=1)
            X_subset = X_test[:n]
            y_subset = y_test[:n]
        else:
            X_train = data.X_train
            y_train = data.y_train
            X_test = data.X_test
            y_test = data.y_test
            ben_preds = art_model.predict(X_train)
            if isinstance(ben_preds, Tensor):
                ben_preds = ben_preds.cpu().numpy().astype(ART_NUMPY_DTYPE)
            ben_pred_labels = ben_preds.argmax(axis=1)
            X_subset = X_train[:n]
            y_subset = y_train[:n]
        if isinstance(y_subset, Tensor):
            y_subset = y_subset.cpu().numpy().astype(ART_NUMPY_DTYPE)
        assert isinstance(
            ben_pred_labels,
            np.ndarray,
        ), f"ben_pred_labels should be np.ndarray, got {type(ben_pred_labels)}"
        assert isinstance(
            X_subset,
            np.ndarray,
        ), f"X_subset should be np.ndarray, got {type(X_subset)}"
        assert isinstance(
            y_subset,
            np.ndarray,
        ), f"y_subset should be np.ndarray, got {type(y_subset)}"
        return n, ben_pred_labels, X_subset, y_subset

    def _get_feature_vector_preds(self, data, targeted_attribute, train=False):
        """
        Extracts a subset of feature vectors, labels, and attributes from the provided data for either training or testing.

        Parameters
        ----------
        data : callable
            A function that returns tuples of (X_train, y_train, a_train, X_test, y_test, a_test) when called with targeted_attribute.
        targeted_attribute : str
            The attribute to target when extracting data.
        train : bool, optional
            If True, extracts from training data; otherwise, extracts from test data. Defaults to False.

        Returns
        -------
        tuple
            n (int): The number of samples to extract (self.attack_size).
            X_subset (pd.DataFrame or pd.Series): Subset of feature vectors.
            y_subset (pd.Series): Subset of labels.
            a_subset (pd.Series): Subset of attributes.

        Raises
        ------
        AssertionError
            If the lengths of the extracted feature vectors, labels, and attributes do not match.
        """
        n = self.attack_size
        if train is False:
            X_train = data.X_train
            y_train = data.y_train
            a_train = data.X_train[targeted_attribute]
            X_test = data.X_test
            y_test = data.y_test
            a_test = data.X_test[targeted_attribute]
            X_train = X_train.drop(columns=[targeted_attribute])
            X_test = X_test.drop(columns=[targeted_attribute])
            assert (
                len(X_test) == len(y_test) == len(a_test)
            ), "X_test, y_test, and a_test must have the same length, but got lengths: {}, {}, {}".format(
                len(X_test),
                len(y_test),
                len(a_test),
            )
            X_subset = X_test[:n]
            y_subset = y_test[:n]
            a_subset = a_test[:n]
        else:

            assert (
                len(X_train) == len(y_train) == len(a_train)
            ), "X_train, y_train, and a_train must have the same length, but got lengths: {}, {}, {}".format(
                len(X_train),
                len(y_train),
                len(a_train),
            )
            X_subset = X_train[:n]
            y_subset = y_train[:n]
            a_subset = a_train[:n]
        return n, X_subset, y_subset, a_subset

    def _score_attack(self, ben_pred_labels, adv_pred_labels, y_test_numeric):
        """
        Computes and logs various performance metrics for adversarial attack predictions.

        Parameters
        ----------
        ben_pred_labels : array-like
            Predicted labels from the benign (original) model.
        adv_pred_labels : array-like
            Predicted labels from the adversarially perturbed model.
        y_test_numeric : array-like
            True labels for the test set.

        Calculates the following metrics for the adversarial predictions:
            - Accuracy
            - Precision
            - Recall
            - F1-score
            - Success rate (agreement between benign and adversarial predictions)

        Returns
        -------
        None
            The function updates the instance's score_dict attribute with the computed metrics.
        """
        start_time = time.process_time()
        adv_accuracy = accuracy_score(y_test_numeric, adv_pred_labels)
        adv_precision = precision_score(
            y_test_numeric,
            adv_pred_labels,
            zero_division=0,
            average="weighted",
        )
        adv_recall = recall_score(
            y_test_numeric,
            adv_pred_labels,
            zero_division=0,
            average="weighted",
        )
        adv_f1 = f1_score(
            y_test_numeric,
            adv_pred_labels,
            zero_division=0,
            average="weighted",
        )
        adv_success = 1 - accuracy_score(ben_pred_labels, adv_pred_labels)
        end_time = time.process_time()
        self.attack_score_time = end_time - start_time
        score_dict = {
            "evasion_accuracy": adv_accuracy,
            "evasion_precision": adv_precision,
            "evasion_recall": adv_recall,
            "evasion_f1-score": adv_f1,
            "evasion_success": adv_success,
        }
        sig_figs = np.floor(np.log10(len(adv_pred_labels))) + 1
        score_dict = {k: round(v, int(sig_figs)) for k, v in score_dict.items()}
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        # Add attack size and timing info
        score_dict["attack_size"] = self.attack_size
        score_dict["attack_score_time"] = self.attack_score_time
        self.score_dict = {**self.score_dict, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")

    def _evade(self, data, art_model, attack):
        """
        Executes an evasion attack on a given dataset using the specified ART model and attack method.

        This method assumes a classification task and generates adversarial examples from a subset of the test data.
        It measures and logs the time taken for both the attack generation and adversarial prediction steps.
        The method then evaluates the attack by comparing benign and adversarial predictions against the true labels,
        and stores the attack results and scores.

        Parameters
        ----------
        data : object
            The dataset containing features and labels.
        art_model : object
            The adversarial robustness toolbox (ART) model used for predictions.
        attack : object
            The ART attack object used to generate adversarial examples.
        train : bool, optional
            If True, uses the training set for evaluation; otherwie, uses the test set. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing the scores and metrics of the attack evaluation.
        """
        start_time = time.process_time()
        n = self.attack_size
        x_subset = data.X_test[:n]
        y_subset = data.y_test[:n]
        if isinstance(x_subset, Tensor):
            x_subset = x_subset.cpu().numpy().astype(ART_NUMPY_DTYPE)
            if hasattr(art_model, "_model") and hasattr(art_model._model, "to"):
                art_model._model.to("cpu")
            elif hasattr(art_model, "_model") and hasattr(art_model._model, "_device"):
                art_model._model._device = "cpu"
            else:
                logger.warning("Unable to move model to CPU for prediction.")
        elif isinstance(x_subset, pd.DataFrame):
            x_subset = x_subset.values
        else:
            x_subset = x_subset.astype(ART_NUMPY_DTYPE)
        if isinstance(y_subset, Tensor):
            y_subset = y_subset.cpu().numpy().astype(ART_NUMPY_DTYPE)
        elif isinstance(y_subset, pd.Series):
            y_subset = y_subset.values
        else:
            assert isinstance(
                y_subset,
                (list, np.ndarray),
            ), f"Expected labels to be a list of np.ndarray. Got {type(y_subset)}"
        # Move model to appropriate device
        ben_preds = art_model.predict(x_subset)
        ben_pred_labels = ben_preds.argmax(axis=1)
        if isinstance(ben_pred_labels, Tensor):
            ben_pred_labels = ben_pred_labels.cpu().numpy().astype(ART_NUMPY_DTYPE)
        if "AdversarialPatch" in str(type(attack)):
            # Special handling for AdversarialPatch attack
            patches = attack.generate(x=x_subset, y=ben_pred_labels)
            # Caclulate the scale of the patch, relative to the input size
            input_shape = x_subset[0].shape[
                1:
            ]  # Exclude batch dimension, channel dimension
            patch_shape = patches[0].shape[
                1:
            ]  # Exclude batch dimension, channel dimension
            # Assume that the patch is square (required by the attack)
            # Calculate the scale based on the larger input_dimension
            scale = max(
                patch_shape[0] / input_shape[0],
                patch_shape[1] / input_shape[1],
            )
            X_test_adv = attack.apply_patch(x_subset, scale=scale)
        else:
            X_test_adv = attack.generate(x=x_subset)
        end_time = time.process_time()
        self.attack_time = end_time - start_time
        logger.info(f"Evasion attack took {self.attack_time} seconds for {n} samples")
        start_time = time.process_time()
        adv_pred = art_model.predict(X_test_adv)
        self.predictions = adv_pred
        self.labels = y_subset
        # adv_pred_labels = adv_pred.argmax(axis=1)
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Adversarial prediction took {self.attack_prediction_time} seconds for {n} samples",
        )
        adv_pred_labels = adv_pred.argmax(axis=1)
        if isinstance(y_subset, pd.Series):
            y_test_numeric = y_subset.astype("category").cat.codes
        elif isinstance(y_subset, pd.DataFrame):
            y_test_numeric = y_subset.iloc[:, 0].astype("category").cat.codes
        elif isinstance(y_subset, np.ndarray):
            y_test_numeric = y_subset
        elif isinstance(y_subset, Tensor):
            y_test_numeric = y_subset
        else:
            raise TypeError(
                f"Unsupported type for y_subset: {type(y_subset)}",
            )
        self._score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)
        self.attack = adv_pred
        return self.score_dict

    def _infer_attribute(
        self,
        data,
        art_model,
        attack,
        targeted_attribute,
    ):
        """
        Perform an attribute inference attack on a dataset using a specified attack model and model.

        This method fits the attack model to the provided data, performs predictions, and evaluates the
        attack's performance in inferring the targeted attribute. It logs timing and scoring information
        throughout the process.

        Parameters
        ----------
        data : object
            An object containing training and test data with attributes `X_train`, `y_train`, `_X_test`, and `_y_test`.
        art_model : object
            The model used for predictions, expected to have a `predict` method.
        attack : object
            The attack model, expected to have `fit` and `infer` methods.
        targeted_attribute : str
            The name of the attribute to be inferred.
        train : bool, optional
            If True, use training data for the attack; otherwise, use test data. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing accuracy, precision, recall, and F1 score for the inferred attribute.

        Raises
        ------
        AssertionError
            If required data attributes are missing or if the test set size does not match `attack_size`.
        TypeError
            If the attack model's `fit` method does not accept the expected arguments.
        """
        assert hasattr(data, "X_train") and hasattr(
            data,
            "y_train",
        ), "DataConfig must have X_train, y_train attributes. Please ensure data() has been called."
        if isinstance(targeted_attribute, str):
            assert targeted_attribute in data.X_test.columns, (
                f"Targeted attribute '{targeted_attribute}' not found in test data columns.",
            )
        else:
            assert isinstance(targeted_attribute, (list, ListConfig)), "targeted attribute must be a string or a list of strings"
            if isinstance(targeted_attribute, ListConfig):
                targeted_attribute = OmegaConf.to_container(targeted_attribute)
            if not isinstance(targeted_attribute, (list, ListConfig)):
                targeted_attribute = [targeted_attribute]
            for attr in targeted_attribute:
                try:
                    assert attr in data.X_test.columns
                except AssertionError:
                    possible_cols = []
                    for col in data.X_test.columns:
                        if str(attr).split("_")[0] in col:
                            possible_cols.append(col)
                    raise ValueError(f"Targeted attribute '{attr}' not found in test data columns.")
        X_test = data.X_test.copy()
        target = X_test[targeted_attribute].copy()
        X_test_subset = X_test.iloc[: self.attack_size, :].copy().values
        target = target[: self.attack_size].values
        
        X_test_subset_without_feature = X_test.drop(
            columns=targeted_attribute,
        ).copy().iloc[: self.attack_size, :].values
        assert (
            len(X_test_subset) == self.attack_size
        ), f"Test set size {len(X_test_subset)} does not match attack_size {self.attack_size}"
        start_time = time.process_time()
        try:
            attack.fit(x=X_test_subset)
        except TypeError as e:
            raise e
        attack_time = time.process_time() - start_time
        logger.info(
            f"Attribute inference attack training took {attack_time} seconds for {self.attack_size} samples",
        )
        self.attack_time = attack_time
        preds = np.array([np.argmax(arr) for arr in art_model.predict(X_test_subset)]).reshape(
            -1,
            1,
        )
        assert isinstance(
            preds,
            np.ndarray,
        ), f"Predictions should be a numpy array, got {type(preds)}"
        unique, counts = np.unique(preds, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"Class {u}: {c} samples")
        possible_values = np.unique(target, axis =0)
        logger.info(
            f"Possible values for targeted attribute '{targeted_attribute}': {possible_values}",
        )
        self.predictions = preds
        self.labels = target
        start_time = time.process_time()
        preds = np.array(preds, dtype=ART_NUMPY_DTYPE)
        X_test_subset_without_feature = np.array(
            X_test_subset_without_feature,
            dtype=ART_NUMPY_DTYPE,
        )
        inferred = attack.infer(
            x=X_test_subset_without_feature,
            pred=preds,
            values=possible_values,
        )
        end_time = time.process_time()
        if isinstance(inferred, list):
            inferred = np.array(inferred)
        elif isinstance(inferred, pd.Series):
            inferred = inferred.values
        elif isinstance(inferred, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported inferred type: {type(inferred)}")
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Attribute inference attack scoring took {self.attack_score_time} seconds for {self.attack_size} samples",
        )
        # Determine if the target is categorical or continuous
        is_classification = not attack._is_continuous
        start_time = time.process_time()

        if is_classification:
            inferred_accuracy = accuracy_score(target, inferred)
            inferred_precision = precision_score(
                target,
                inferred,
                zero_division=0,
                average="weighted",
            )
            inferred_recall = recall_score(
                target,
                inferred,
                zero_division=0,
                average="weighted",
            )
            inferred_f1 = f1_score(
                target,
                inferred,
                zero_division=0,
                average="weighted",
            )
            end_time = time.process_time()
            self.attack_score_time = end_time - start_time
            score_dict = {
                f"inferred_{targeted_attribute}_accuracy": inferred_accuracy,
                f"inferred_{targeted_attribute}_precision": inferred_precision,
                f"inferred_{targeted_attribute}_recall": inferred_recall,
                f"inferred_{targeted_attribute}_f1": inferred_f1,
            }
        else:

            inferred_mse = mean_squared_error(target, inferred)
            inferred_mae = mean_absolute_error(target, inferred)
            inferred_r2 = r2_score(target, inferred)
            end_time = time.process_time()
            self.attack_score_time = end_time - start_time
            score_dict = {
                f"inferred_{targeted_attribute}_mse": inferred_mse,
                f"inferred_{targeted_attribute}_mae": inferred_mae,
                f"inferred_{targeted_attribute}_r2": inferred_r2,
            }
        sig_figs = np.floor(np.log10(len(target))) + 1
        score_dict = {k: round(v, int(sig_figs)) for k, v in score_dict.items()}
        # Add attack size and timing info
        score_dict["attack_size"] = self.attack_size
        score_dict["attack_score_time"] = self.attack_score_time
        score_dict["attack_generation_time"] = self.attack_time
        self.score_dict = {**self.score_dict, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        self.attack = inferred
        return self.score_dict

    def _infer_membership(self, data, attack):
        """
        Perform membership inference attack on the given dataset using the specified attack and model.

        This method fits the attack model using training and test data, obtains benign predictions,
        performs membership inference, and scores the attack's performance.

        Parameters
        ----------
        data : object
            An object containing training and test data attributes (X_train, y_train, _X_test, _y_test).
        art_model : object
            The model/model used for benign predictions.
        attack : object
            The membership inference attack object with fit and infer methods.

        Returns
        -------
        dict
            A dictionary containing the scores and metrics of the membership inference attack.

        Raises
        ------
        Exception
            If the attack fitting process fails.
        ValueError
            If the inferred membership type is unsupported or its length does not match the number of samples.
        """
        start_time = time.process_time()
        try:
            attack.fit(
                x=data.X_train.copy().values,
                y=data.y_train.copy().values,
                test_x=data.X_test.copy().values,
            )
        except Exception as e:
            raise e
        end_time = time.process_time()
        self.attack_time = time.process_time() - start_time

        logger.info(
            f"Membership inference attack training took {self.attack_time} seconds for {self.attack_size} samples",
        )
        big_X = np.vstack((data.X_train.copy().values, data.X_test.copy().values))
        big_y = np.hstack((data.y_train.copy().values, data.y_test.copy().values))
        labels = np.array([1] * len(data.X_train) + [0] * len(data.X_test))
        # Randomly sample self.attack_size indices from big_X, big_y, and labels
        n = self.attack_size
        indices = np.arange(len(big_X))
        indices = np.random.choice(indices, size=n, replace=False)
        big_X = big_X[indices]
        big_y = big_y[indices]
        labels = labels[indices]
        start_time = time.process_time()
        inferred = attack.infer(
            x=big_X,
            y=big_y,
        )
        end_time = time.process_time()
        self.attack_time = end_time - start_time
        logger.info(
            f"Membership inference attack took {self.attack_time} seconds for {self.attack_size} samples",
        )
        if isinstance(inferred, (list, np.ndarray)):
            inferred = np.array(inferred)
        elif isinstance(inferred, pd.Series):
            inferred = inferred
        else:
            raise ValueError(f"Unsupported inferred type: {type(inferred)}")
        assert (
            len(inferred) == n
        ), f"Length of inferred {len(inferred)} does not match number of samples {self.attack_size}"
        start_time = time.process_time()
        if isinstance(inferred, (pd.Series, pd.DataFrame, np.ndarray)):
            inferred = inferred.astype(int)
        elif isinstance(inferred, Tensor):
            inferred = Tensor(inferred.cpu().numpy().astype(int))
        logger.info(
            f"Membership inference prediction took {self.attack_prediction_time} seconds for {self.attack_size} samples",
        )
        self.predictions = inferred
        self.labels = labels
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Membership inference attack prediction took {self.attack_prediction_time} seconds for {self.attack_size} samples",
        )
        start_time = time.process_time()
        if labels.ndim > inferred.ndim:
            inferred = pd.get_dummies(inferred).values
        elif inferred.ndim > labels.ndim:
            inferred = np.argmax(inferred, axis=1)
        else:
            pass
        self.predictions = inferred
        self.labels
        inferred_accuracy = accuracy_score(
            labels,
            inferred,
        )
        inferred_precision = precision_score(
            labels,
            inferred,
            zero_division=0,
            average="weighted",
        )
        inferred_recall = recall_score(
            labels,
            inferred,
            zero_division=0,
            average="weighted",
        )
        inferred_f1 = f1_score(
            labels,
            inferred,
            zero_division=0,
            average="weighted",
        )
        end_time = time.process_time()
        self.attack_score_time = end_time - start_time
        score_dict = {
            "membership_inference_accuracy": inferred_accuracy,
            "membership_inference_precision": inferred_precision,
            "membership_inference_recall": inferred_recall,
            "membership_inference_f1": inferred_f1,
        }
        # Calculate the number of significant figures
        sig_figs = np.floor(np.log10(len(labels))) + 1
        score_dict = {k: round(v, int(sig_figs)) for k, v in score_dict.items()}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        # Add attack size and timing info
        score_dict["attack_size"] = self.attack_size
        score_dict["attack_score_time"] = self.attack_score_time
        self.score_dict = {**self.score_dict, **score_dict}
        logger.info(
            f"Membership inference attack scoring took {self.attack_score_time} seconds for {self.attack_size} samples",
        )
        self.attack = inferred
        return self.score_dict

    def _poison(self):
        """
        Not implemented yet.
        """
        raise NotImplementedError("Poisoning attack not implemented yet.")

    def _extract(self):
        """
        Not implemented yet.
        """
        raise NotImplementedError("Extraction attack not implemented yet.")

    def _save(self, filepath: Union[str, Path]):
        """
        Saves the current object to a pickle file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path where the object should be saved.
            If the provided path does not end with '.pkl', the extension will be appended automatically.

        Side Effects
        -----------
        Serializes the object and writes it to the specified file in binary format.
        Logs an info message indicating the save location.
        """
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"AttackConfig saved to {filepath}")
