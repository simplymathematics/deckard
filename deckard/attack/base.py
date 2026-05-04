# Standard library imports
import copy
import pickle
import time
import logging

from pathlib import Path
import pandas as pd

# Typing imports
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, TYPE_CHECKING

# Sklearn and numpy imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
from numpy.exceptions import AxisError

# ART imports
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierNeuralNetwork

from omegaconf import DictConfig, OmegaConf, ListConfig

from ..model import ModelConfig
from ..model.defend import sklearn_dict
from ..score.base import (
    DefaultClassifierConfig,
    DefaultRegressorConfig,
    ScorerDictConfig,
)
from ..utils import ConfigBase, resolve_class, resolve_torch_device
from .torch_utils import (
    build_torch_art_model,
    collect_subset_from_dataloader,
    is_dataloader,
    is_tensor,
    is_torch_model,
    tensor_to_numpy,
)

logger = logging.getLogger(__name__)


def _sensitive_slice(sensitive, n):
    """Return the first *n* rows of *sensitive*, or None if unavailable."""
    if sensitive is None:
        return None
    arr = np.asarray(sensitive)
    return arr[:n]


if TYPE_CHECKING:
    from ..score.attack import AttackScorerConfig


class SensitiveFeaturesWrapper(BaseEstimator):
    """Wraps an estimator that requires `sensitive_features` in predict.

    At predict time the stored sensitive features are sliced to match the
    number of rows in ``X``, so adversarial examples (same n rows, different
    feature values) continue to work correctly.
    """

    def __init__(self, estimator, sensitive_features):
        self.estimator = estimator
        self._sensitive = np.asarray(sensitive_features)

    def fit(self, X, y, **kwargs):
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        n = len(X)
        sf = self._sensitive[:n]
        return self.estimator.predict(X, sensitive_features=sf)

    def predict_proba(self, X):
        n = len(X)
        sf = self._sensitive[:n]
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X, sensitive_features=sf)
        # Fall back: convert hard labels to a two-column probability matrix
        labels = self.estimator.predict(X, sensitive_features=sf)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        n_classes = max(len(unique_labels), 2)
        proba = np.zeros((len(labels), n_classes), dtype=float)
        for i, label in enumerate(labels):
            idx = int(label) if label < n_classes else n_classes - 1
            proba[i, idx] = 1.0
        return proba

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "sensitive_features": self._sensitive,
        }

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        if "sensitive_features" in params:
            self._sensitive = np.asarray(params["sensitive_features"])
        return self


supported_attacks = [
    "blackbox_membership_inference",
    "blackbox_evasion",
    "whitebox_evasion",
    "blackbox_attribute_inference",
    "whitebox_attribute_inference",
]

sklearn_supported_models = list(sklearn_dict.values())
supported_models = sklearn_supported_models


@dataclass(eq=False)
class AttackConfig(ConfigBase):
    """
    AttackConfig
    Configuration and execution class for adversarial attacks on machine learning models.
    This class provides a unified interface for configuring, executing, and scoring various types of adversarial attacks,
    including evasion, poisoning, extraction, and inference attacks. It supports integration with scikit-learn models
    and the Adversarial Robustness Toolbox (ART), and provides detailed logging and timing for attack operations.

    Attributes
    ----------
    attack_type : str
        Fully qualified ART attack class path.
    attack_params : dict
        Parameters passed to the ART attack initializer.
    attack_size : int
        Number of samples to include in attack evaluation.
    targeted_attribute : str
        Feature name for attribute inference attacks.
    alias : str or None
        Optional alias for this attack configuration.
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
    predictions : object, optional
        Predicted labels or scores produced by the attack routine.
    labels : object, optional
        Labels aligned with ``predictions`` for scoring.
    target_index : int, optional
        Cached column index for ``targeted_attribute``.
    _attack_type : str, optional
        Parsed attack family (e.g., evasion, inference).
    _attack_subtype : str, optional
        Parsed attack subtype.
    score_dict : dict, optional
        Stores the computed scores and metrics for the attack.
    _target_ : str, optional
        Internal target identifier used by config tooling.

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

    # Configuration fields
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
    scorer: Union["AttackScorerConfig", None] = None
    alias: Union[str, None] = None
    device: Union[str, None] = None
    mode: Literal["test", "val"] = "test"

    # Runtime state fields
    attack_time: Union[float, None] = None
    attack_prediction_time: Union[float, None] = None
    attack_score_time: Union[float, None] = None
    attack: Union[object, None] = None
    attack_predictions: Union[object, None] = None
    predictions: Union[object, None] = None
    labels: Union[object, None] = None
    target_index: Union[int, None] = None
    _attack_type: Union[str, None] = None
    _attack_subtype: Union[str, None] = None
    score_dict: dict = field(default_factory=dict)
    _target_: Union[str, None] = None

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        """
        Initializes post-construction attributes for the class.

        Sets the internal attack attribute to None. If attack_params is not provided,
        initializes it as an empty dictionary.
        """
        self._target_ = "deckard.attack.AttackConfig"
        attack_scorer_cls = resolve_class(
            "deckard.score.attack.AttackScorerConfig",
        )
        if self.scorer is None:
            self.scorer = attack_scorer_cls()
        elif isinstance(self.scorer, dict):
            self.scorer = attack_scorer_cls(**self.scorer)
        self._validate_poisoning_params()
        self.device = str(resolve_torch_device(self.device))

    def _validate_poisoning_params(self):
        """Validate poisoning-specific configuration parameters."""
        attack_type = (self.attack_family or "").lower()
        if attack_type != "poisoning":
            return

        required_keys = ("class_source", "class_target")
        missing_keys = [k for k in required_keys if k not in self.attack_params]
        if missing_keys:
            raise ValueError(
                "Poisoning attacks require attack_params to include "
                f"{required_keys}. Missing: {tuple(missing_keys)}",
            )

        class_source = int(self.attack_params["class_source"])
        class_target = int(self.attack_params["class_target"])
        if class_source == class_target:
            raise ValueError(
                "Poisoning attacks require class_source and class_target to differ.",
            )

    def _parse_attack_path(self) -> tuple[str, str]:
        parts = (self.attack_type or "").split("attacks.")[-1].split(".")
        attack_type = parts[0] if len(parts) > 0 else ""
        attack_subtype = parts[1] if len(parts) > 1 else ""
        return attack_type, attack_subtype

    @property
    def attack_family(self) -> Optional[str]:
        if self._attack_type:
            return self._attack_type
        attack_type, _ = self._parse_attack_path()
        return attack_type or None

    @property
    def attack_subtype(self) -> Optional[str]:
        if self._attack_subtype:
            return self._attack_subtype
        _, attack_subtype = self._parse_attack_path()
        return attack_subtype or None

    @property
    def attack_kind(self) -> Optional[str]:
        attack_type = (self.attack_family or "").lower()
        subtype = (self.attack_subtype or "").lower()

        if attack_type == "evasion":
            return "evasion"
        if attack_type == "inference" and "membership" in subtype:
            return "membership"
        if attack_type == "inference" and "attribute" in subtype:
            return "attribute"
        return None

    @staticmethod
    def _infer_task_is_classification(data, model) -> Optional[bool]:
        """Infer task type from model first, then data config as fallback."""
        if isinstance(model, ModelConfig) and model.classifier is not None:
            return bool(model.classifier)
        if isinstance(model, RegressorMixin) and not isinstance(
            model,
            ClassifierMixin,
        ):
            return False
        if isinstance(model, ClassifierMixin):
            return True
        if hasattr(data, "classifier") and getattr(data, "classifier") is not None:
            return bool(getattr(data, "classifier"))
        return None

    def _validate_attack_task_compatibility(self, data, model):
        """Fail fast for known unsupported task/attack combinations."""
        attack_type = (self.attack_family or "").lower()
        task_is_classification = self._infer_task_is_classification(data, model)
        if attack_type == "evasion" and task_is_classification is False:
            raise ValueError(
                "Evasion attacks are not supported for regression models in the current sklearn+ART integration.",
            )

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
        art_model = None
        if isinstance(model, ModelConfig):
            art_model = model.get_art_model(data)
        elif is_torch_model(model):
            art_model = build_torch_art_model(model=model, data=data)
        else:
            check_is_fitted(model)
        attack_type = self.attack_family or ""
        attack_subtype = self.attack_subtype or ""

        # Validate attack type
        if attack_type not in [
            "evasion",
            "poisoning",
            "extraction",
            "inference",
        ]:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        if attack_type == "poisoning":
            self._validate_poisoning_params()

        attack_class = resolve_class(self.attack_type)
        if art_model is None:
            if isinstance(model, tuple(supported_models)):
                art_model = model
            elif (
                isinstance(model, BaseEstimator)
                and type(model).__name__ in sklearn_dict
            ):
                assert isinstance(
                    model,
                    ClassifierMixin,
                ), f"Model must be a ClassifierMixin, got {type(model)}"
                model_alias = type(model).__name__
                art_cls = sklearn_dict[model_alias]
                try:
                    check_is_fitted(model)
                except NotFittedError as e:
                    logger.debug(e)
                    model.fit(data.X_train, data.y_train)
                art_model = art_cls(model)
            elif isinstance(model, BaseEstimator):
                try:
                    check_is_fitted(model)
                except NotFittedError:
                    model.fit(data.X_train, data.y_train)
                # Wrap models that require sensitive_features in predict (e.g. ThresholdOptimizer)
                import inspect

                predict_sig = inspect.signature(model.predict)
                if "sensitive_features" in predict_sig.parameters:
                    sensitive = getattr(data, "_sensitive_test", None)
                    if sensitive is None:
                        sensitive = getattr(data, "_sensitive_train", None)
                    if sensitive is not None:
                        model = SensitiveFeaturesWrapper(model, sensitive)
                if isinstance(model, RegressorMixin) and not isinstance(
                    model,
                    ClassifierMixin,
                ):
                    art_model = sklearn_dict["sklearn-regressor"](model)
                else:
                    art_model = sklearn_dict["sklearn-classifier"](model)
                if art_model.input_shape is None:
                    art_model._input_shape = (data.X_train.shape[1],)
                nb = getattr(art_model, "nb_classes", None)
                if nb is None or nb <= 0:
                    art_model.nb_classes = len(
                        np.unique(np.asarray(data.y_train).flatten()),
                    )
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
                    cols = [
                        col
                        for col in data.X_train.columns
                        if feature_name.split("_")[0] in col
                    ]
                    raise ValueError(
                        f"{feature_name} not found. Did you mean one of these: {cols}?",
                    )
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
                assert Path(
                    attack_model,
                ).exists(), f"attack_model path {attack_model} does not exist"
                with open(attack_model, "rb") as f:
                    attack_model = pickle.load(f)
                    assert isinstance(
                        attack_model,
                        ModelConfig,
                    ), "Loaded attack_model must be a ModelConfig instance"
                    attack_model = attack_model.get_art_model(data)
            else:
                raise ValueError(
                    f"attack_model must be a ModelConfig instance. Got {type(attack_model)}",
                )
            self.attack_params["attack_model"] = attack_model
        attack_init_params = copy.deepcopy(self.attack_params)
        if attack_type == "poisoning":
            # Internal orchestration fields are not constructor args for ART attacks.
            for key in (
                "class_source",
                "class_target",
                "trigger_index",
                "poison_fit_params",
                "num_workers",
            ):
                attack_init_params.pop(key, None)
        attack = attack_class(art_model, **attack_init_params)
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
            loaded_self = self.load_object(
                attack_file,
                ignore_corrupt=True,
                delete_corrupt=True,
            )
            if loaded_self is not None:
                self = loaded_self
        if (
            attack_predictions_file is not None
            and Path(attack_predictions_file).exists()
        ):
            try:
                self.attack_predictions = self.load_data(attack_predictions_file)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "Failed to load cached attack predictions %s (%s). Recomputing predictions.",
                    attack_predictions_file,
                    exc,
                )
                Path(attack_predictions_file).unlink(missing_ok=True)
        if score_file is not None and Path(score_file).exists():
            self.score_dict = self.load_scores(score_file)

        self._validate_attack_task_compatibility(data, model)

        attack, art_model, attack_type, attack_subtype = self._initialize_attack(
            model,
            data,
        )
        # Execute the attack based on type and subtype
        if attack_type == "evasion":
            scores = self._evade(data, art_model, attack)
        elif attack_type == "poisoning":
            scores = self._poison(
                data=data,
                art_model=art_model,
                attack=attack,
            )
        elif attack_type == "extraction":
            scores = self._extract(
                data=data,
                art_model=art_model,
                attack=attack,
            )
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
            raise NotImplementedError(
                f"Attack type {attack_type} not implemented yet.",
            )
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
            try:
                self.save_object(self, attack_file)
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                logger.warning(
                    "Failed to cache attack object %s (%s). Continuing without cache.",
                    attack_file,
                    exc,
                )
                Path(attack_file).unlink(missing_ok=True)
        if attack_predictions_file is not None:
            self.save_data(self.attack_predictions, attack_predictions_file)
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
            ben_preds = art_model.predict(data.X_test)
            ben_pred_labels = ben_preds.argmax(axis=1)
            n, X_subset, y_subset = self.get_attack_subset(data, test=True)
        else:
            ben_preds = art_model.predict(data.X_train)
            ben_preds = tensor_to_numpy(ben_preds, dtype=ART_NUMPY_DTYPE)
            ben_pred_labels = ben_preds.argmax(axis=1)
            n, X_subset, y_subset = self.get_attack_subset(data, test=False)
        y_subset = tensor_to_numpy(y_subset, dtype=ART_NUMPY_DTYPE)
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
        score_dict = self._score(
            attack_kind="evasion",
            y_true=y_test_numeric,
            y_pred=adv_pred_labels,
            ben_pred_labels=ben_pred_labels,
        )
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        self.score_dict = {**self.score_dict, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")

    def _score(self, attack_kind: str, y_true, y_pred, **kwargs) -> dict:
        """Dispatch attack scoring through the configured AttackScorerConfig."""
        if self.scorer is None:
            raise ValueError(
                "AttackConfig.scorer must be configured with an AttackScorerConfig instance",
            )
        score_dict = self.scorer._score(
            attack_kind=attack_kind,
            y_true=y_true,
            y_pred=y_pred,
            attack_size=self.attack_size,
            **kwargs,
        )
        self.attack_score_time = score_dict.get("attack_score_time")
        return score_dict

    @staticmethod
    def _is_regression_prediction_output(y_true, predictions) -> bool:
        """Infer whether attack predictions represent regression outputs."""
        preds = np.asarray(predictions)
        labels = np.asarray(y_true)
        if preds.ndim > 1 and preds.shape[1] > 1:
            return False
        if preds.ndim > 1 and preds.shape[1] == 1:
            return True
        if preds.dtype.kind == "f" and labels.dtype.kind == "f":
            return True
        return False

    @staticmethod
    def _to_numpy_array(value, dtype=None, flatten: bool = False) -> np.ndarray:
        """Normalize array-like inputs (tensor/pandas/list/ndarray) to numpy arrays."""
        if is_tensor(value):
            arr = tensor_to_numpy(value, dtype=dtype)
        elif isinstance(value, pd.DataFrame):
            arr = value.values
            if dtype is not None:
                arr = arr.astype(dtype)
        elif isinstance(value, pd.Series):
            arr = value.values
            if dtype is not None:
                arr = arr.astype(dtype)
        elif isinstance(value, np.ndarray):
            arr = value.astype(dtype) if dtype is not None else value
        else:
            arr = np.asarray(value, dtype=dtype)

        arr = np.asarray(arr)
        return arr.reshape(-1) if flatten else arr

    def _prepare_features_for_attack(self, value):
        """Prepare feature inputs for attack APIs.

        Subclasses can override this to preserve framework-native tensors.
        """
        if is_tensor(value):
            return tensor_to_numpy(value, dtype=ART_NUMPY_DTYPE)
        if isinstance(value, pd.DataFrame):
            return value.values
        if isinstance(value, pd.Series):
            return value.values
        return value

    def _prepare_labels_for_attack(self, value):
        """Prepare label inputs for attack APIs.

        Subclasses can override this to preserve framework-native tensors.
        """
        if is_tensor(value):
            return tensor_to_numpy(value, dtype=ART_NUMPY_DTYPE)
        if isinstance(value, pd.DataFrame):
            return value.values
        if isinstance(value, pd.Series):
            return value.values
        return value

    @classmethod
    def _prediction_to_labels(cls, predictions, is_regression: bool = False):
        """Convert model/attack prediction outputs into score-ready labels."""
        arr = cls._to_numpy_array(predictions)
        if is_regression:
            return arr.reshape(-1)
        return cls._labels_from_classifier_predictions(arr)

    @classmethod
    def _normalize_ground_truth(cls, y_true, is_regression: bool = False):
        """Normalize y_true into a consistent 1D numpy representation."""
        if isinstance(y_true, pd.Series):
            if is_regression:
                return y_true.astype(float).values
            return y_true.astype("category").cat.codes.values
        if isinstance(y_true, pd.DataFrame):
            series = y_true.iloc[:, 0]
            if is_regression:
                return series.astype(float).values
            return series.astype("category").cat.codes.values
        arr = cls._to_numpy_array(y_true)
        if not is_regression and arr.ndim == 2 and arr.shape[1] > 1:
            return np.argmax(arr, axis=1)
        return arr.reshape(-1)

    @classmethod
    def _target_to_class_labels(cls, y) -> np.ndarray:
        """Convert labels/targets to 1D class-index labels."""
        arr = cls._to_numpy_array(y)
        if arr.ndim == 1:
            return arr.astype(int)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.reshape(-1).astype(int)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return np.argmax(arr, axis=1).astype(int)
        raise ValueError(f"Unsupported target shape for class labels: {arr.shape}")

    @staticmethod
    def _one_hot_encode(labels, nb_classes: int) -> np.ndarray:
        """One-hot encode class-index labels using ART default dtype."""
        labels = np.asarray(labels).reshape(-1).astype(int)
        one_hot = np.zeros((len(labels), int(nb_classes)), dtype=ART_NUMPY_DTYPE)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return one_hot

    @classmethod
    def _normalize_inferred_output(cls, inferred, reference=None):
        """Normalize inferred outputs and align dimensions with reference labels."""
        arr = cls._to_numpy_array(inferred)
        if reference is None:
            return arr
        ref = cls._to_numpy_array(reference)
        if ref.ndim > arr.ndim:
            return pd.get_dummies(arr).values
        if arr.ndim > ref.ndim:
            return np.argmax(arr, axis=1)
        return arr

    @staticmethod
    def _select_extraction_scorer(benign_pred, extracted_pred):
        """Use full classifier metrics when probabilities are available, else label-only metrics."""
        preds = [np.asarray(benign_pred), np.asarray(extracted_pred)]
        has_probabilities = any(pred.ndim == 2 and pred.shape[1] > 1 for pred in preds)
        if has_probabilities:
            return DefaultClassifierConfig(), True
        full_classifier = DefaultClassifierConfig()
        label_only = {
            name: scorer
            for name, scorer in full_classifier.scorers.items()
            if not scorer.needs_proba
        }
        return ScorerDictConfig(scorers=label_only), False

    def _score_attack_legacy(
        self,
        ben_pred_labels,
        adv_pred_labels,
        y_test_numeric,
    ):
        """Backward-compatible alias retained for older call sites."""
        return self._score_attack(
            ben_pred_labels,
            adv_pred_labels,
            y_test_numeric,
        )

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
        n, x_subset, y_subset = self.get_attack_subset(data)
        x_subset = self._prepare_features_for_attack(x_subset)
        y_subset = self._prepare_labels_for_attack(y_subset)
        if not isinstance(y_subset, (list, np.ndarray)) and not is_tensor(y_subset):
            raise TypeError(
                f"Expected labels to be a list, numpy array, or tensor. Got {type(y_subset)}",
            )
        ben_preds = art_model.predict(x_subset)
        is_regression = self._is_regression_prediction_output(
            y_subset,
            ben_preds,
        )
        ben_pred_labels = self._prediction_to_labels(
            ben_preds,
            is_regression=is_regression,
        )
        if is_tensor(ben_pred_labels):
            ben_pred_labels = tensor_to_numpy(
                ben_pred_labels,
                dtype=ART_NUMPY_DTYPE,
            )
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
        logger.info(
            f"Evasion attack took {self.attack_time} seconds for {n} samples",
        )
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
        adv_pred_labels = self._prediction_to_labels(
            adv_pred,
            is_regression=is_regression,
        )
        y_test_numeric = self._normalize_ground_truth(
            y_subset,
            is_regression=is_regression,
        )
        if is_regression:
            benign_scorer = DefaultRegressorConfig()
            benign_scores = benign_scorer(
                y_true=y_test_numeric,
                y_pred=ben_pred_labels,
                mode=None,
            )
        else:
            full_classifier = DefaultClassifierConfig()
            label_only = {
                name: scorer
                for name, scorer in full_classifier.scorers.items()
                if not scorer.needs_proba
            }
            benign_scorer = ScorerDictConfig(scorers=label_only)
            benign_scores = benign_scorer(
                y_true=y_test_numeric,
                y_pred=ben_pred_labels,
                mode=None,
            )

        score_dict = self._score(
            attack_kind="evasion",
            y_true=y_test_numeric,
            y_pred=adv_pred_labels,
            ben_pred_labels=ben_pred_labels,
            is_classification=not is_regression,
            sensitive_features=_sensitive_slice(
                getattr(data, "_sensitive_test", None),
                n,
            ),
        )
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        prefixed_benign = {f"benign_{k}": v for k, v in benign_scores.items()}
        self.score_dict = {**self.score_dict, **prefixed_benign, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        self.attack = adv_pred
        return self.score_dict

    def get_attack_subset(self, data, test=True):
        n = self.attack_size
        if test is True:
            x_ = data.X_test
            y_ = data.y_test
        else:
            x_ = data.X_train
            y_ = data.y_train
        if isinstance(x_, (pd.Series, np.ndarray, pd.DataFrame)) or is_tensor(
            x_,
        ):
            x_subset = x_[:n]
            y_subset = y_[:n]
        elif is_dataloader(x_):
            x_subset, y_subset = collect_subset_from_dataloader(x_, n=n)
        else:
            raise ValueError(
                f"Expected data.X_test to be a pd.Series, np.ndarray, or a torch Tensor or torch DataLoader. Got: {type(data.X_test)}",
            )
        return n, x_subset, y_subset

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
        targeted_attribute_string = str(targeted_attribute)
        if isinstance(targeted_attribute, str):
            assert targeted_attribute in data.X_test.columns, (
                f"Targeted attribute '{targeted_attribute}' not found in test data columns.",
            )
        else:
            assert isinstance(
                targeted_attribute,
                (list, ListConfig),
            ), "targeted attribute must be a string or a list of strings"
            if isinstance(targeted_attribute, ListConfig):
                targeted_attribute = OmegaConf.to_container(targeted_attribute)
            if not isinstance(targeted_attribute, list):
                targeted_attribute = [targeted_attribute]
            targeted_attribute_string = ""
            for attr in targeted_attribute:
                try:
                    assert attr in data.X_test.columns
                    if len(targeted_attribute_string) > 0:
                        targeted_attribute_string += f"-{attr}"
                    else:
                        targeted_attribute_string = f"{attr}"
                except AssertionError:
                    possible_cols = []
                    for col in data.X_test.columns:
                        if str(attr).split("_")[0] in col:
                            possible_cols.append(col)
                    raise ValueError(
                        f"Targeted attribute '{attr}' not found in test data columns.",
                    )
        X_test = data.X_test.copy()
        target = X_test[targeted_attribute].copy()
        X_test_subset = X_test.iloc[: self.attack_size, :].copy().values
        target = target[: self.attack_size].values

        X_test_subset_without_feature = (
            X_test.drop(
                columns=targeted_attribute,
            )
            .copy()
            .iloc[: self.attack_size, :]
            .values
        )
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
        preds = self._prediction_to_labels(
            art_model.predict(X_test_subset),
            is_regression=False,
        ).reshape(-1, 1)
        assert isinstance(
            preds,
            np.ndarray,
        ), f"Predictions should be a numpy array, got {type(preds)}"
        unique, counts = np.unique(preds, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"Class {u}: {c} samples")
        possible_values = np.unique(target, axis=0)
        if isinstance(possible_values, np.ndarray):
            possible_values = possible_values.tolist()
        if (
            isinstance(possible_values, list)
            and len(possible_values) > 0
            and isinstance(possible_values[0], list)
            and len(possible_values[0]) == 1
        ):
            possible_values = [v[0] for v in possible_values]
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
        # Determine if the target is categorical or continuous
        is_classification = not attack._is_continuous
        inferred = self._normalize_inferred_output(inferred)
        if is_classification:
            inferred = self._prediction_to_labels(inferred, is_regression=False)
            target = self._normalize_ground_truth(target, is_regression=False)
        else:
            inferred = self._prediction_to_labels(inferred, is_regression=True)
            target = self._normalize_ground_truth(target, is_regression=True)
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Attribute inference attack scoring took {self.attack_score_time} seconds for {self.attack_size} samples",
        )
        sensitive_attribute = _sensitive_slice(
            getattr(data, "_sensitive_train", None),
            self.attack_size,
        )
        score_dict = self._score(
            attack_kind="attribute",
            y_true=target,
            y_pred=inferred,
            targeted_attribute=targeted_attribute_string,
            is_classification=is_classification,
            attack_generation_time=self.attack_time,
            sensitive_features=sensitive_attribute,
        )
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
        y_train_raw = self._prepare_labels_for_attack(getattr(data, "y_train"))
        y_train_values = self._to_numpy_array(y_train_raw)
        if y_train_values.ndim == 1:
            # Transform labels into one-hot encoding for attacks expecting 2D y
            y_data = pd.get_dummies(y_train_values).values
        elif y_train_values.ndim == 2 and y_train_values.shape[1] == 1:
            y_data = pd.get_dummies(y_train_values.ravel()).values
        else:
            y_data = y_train_values
        try:
            attack.fit(
                x=self._prepare_features_for_attack(getattr(data, "X_train")),
                y=y_data,
                test_x=self._prepare_features_for_attack(getattr(data, "X_test")),
            )
        except AxisError:
            # Fallback: ensure y is strictly 2D one-hot to avoid axis=1 errors
            safe_y_data = pd.get_dummies(
                np.asarray(y_train_values).reshape(-1),
            ).values
            attack.fit(
                x=self._prepare_features_for_attack(getattr(data, "X_train")),
                y=safe_y_data,
                test_x=self._prepare_features_for_attack(getattr(data, "X_test")),
            )
        end_time = time.process_time()
        self.attack_time = time.process_time() - start_time

        logger.info(
            f"Membership inference attack training took {self.attack_time} seconds for {self.attack_size} samples",
        )
        x_train = self._prepare_features_for_attack(getattr(data, "X_train"))
        x_test = self._prepare_features_for_attack(getattr(data, "X_test"))
        big_X = np.vstack(
            (
                self._to_numpy_array(x_train),
                self._to_numpy_array(x_test),
            ),
        )
        big_y = np.hstack(
            (
                self._to_numpy_array(
                    self._prepare_labels_for_attack(getattr(data, "y_train")),
                ),
                self._to_numpy_array(
                    self._prepare_labels_for_attack(getattr(data, "y_test")),
                ),
            ),
        )
        labels = np.array([1] * len(data.X_train) + [0] * len(data.X_test))
        # Build combined sensitive-feature array aligned with big_X (train then test).
        sensitive_train = getattr(data, "_sensitive_train", None)
        sensitive_test = getattr(data, "_sensitive_test", None)
        if sensitive_train is not None and sensitive_test is not None:
            big_sensitive = np.concatenate(
                [np.asarray(sensitive_train), np.asarray(sensitive_test)],
            )
        else:
            big_sensitive = None
        # Randomly sample self.attack_size indices from big_X, big_y, and labels
        n = self.attack_size
        indices = np.arange(len(big_X))
        indices = np.random.choice(indices, size=n, replace=False)
        big_X = big_X[indices]
        big_y = big_y[indices]
        labels = labels[indices]
        sensitive_membership = (
            big_sensitive[indices] if big_sensitive is not None else None
        )
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
        inferred = self._normalize_inferred_output(inferred)
        assert (
            len(inferred) == n
        ), f"Length of inferred {len(inferred)} does not match number of samples {self.attack_size}"
        start_time = time.process_time()
        inferred = self._normalize_inferred_output(inferred, reference=labels)
        inferred = self._prediction_to_labels(inferred, is_regression=False)
        labels = self._normalize_ground_truth(labels, is_regression=False)
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
        self.predictions = inferred
        self.labels
        score_dict = self._score(
            attack_kind="membership",
            y_true=labels,
            y_pred=inferred,
            sensitive_features=sensitive_membership,
        )
        self.score_dict = {**self.score_dict, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        logger.info(
            f"Membership inference attack scoring took {self.attack_score_time} seconds for {self.attack_size} samples",
        )
        self.attack = inferred
        return self.score_dict

    def _poison(self, data, art_model, attack):
        """Execute a poisoning attack and score benign vs poisoned model accuracy."""
        class_source = int(self.attack_params["class_source"])
        class_target = int(self.attack_params["class_target"])
        trigger_index = int(self.attack_params.get("trigger_index", 0))
        poison_fit_params = self.attack_params.get("poison_fit_params", {})

        # ART GradientMatching on macOS can fail with spawned DataLoader workers;
        # force CPU and single-worker loaders for deterministic smoke/integration runs.
        attack_name = type(attack).__name__.lower()
        if "gradientmatchingattack" in attack_name:
            try:
                import torch

                art_device = getattr(art_model, "_device", None)
                if getattr(art_device, "type", None) == "mps":
                    if hasattr(art_model, "_model") and hasattr(
                        art_model._model,
                        "to",
                    ):
                        art_model._model = art_model._model.to(torch.device("cpu"))
                    if hasattr(art_model, "_device"):
                        art_model._device = torch.device("cpu")
            except ImportError:
                pass

        x_train = self._to_numpy_array(
            self._prepare_features_for_attack(getattr(data, "X_train")),
            dtype=ART_NUMPY_DTYPE,
        )
        y_train_raw = self._to_numpy_array(
            self._prepare_labels_for_attack(getattr(data, "y_train")),
        )

        mode_used, x_eval_raw, y_eval_raw = self._resolve_eval_split(data)
        x_eval = self._to_numpy_array(
            self._prepare_features_for_attack(x_eval_raw),
            dtype=ART_NUMPY_DTYPE,
        )
        y_eval = self._normalize_ground_truth(y_eval_raw, is_regression=False)
        y_eval_class = self._target_to_class_labels(y_eval_raw)

        source_indices = np.where(y_eval_class == class_source)[0]
        if len(source_indices) == 0:
            fallback_source = int(y_eval_class[0])
            logger.warning(
                "No samples for class_source=%s on %s split; using first available class=%s",
                class_source,
                mode_used,
                fallback_source,
            )
            class_source = fallback_source
            source_indices = np.where(y_eval_class == class_source)[0]

        trigger_pos = trigger_index if trigger_index < len(source_indices) else 0
        trigger_idx = int(source_indices[trigger_pos])
        x_trigger = x_eval[trigger_idx : trigger_idx + 1]

        nb_classes = getattr(art_model, "nb_classes", None)
        if nb_classes is None or int(nb_classes) <= 0:
            nb_classes = int(np.max(y_eval_class)) + 1
        y_trigger = self._one_hot_encode([class_target], nb_classes=int(nb_classes))

        if y_train_raw.ndim == 1 or (
            y_train_raw.ndim == 2 and y_train_raw.shape[1] == 1
        ):
            y_train_for_poison = self._one_hot_encode(
                y_train_raw.reshape(-1),
                nb_classes,
            )
        else:
            y_train_for_poison = y_train_raw

        start_time = time.process_time()
        patched_torch_utils_data = None
        patched_dataloader = None
        try:
            if "gradientmatchingattack" in attack_name:
                import torch.utils.data as torch_utils_data

                patched_torch_utils_data = torch_utils_data
                patched_dataloader = torch_utils_data.DataLoader

                def _single_worker_loader(*args, **kwargs):
                    kwargs["num_workers"] = 0
                    return patched_dataloader(*args, **kwargs)

                torch_utils_data.DataLoader = _single_worker_loader

            x_poison, y_poison = attack.poison(
                x_trigger,
                y_trigger,
                x_train,
                y_train_for_poison,
            )
        finally:
            if patched_torch_utils_data is not None and patched_dataloader is not None:
                patched_torch_utils_data.DataLoader = patched_dataloader
        self.attack_time = time.process_time() - start_time
        logger.info(
            f"Poison generation took {self.attack_time} seconds for {len(x_poison)} training samples",
        )

        start_time = time.process_time()
        benign_pred = art_model.predict(x_eval)
        art_model.fit(x_poison, y_poison, **poison_fit_params)
        poisoned_pred = art_model.predict(x_eval)
        self.attack_prediction_time = time.process_time() - start_time
        logger.info(
            f"Poisoned model fit + prediction took {self.attack_prediction_time} seconds on {mode_used} split",
        )

        benign_labels = self._prediction_to_labels(benign_pred, is_regression=False)
        poisoned_labels = self._prediction_to_labels(
            poisoned_pred,
            is_regression=False,
        )

        start_time = time.process_time()
        full_classifier = DefaultClassifierConfig()
        label_only = {
            name: scorer
            for name, scorer in full_classifier.scorers.items()
            if not scorer.needs_proba
        }
        classifier_scorer = ScorerDictConfig(scorers=label_only)
        benign_kwargs = {
            "y_true": y_eval,
            "y_pred": benign_labels,
            "mode": None,
        }
        poisoned_kwargs = {
            "y_true": y_eval,
            "y_pred": poisoned_labels,
            "mode": None,
        }

        benign_scores = classifier_scorer(
            y_true=y_eval,
            **{k: v for k, v in benign_kwargs.items() if k != "y_true"},
        )
        poisoned_scores = classifier_scorer(
            y_true=y_eval,
            **{k: v for k, v in poisoned_kwargs.items() if k != "y_true"},
        )

        trigger_pred = art_model.predict(x_trigger)
        trigger_label = int(self._labels_from_classifier_predictions(trigger_pred)[0])
        self.attack_score_time = time.process_time() - start_time

        self.predictions = poisoned_labels
        self.labels = y_eval
        self.attack_predictions = poisoned_pred
        self.attack = art_model
        self.score_dict = {
            **self.score_dict,
            **{f"benign_{k}": v for k, v in benign_scores.items()},
            **{f"poisoned_{k}": v for k, v in poisoned_scores.items()},
            "poison_attack_target_class": class_target,
            "poison_attack_source_class": class_source,
            "poison_trigger_index": trigger_idx,
            "poison_trigger_predicted_class": trigger_label,
            "poison_trigger_success": int(trigger_label == class_target),
            "attack_size": len(x_poison),
            "poison_mode": mode_used,
        }
        return self.score_dict

    @staticmethod
    def _is_nn_art_classifier(art_model) -> bool:
        """Return True when the ART estimator appears to be a neural classifier."""
        if isinstance(art_model, ClassifierNeuralNetwork):
            return True
        class_name = type(art_model).__name__.lower()
        if any(
            token in class_name
            for token in (
                "pytorchclassifier",
                "kerasclassifier",
                "tensorflowv2classifier",
            )
        ):
            return True
        model_obj = getattr(art_model, "_model", None)
        if model_obj is None:
            model_obj = getattr(art_model, "model", None)
        return bool(model_obj is not None and is_torch_model(model_obj))

    @staticmethod
    def _labels_from_classifier_predictions(predictions):
        preds = np.asarray(predictions)
        if preds.ndim == 1:
            # Binary classifiers can expose scores/probabilities as a single vector.
            if preds.dtype.kind == "f":
                return (preds >= 0.5).astype(int)
            return preds.astype(int)
        if preds.ndim == 2:
            if preds.shape[1] == 1:
                return (preds.reshape(-1) >= 0.5).astype(int)
            return np.argmax(preds, axis=1)
        raise ValueError(
            f"Unsupported prediction shape for classifier output: {preds.shape}",
        )

    def _resolve_eval_split(self, data):
        requested_mode = str(self.mode or "test").lower()
        if requested_mode not in {"test", "val"}:
            raise ValueError(
                f"Unsupported attack mode '{self.mode}'. Expected 'test' or 'val'.",
            )

        if requested_mode == "val":
            X_val = getattr(data, "X_val", None)
            y_val = getattr(data, "y_val", None)
            if X_val is not None and y_val is not None:
                return "val", X_val, y_val
            logger.warning(
                "Attack mode='val' requested but validation split is unavailable; falling back to test split.",
            )

        X_test = getattr(data, "X_test", None)
        y_test = getattr(data, "y_test", None)
        if X_test is None or y_test is None:
            raise ValueError(
                "Extraction attacks require test features/labels (or val when mode='val').",
            )
        return "test", X_test, y_test

    def _extract(self, data, art_model, attack):
        """Execute a model extraction attack and score victim vs extracted classifiers."""
        task_is_classification = self._infer_task_is_classification(
            data,
            model=art_model,
        )
        if task_is_classification is False:
            raise ValueError(
                "Extraction attacks are only supported for classification tasks.",
            )
        if not self._is_nn_art_classifier(art_model):
            raise ValueError(
                "Extraction attacks currently require a neural-network ART classifier (e.g., PyTorchClassifier).",
            )

        n, x_query, _ = self.get_attack_subset(data, test=False)
        x_query = self._prepare_features_for_attack(x_query)

        mode_used, x_eval, y_eval = self._resolve_eval_split(data)
        x_eval = self._prepare_features_for_attack(x_eval)
        y_eval = self._normalize_ground_truth(y_eval, is_regression=False)

        thieved_classifier = copy.deepcopy(art_model)
        thieved_model = getattr(thieved_classifier, "_model", None)
        if thieved_model is not None and hasattr(thieved_model, "apply"):

            def _reset_module_weights(module):
                reset_fn = getattr(module, "reset_parameters", None)
                if callable(reset_fn):
                    reset_fn()

            thieved_model.apply(_reset_module_weights)

        start_time = time.process_time()
        extracted_classifier = attack.extract(
            x=x_query,
            thieved_classifier=thieved_classifier,
        )
        self.attack_time = time.process_time() - start_time
        logger.info(
            f"Extraction attack training took {self.attack_time} seconds for {n} query samples",
        )

        start_time = time.process_time()
        benign_pred = art_model.predict(x_eval)
        extracted_pred = extracted_classifier.predict(x_eval)
        self.attack_prediction_time = time.process_time() - start_time
        logger.info(
            f"Extraction prediction took {self.attack_prediction_time} seconds on {mode_used} split",
        )

        benign_labels = self._labels_from_classifier_predictions(benign_pred)
        extracted_labels = self._labels_from_classifier_predictions(extracted_pred)

        start_time = time.process_time()
        classification_scorer, use_proba_metrics = self._select_extraction_scorer(
            benign_pred,
            extracted_pred,
        )
        benign_kwargs = {
            "y_true": y_eval,
            "y_pred": benign_labels,
            "mode": None,
        }
        extracted_kwargs = {
            "y_true": y_eval,
            "y_pred": extracted_labels,
            "mode": None,
        }
        if use_proba_metrics:
            benign_kwargs["y_proba"] = self._to_numpy_array(benign_pred)
            extracted_kwargs["y_proba"] = self._to_numpy_array(extracted_pred)

        benign_scores = classification_scorer(
            y_true=y_eval,
            **{k: v for k, v in benign_kwargs.items() if k != "y_true"},
        )
        extracted_scores = classification_scorer(
            y_true=y_eval,
            **{k: v for k, v in extracted_kwargs.items() if k != "y_true"},
        )
        self.attack_score_time = time.process_time() - start_time

        prefixed_benign = {f"benign_{k}": v for k, v in benign_scores.items()}
        prefixed_extracted = {f"extracted_{k}": v for k, v in extracted_scores.items()}
        self.predictions = extracted_labels
        self.labels = y_eval
        self.attack_predictions = extracted_pred
        self.attack = extracted_classifier
        self.score_dict = {
            **self.score_dict,
            **prefixed_benign,
            **prefixed_extracted,
            "attack_size": n,
            "extraction_mode": mode_used,
        }
        return self.score_dict

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
