# Standard library imports
import pickle
import time
import logging

from pathlib import Path
import pandas as pd

# Typing imports
from dataclasses import dataclass, field
from typing import Optional, Union, TYPE_CHECKING

# Sklearn and numpy imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
from numpy.exceptions import AxisError

# ART imports
from art.config import ART_NUMPY_DTYPE

from omegaconf import DictConfig, OmegaConf, ListConfig

from ..model import ModelConfig
from ..model.defend import sklearn_dict
from ..utils import ConfigBase, resolve_class
from .pytorch import (
    build_torch_art_model,
    collect_subset_from_dataloader,
    is_dataloader,
    is_tensor,
    is_torch_model,
    tensor_to_numpy,
)

logger = logging.getLogger(__name__)

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
        return {"estimator": self.estimator, "sensitive_features": self._sensitive}

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
        attack_scorer_cls = resolve_class("deckard.score.attack.AttackScorerConfig")
        if self.scorer is None:
            self.scorer = attack_scorer_cls()
        elif isinstance(self.scorer, dict):
            self.scorer = attack_scorer_cls(**self.scorer)

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
        if isinstance(model, RegressorMixin) and not isinstance(model, ClassifierMixin):
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
        if attack_type not in ["evasion", "poisoning", "extraction", "inference"]:
            raise ValueError(f"Unsupported attack type: {attack_type}")
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

        self._validate_attack_task_compatibility(data, model)

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

    def _score_attack_legacy(self, ben_pred_labels, adv_pred_labels, y_test_numeric):
        """Backward-compatible alias retained for older call sites."""
        return self._score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)

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
        if is_tensor(x_subset):
            x_subset = tensor_to_numpy(x_subset, dtype=ART_NUMPY_DTYPE)
            if hasattr(art_model, "_model") and hasattr(art_model._model, "to"):
                art_model._model.to("cpu")
            elif hasattr(art_model, "_model") and hasattr(art_model._model, "_device"):
                art_model._model._device = "cpu"
            else:
                logger.warning("Unable to move model to CPU for prediction.")
        elif isinstance(x_subset, pd.DataFrame):
            x_subset = x_subset.values
        else:
            # x_subset = x_subset.astype(ART_NUMPY_DTYPE)
            pass
        if is_tensor(y_subset):
            y_subset = tensor_to_numpy(y_subset, dtype=ART_NUMPY_DTYPE)
        elif isinstance(y_subset, pd.Series):
            y_subset = y_subset.values
        else:
            assert isinstance(
                y_subset,
                (list, np.ndarray),
            ), f"Expected labels to be a list of np.ndarray. Got {type(y_subset)}"
        ben_preds = art_model.predict(x_subset)
        is_regression = self._is_regression_prediction_output(y_subset, ben_preds)
        if is_regression:
            ben_pred_labels = np.asarray(ben_preds).reshape(-1)
        else:
            ben_pred_labels = np.asarray(ben_preds).argmax(axis=1)
        if is_tensor(ben_pred_labels):
            ben_pred_labels = tensor_to_numpy(ben_pred_labels, dtype=ART_NUMPY_DTYPE)
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
        if is_regression:
            adv_pred_labels = np.asarray(adv_pred).reshape(-1)
        else:
            adv_pred_labels = np.asarray(adv_pred).argmax(axis=1)
        if isinstance(y_subset, pd.Series):
            if is_regression:
                y_test_numeric = y_subset.astype(float).values
            else:
                y_test_numeric = y_subset.astype("category").cat.codes
        elif isinstance(y_subset, pd.DataFrame):
            if is_regression:
                y_test_numeric = y_subset.iloc[:, 0].astype(float).values
            else:
                y_test_numeric = y_subset.iloc[:, 0].astype("category").cat.codes
        elif isinstance(y_subset, np.ndarray):
            y_test_numeric = np.asarray(y_subset).reshape(-1)
        elif is_tensor(y_subset):
            y_test_numeric = tensor_to_numpy(y_subset).reshape(-1)
        else:
            raise TypeError(
                f"Unsupported type for y_subset: {type(y_subset)}",
            )
        score_dict = self._score(
            attack_kind="evasion",
            y_true=y_test_numeric,
            y_pred=adv_pred_labels,
            ben_pred_labels=ben_pred_labels,
            is_classification=not is_regression,
        )
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        self.score_dict = {**self.score_dict, **score_dict}
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
        if isinstance(x_, (pd.Series, np.ndarray, pd.DataFrame)) or is_tensor(x_):
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
        preds = np.array(
            [np.argmax(arr) for arr in art_model.predict(X_test_subset)],
        ).reshape(
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
        score_dict = self._score(
            attack_kind="attribute",
            y_true=target,
            y_pred=inferred,
            targeted_attribute=targeted_attribute_string,
            is_classification=is_classification,
            attack_generation_time=self.attack_time,
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
        y_train_values = (
            data.y_train.copy().values
            if hasattr(data.y_train, "values")
            else np.asarray(data.y_train)
        )
        if y_train_values.ndim == 1:
            # Transform labels into one-hot encoding for attacks expecting 2D y
            y_data = pd.get_dummies(y_train_values).values
        elif y_train_values.ndim == 2 and y_train_values.shape[1] == 1:
            y_data = pd.get_dummies(y_train_values.ravel()).values
        else:
            y_data = y_train_values
        try:
            attack.fit(
                x=data.X_train.copy().values,
                y=y_data,
                test_x=data.X_test.copy().values,
            )
        except AxisError:
            # Fallback: ensure y is strictly 2D one-hot to avoid axis=1 errors
            safe_y_data = pd.get_dummies(np.asarray(y_train_values).reshape(-1)).values
            attack.fit(
                x=data.X_train.copy().values,
                y=safe_y_data,
                test_x=data.X_test.copy().values,
            )
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
        elif is_tensor(inferred):
            inferred = tensor_to_numpy(inferred).astype(int)
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
        score_dict = self._score(
            attack_kind="membership",
            y_true=labels,
            y_pred=inferred,
        )
        self.score_dict = {**self.score_dict, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
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
