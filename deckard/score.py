from dataclasses import dataclass
import logging
from typing import Literal, Dict
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
)
from .data import DataConfig
from .model import ModelConfig
from .attack import AttackConfig
from .utils import ConfigBase

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """
    ScorerConfig is a data class that encapsulates a scoring function along with its name and runtime parameters.
     It ensures the provided scoring function is callable and initializes a scorer using the specified function and parameters.

     Attributes
     ----------
         score_name (str): The name of the scoring function.
         score_function (callable): The scoring function to be used.
         score_params (dict, optional): Runtime parameters for the scoring function. Defaults to an empty dictionary.

     Methods
     -------
         __post_init__(): Validates the scoring function and initializes the scorer using make_scorer.
     class ScorerConfig(ConfigBase):
    """

    score_name: str
    score_function: callable
    score_params: dict = None
    greater_is_better: bool = True
    needs_proba: bool = False

    def __post_init__(self):
        """
        Validates the scoring function and initializes the scorer using make_scorer.
        """
        if self.score_params is None:
            self.score_params = {}
        assert callable(self.score_function), "score_function must be callable"
        # Create a scorer using the provided function and parameters

    def __call__(self, y_true, y_pred, swap=False, **kwargs):
        """
        Computes the score between true and predicted values using the specified score function.

        Args
        -----
            y_true: The ground truth values.
            y_pred: The predicted values.
            swap (bool, optional): If True, swaps the order of y_true and y_pred when passing to the score function.
            **kwargs: Additional keyword arguments to pass to the score function.

        Returns
        -------
            The result of the score function applied to the provided inputs.
        """
        if swap:
            y_1 = y_pred
            y_2 = y_true
        else:
            y_1 = y_true
            y_2 = y_pred
        all_params = {**self.score_params, **kwargs}
        return self.score_function(y_1, y_2, **all_params)


class ScorerDictConfig(ConfigBase):
    """
    ----
    ScorerDictConfig manages a dictionary of ScorerConfig instances for batch scoring.

    This class provides a container for multiple scoring functions, allowing you to apply
    all configured scorers to a set of predictions and ground truth labels in a single call.
    It ensures type safety by validating that all values in the dictionary are ScorerConfig instances.

    Parameters
    ----------
    scorers : dict
        A dictionary mapping string keys to ScorerConfig instances.

    Methods
    -------
    __post_init__():
        Validates that all values in the scorers dictionary are instances of ScorerConfig and
        calls their __post_init__ methods.

    __iter__():
        Returns an iterator over the (key, ScorerConfig) pairs in the scorers dictionary.

    __getitem__(key):
        Retrieves the ScorerConfig instance associated with the given key.

    get_callables():
        Returns a dictionary mapping keys to their corresponding ScorerConfig instances.

    __call__(y_true, y_pred, **kwargs) -> Dict[str, float]:
        Applies each scorer to the provided true and predicted labels, returning a dictionary
        of scores keyed by scorer name.

    ----
    """

    def __init__(self, scorers: dict):
        """
        ----
        Initializes the instance with a dictionary of scoring functions.

        Parameters
        ----------
        scorers : dict
            A dictionary mapping metric names to scoring functions or callables.

        ----
        """
        self._scorers = scorers
        self.__post_init__()

    def __post_init__(self):
        """
        ----
        Post-initialization hook for the class.

        Iterates over all items in the `_scorers` dictionary, ensuring each value is an instance of `ScorerConfig`.
        Calls the `__post_init__` method of each `ScorerConfig` to perform any additional setup required.

        Raises
        ------
        AssertionError
            If any value in `_scorers` is not an instance of `ScorerConfig`.
        """
        for key, value in self._scorers.items():
            assert isinstance(
                value,
                ScorerConfig,
            ), f"Value for key '{key}' must be an instance of ScorerConfig"
            value.__post_init__()

    def __iter__(self):
        return iter(self._scorers.items())

    def __getitem__(self, key):
        return self._scorers[key]

    def get_callables(self):
        """
        ----
        Returns a dictionary of all available scoring functions.

        Each key in the returned dictionary corresponds to a scoring metric name,
        and the value is the associated callable scorer function.

        Returns
        -------
        dict
            A mapping from metric names to scorer callables.
        ----
        """
        return {key: scorer for key, scorer in self._scorers.items()}

    def __call__(self, mode: Literal["test", "train", "attack", None] = "test", data: DataConfig =None,  model: ModelConfig = None, attack: AttackConfig = None, y_pred =None, y_true = None, score_file =None, **kwargs) -> Dict[str, float]:
        """
        ----
        Computes and returns a dictionary of scores for the given true and predicted labels.

        Parameters
        ----------
        data : DataConfig
            The data configuration containing true and predicted labels.
        model : ModelConfig, optional
            The model configuration (not used in scoring).
        attack : AttackConfig, optional
            The attack configuration (not used in scoring).
        mode : Literal["test", "train", "attack", None], optional
            The mode indicating which dataset to use for scoring. 
            Default is "test" where y_true is data.y_test and y_pred=model.test_predictions.
            "train" uses data.y_train and model.training_predictions.
            "attack" uses data.y_test[:attack.attack_size] and attack.attack_predictions.
            If None, y_true and y_pred must be provided directly.
        y_pred : array-like, optional
            The predicted labels. If None, predictions will be fetched from the model/data based on the mode.
        y_true : array-like
            The true labels.
        score_file : str, optional
            Path to a file containing precomputed scores. If provided and the file exists,
        **kwargs : dict, optional
            Additional keyword arguments passed to each scorer.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping scorer names to their computed score values.
        ----
        """
        if score_file is not None:
            if Path(score_file).exists():
                results = self.load_scores(score_file)
        else:
            results = {}
        if y_pred is not None:
            assert y_true is not None, "If y_pred is provided, y_true must also be provided. Otherwise, set y_pred to None and let the scorer fetch from data/model."
        else:
            if mode == "test":
                y_true = data.y_test
            elif mode == "train":
                y_true = data.y_train
            elif mode == "attack":
                assert isinstance(attack, AttackConfig), "attack must be an instance of AttackConfig"
                y_true = data.y_test[:attack.attack_size]
            else:
                assert y_true is not None, "y_true must be provided if mode is None"
            if model is not None:
                assert isinstance(model, ModelConfig), "model must be an instance of ModelConfig"
                assert hasattr(model, "_model"), "model must have a loaded _model attribute. Call model() first."
                assert hasattr(model, "predictions"), "model must have predictions attribute. Call model() first."
                loaded_model = model._model
                # Replace the {model} placeholder in kwargs if present
                assert "{model}" in kwargs.values(), "If model is provided, '{model}' must be in kwargs"
                for k, v in kwargs.items():
                    if v == "{model}":
                        kwargs[k] = loaded_model
            if mode == "train":
                y_pred = model.training_predictions
            elif mode == "test":
                y_pred = model.predictions
            elif mode == "attack":
                assert isinstance(attack, AttackConfig), "attack must be an instance of AttackConfig"
                y_pred = attack.attack_predictions
            else:
                assert y_pred is not None, "y_pred must be provided if mode is None"
        if attack is not None:
            for k, v in kwargs.items():
                    if v == "{attack}":
                        assert isinstance(attack, AttackConfig), "attack must be an instance of AttackConfig"
                        assert hasattr(attack, "_attack"), "attack must have a loaded _attack attribute. Call attack() first."
                        kwargs[k] = attack._attack
        for key, scorer in self._scorers.items():
            if mode == "test":
                pass
            elif model == "train":
                key = f"training_{key}"
            elif mode == "attack":
                key = f"attack_{key}"
            if results.get(key) is None:
                results[key] = scorer(y_true=y_true, y_pred=y_pred, **kwargs)
            else:
                pass
        if score_file is not None:
            self.save_scores(score_file, results)
        return results


class DefaultClassifierDict:
    """
    ----
    DefaultClassifierDict
    ----

    Provides a default dictionary of scoring metrics for classification tasks.

    Attributes
    ----------
    scorers : ScorerDictConfig
        A configuration object containing common classification scorers:
        - "accuracy": Uses `accuracy_score` to measure overall accuracy.
        - "precision": Uses `precision_score` with weighted averaging and zero_division=0.
        - "recall": Uses `recall_score` with weighted averaging and zero_division=0.
        - "f1": Uses `f1_score` with weighted averaging and zero_division=0.
        - "roc_auc": Uses `roc_auc_score` with weighted averaging and multi-class 'ovr'.
        - "log_loss": Uses `log_loss` with optional label specification.

    Usage
    -----
    This class is intended to provide a standardized set of scorers for evaluating classification models.
    Each scorer is configurable via its associated `ScorerConfig`.

    Example
    -------
    >>> scorers = DefaultClassifierDict.scorers
    >>> scorers["accuracy"].score_function(y_true, y_pred)
    """

    scorers: ScorerDictConfig = ScorerDictConfig(
        scorers={
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function=accuracy_score,
                score_params={},
            ),
            "precision": ScorerConfig(
                score_name="precision",
                score_function=precision_score,
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "recall": ScorerConfig(
                score_name="recall",
                score_function=recall_score,
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "f1": ScorerConfig(
                score_name="f1",
                score_function=f1_score,
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "roc_auc": ScorerConfig(
                score_name="roc_auc",
                score_function=roc_auc_score,
                score_params={"average": "weighted", "multi_class": "ovr"},
            ),
            "log_loss": ScorerConfig(
                score_name="log_loss",
                score_function=log_loss,
                score_params={"labels": None},
            ),
        },
    )


class DefaultRegressorDict:
    """
    ----
    Provides a default dictionary of regression scorers for model evaluation.

    Attributes
    ----------
    scorers : ScorerDictConfig
        A configuration object containing standard regression metrics:
            - "mse": Mean Squared Error (lower is better)
            - "mae": Mean Absolute Error (lower is better)
            - "r2": R^2 Score (higher is better)

    Usage
    -----
    Used to supply common regression metrics for scoring models in Deckard.
    ----
    """

    scorers: ScorerDictConfig = ScorerDictConfig(
        scorers={
            "mse": ScorerConfig(
                score_name="mse",
                score_function=mean_squared_error,
                greater_is_better=False,
            ),
            "mae": ScorerConfig(
                score_name="mae",
                score_function=mean_absolute_error,
                greater_is_better=False,
            ),
            "r2": ScorerConfig(
                score_name="r2",
                score_function=r2_score,
                greater_is_better=True,
            ),
        },
    )
