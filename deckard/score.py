
from dataclasses import dataclass
import logging
from typing import Dict
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
            assert isinstance(value, ScorerConfig), f"Value for key '{key}' must be an instance of ScorerConfig"
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

    def __call__(self, y_true, y_pred, **kwargs) -> Dict[str, float]:
        """
        ----
        Computes and returns a dictionary of scores for the given true and predicted labels.

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.
        **kwargs : dict, optional
            Additional keyword arguments passed to each scorer.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping scorer names to their computed score values.
        ----
        """
        results = {}
        for key, scorer in self._scorers.items():
            results[key] = scorer(y_true=y_true, y_pred=y_pred,  **kwargs)
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
    scorers : ScorerDictConfig = ScorerDictConfig(
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
        }
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
    scorers : ScorerDictConfig = ScorerDictConfig(
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
        }
    )