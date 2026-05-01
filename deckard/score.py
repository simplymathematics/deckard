"""Scoring configuration primitives for Deckard models and attacks.

This module provides small wrappers around callable metrics and the default
classifier/regressor scorer collections used throughout the experiment and model
pipelines.
"""

from dataclasses import dataclass, field
import logging
from typing import Literal, Dict, Any
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
from .utils import load_class

logger = logging.getLogger(__name__)

__all__ = [
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultClassifierDict",
    "DefaultRegressorDict",
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
]


def survival_concordance_score(y_true, y_pred, **kwargs):
    """Return survival concordance from a fitted lifelines model when available."""
    if hasattr(y_pred, "concordance_index_"):
        return float(y_pred.concordance_index_)
    raise ValueError("y_pred must be a fitted survival model with concordance_index_")


def survival_aic_score(y_true, y_pred, **kwargs):
    """Return survival AIC from a fitted lifelines model when available."""
    if hasattr(y_pred, "AIC_"):
        return float(y_pred.AIC_)
    if hasattr(y_pred, "partial_AIC_"):
        return float(y_pred.partial_AIC_)
    if hasattr(y_pred, "log_likelihood_"):
        k = None
        if hasattr(y_pred, "params_"):
            k = len(getattr(y_pred, "params_"))
        elif hasattr(y_pred, "params") and callable(getattr(y_pred, "params")):
            k = len(y_pred.params())
        if k is not None:
            return float(-2.0 * float(y_pred.log_likelihood_) + 2.0 * float(k))
    raise ValueError("y_pred must expose AIC_ or enough information to compute AIC")


def survival_bic_score(y_true, y_pred, **kwargs):
    """Return survival BIC from a fitted lifelines model when available."""
    if hasattr(y_pred, "BIC_"):
        return float(y_pred.BIC_)

    n = kwargs.get("n_samples")
    if n is None and y_true is not None:
        try:
            n = len(y_true)
        except TypeError:
            n = None

    if n is not None and hasattr(y_pred, "log_likelihood_"):
        k = None
        if hasattr(y_pred, "params_"):
            k = len(getattr(y_pred, "params_"))
        elif hasattr(y_pred, "params") and callable(getattr(y_pred, "params")):
            k = len(y_pred.params())
        if k is not None and n > 0:
            import math

            return float(-2.0 * float(y_pred.log_likelihood_) + float(k) * math.log(n))

    raise ValueError("y_pred must expose BIC_ or enough information to compute BIC")


def _resolve_sensitive_features(data, y_true):
    if data is None:
        return None
    y_len = len(y_true)
    candidates = [
        getattr(data, "_sensitive_test", None),
        getattr(data, "_sensitive_train", None),
        getattr(data, "_sensitive_all", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if len(candidate) == y_len:
            return candidate
    return None


def fairness_demographic_parity_difference(y_true, y_pred, data=None, **kwargs):
    """Compute demographic parity difference for fairness-aware configurations."""
    try:
        from fairlearn.metrics import demographic_parity_difference
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        ) from exc
    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(data, y_true)
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")
    return float(
        demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    )


def fairness_equalized_odds_difference(y_true, y_pred, data=None, **kwargs):
    """Compute equalized odds difference for fairness-aware configurations."""
    try:
        from fairlearn.metrics import equalized_odds_difference
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        ) from exc
    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(data, y_true)
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")
    return float(
        equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    )


@dataclass
class ScorerConfig:
    """Wrap a metric callable together with its runtime parameters.

    ``ScorerConfig`` is the atomic scoring unit used by ``ScorerDictConfig``.
    It stores the metric name, the callable itself, and any static keyword
    arguments that should be supplied on every invocation.
    """

    score_name: str
    score_function: callable
    score_params: dict = field(default_factory=dict)
    greater_is_better: bool = True
    needs_proba: bool = False

    def __post_init__(self):
        """
        Validates the scoring function and initializes the scorer using make_scorer.
        """
        if len(self.score_params) == 0:
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
    """Container of named ``ScorerConfig`` instances.

    The object behaves like a small registry for metrics and is the public score
    configuration type accepted by experiments, models, and attacks.
    """

    def __init__(self, scorers: dict):
        """Initialize the scorer registry.

        Parameters
        ----------
        scorers : dict
            Mapping from metric names to ``ScorerConfig`` instances.
        """
        self._scorers = scorers
        self.__post_init__()

    def __post_init__(self):
        """Validate scorer entries and initialize child scorers.

        Raises
        ------
        AssertionError
            If any entry in ``_scorers`` is not a ``ScorerConfig`` instance.
        """
        normalized_scorers = {}
        for key, value in self._scorers.items():
            if isinstance(value, ScorerConfig):
                scorer = value
            elif isinstance(value, dict):
                scorer_dict = dict(value)
                score_name = scorer_dict.pop("score_name", key)
                score_function = scorer_dict.pop("score_function", None)
                if isinstance(score_function, str):
                    score_function = load_class(score_function)
                if not callable(score_function):
                    raise TypeError(
                        f"score_function for scorer '{key}' must be callable or import path string",
                    )
                scorer = ScorerConfig(
                    score_name=score_name,
                    score_function=score_function,
                    score_params=scorer_dict.pop("score_params", {}),
                    greater_is_better=scorer_dict.pop("greater_is_better", True),
                    needs_proba=scorer_dict.pop("needs_proba", False),
                )
            else:
                raise TypeError(
                    f"Value for key '{key}' must be ScorerConfig or dict, got {type(value)}",
                )
            scorer.__post_init__()
            normalized_scorers[key] = scorer
        self._scorers = normalized_scorers

    def __iter__(self):
        return iter(self._scorers.items())

    def __getitem__(self, key):
        return self._scorers[key]

    def get_callables(self):
        """Return the configured scorer callables.

        Returns
        -------
        dict
            Mapping from metric names to ``ScorerConfig`` objects.
        """
        return {key: scorer for key, scorer in self._scorers.items()}

    def __call__(
        self,
        mode: Literal["test", "train", "attack", None] = "test",
        data: DataConfig = None,
        model: ModelConfig = None,
        attack: AttackConfig = None,
        y_pred=None,
        y_true=None,
        score_file=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute and return scores for true and predicted labels.

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
        """
        results = {}
        if score_file is not None and Path(score_file).exists():
            results = self.load_scores(score_file)
        if y_pred is not None:
            assert (
                y_true is not None
            ), "If y_pred is provided, y_true must also be provided. Otherwise, set y_pred to None and let the scorer fetch from data/model."
        else:
            if mode == "test":
                y_true = data.y_test
            elif mode == "train":
                y_true = data.y_train
            elif mode == "attack":
                assert isinstance(
                    attack,
                    AttackConfig,
                ), "attack must be an instance of AttackConfig"
                y_true = data.y_test[: attack.attack_size]
            else:
                assert y_true is not None, "y_true must be provided if mode is None"
            if model is not None:
                assert isinstance(
                    model,
                    ModelConfig,
                ), "model must be an instance of ModelConfig"
                assert hasattr(
                    model,
                    "_model",
                ), "model must have a loaded _model attribute. Call model() first."
                assert hasattr(
                    model,
                    "predictions",
                ), "model must have predictions attribute. Call model() first."
                loaded_model = model._model
                # Replace the {model} placeholder in kwargs if present
                for k, v in kwargs.items():
                    if v == "{model}":
                        kwargs[k] = loaded_model
            if mode == "train":
                y_pred = model.training_predictions
            elif mode == "test":
                y_pred = model.predictions
            elif mode == "attack":
                assert isinstance(
                    attack,
                    AttackConfig,
                ), "attack must be an instance of AttackConfig"
                y_pred = attack.attack_predictions
            else:
                assert y_pred is not None, "y_pred must be provided if mode is None"
        if attack is not None:
            for k, v in kwargs.items():
                if v == "{attack}":
                    assert isinstance(
                        attack,
                        AttackConfig,
                    ), "attack must be an instance of AttackConfig"
                    assert hasattr(
                        attack,
                        "_attack",
                    ), "attack must have a loaded _attack attribute. Call attack() first."
                    kwargs[k] = attack._attack
        for key, scorer in self._scorers.items():
            if mode == "test":
                pass
            elif mode == "train":
                key = f"training_{key}"
            elif mode == "attack":
                key = f"attack_{key}"
            if results.get(key) is None:
                results[key] = scorer(y_true=y_true, y_pred=y_pred, **kwargs)
            else:
                pass
        if score_file is not None:
            self.save_scores(results, score_file)
        return results


class DefaultClassifierDict:
    """
    DefaultClassifierDict

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
