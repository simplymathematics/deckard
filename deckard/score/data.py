"""Dataset analysis scorers and ConfigStore registrations.

These scorers are intended for inspecting dataset properties without training a
model. They accept feature matrices through ``y_pred`` and support optional
reference-column overrides so analysis can target a non-label column.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from .base import (
    ScorerConfig,
    ScorerDictConfig,
    _DataScorerMarker,
    _TaskAwareScorerMixin,
    safe_store,
)


def _coerce_features_dataframe(y_pred) -> pd.DataFrame:
    if isinstance(y_pred, pd.DataFrame):
        return y_pred
    if isinstance(y_pred, pd.Series):
        return pd.DataFrame({y_pred.name or "feature_0": y_pred})
    arr = np.asarray(y_pred)
    if arr.ndim == 1:
        return pd.DataFrame({"feature_0": arr})
    return pd.DataFrame(
        arr,
        columns=[f"feature_{i}" for i in range(arr.shape[1])],
    )


def _resolve_reference_vector(y_true, X: pd.DataFrame, **kwargs):
    reference = kwargs.get("reference", None)
    if reference is not None:
        return np.asarray(reference), X

    reference_column = kwargs.get("reference_column", None)
    if reference_column is not None:
        if reference_column not in X.columns:
            raise ValueError(
                f"reference_column '{reference_column}' not found in features",
            )
        reference_values = np.asarray(X[reference_column])
        # Avoid self-information inflation when the reference is itself a feature.
        X = X.drop(columns=[reference_column])
        return reference_values, X

    return np.asarray(y_true), X


def _is_discrete_reference(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    series = pd.Series(values)
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(
        series,
    ):
        return True
    if pd.api.types.is_integer_dtype(series):
        unique = int(series.nunique(dropna=True))
        threshold = min(20, max(2, len(series) // 5))
        return unique <= threshold
    return False


def _feature_mutual_information_vector(y_true, y_pred, **kwargs) -> np.ndarray:
    X = _coerce_features_dataframe(y_pred)
    reference, X = _resolve_reference_vector(y_true=y_true, X=X, **kwargs)
    if X.shape[1] == 0:
        raise ValueError(
            "No feature columns available for mutual-information analysis",
        )

    random_state = kwargs.get("random_state", 42)
    discrete_reference = kwargs.get("discrete_reference", None)
    if discrete_reference is None:
        discrete_reference = _is_discrete_reference(reference)

    if discrete_reference:
        mi = mutual_info_classif(X, reference, random_state=random_state)
    else:
        mi = mutual_info_regression(X, reference, random_state=random_state)
    return np.asarray(mi, dtype=float)


def data_num_classes_score(y_true: Any, y_pred: Any, **kwargs: Any) -> int:
    """Count the number of unique classes in ``y_true``.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    y_pred : array-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    int
        Number of distinct classes in ``y_true`` including NaN if present.
    """
    _ = y_pred, kwargs
    return int(pd.Series(y_true).nunique(dropna=False))


def data_class_count_min_score(y_true: Any, y_pred: Any, **kwargs: Any) -> int:
    """Return the count of the least-frequent class in ``y_true``.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    y_pred : array-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    int
        Sample count of the rarest class, or 0 when ``y_true`` is empty.
    """
    _ = y_pred, kwargs
    counts = pd.Series(y_true).value_counts(dropna=False)
    return int(counts.min()) if len(counts) else 0


def data_class_count_max_score(y_true: Any, y_pred: Any, **kwargs: Any) -> int:
    """Return the count of the most-frequent class in ``y_true``.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    y_pred : array-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    int
        Sample count of the most common class, or 0 when ``y_true`` is empty.
    """
    _ = y_pred, kwargs
    counts = pd.Series(y_true).value_counts(dropna=False)
    return int(counts.max()) if len(counts) else 0


def data_class_imbalance_ratio_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Return the ratio of the most-frequent to the least-frequent class.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    y_pred : array-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    float
        ``max_count / min_count``.  Returns ``0.0`` when ``y_true`` is empty
        and ``float('inf')`` when the minority class has zero samples.
    """
    _ = y_pred, kwargs
    counts = pd.Series(y_true).value_counts(dropna=False)
    if len(counts) == 0:
        return 0.0
    min_count = float(counts.min())
    max_count = float(counts.max())
    if min_count == 0:
        return float("inf")
    return max_count / min_count


def data_mutual_information_mean_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Return the mean mutual information between features and the reference vector.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels used as the reference vector unless ``reference``
        or ``reference_column`` is supplied in *kwargs*.
    y_pred : array-like or pd.DataFrame
        Feature matrix.  Non-DataFrame inputs are coerced automatically.
    **kwargs
        Forwarded to :func:`_feature_mutual_information_vector`.  Accepts
        ``reference``, ``reference_column``, ``random_state``, and
        ``discrete_reference``.

    Returns
    -------
    float
        Mean mutual information across all feature columns.
    """
    mi = _feature_mutual_information_vector(
        y_true=y_true,
        y_pred=y_pred,
        **kwargs,
    )
    return float(np.mean(mi))


def data_mutual_information_max_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Return the maximum mutual information between any single feature and the reference.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels used as the reference vector unless ``reference``
        or ``reference_column`` is supplied in *kwargs*.
    y_pred : array-like or pd.DataFrame
        Feature matrix.  Non-DataFrame inputs are coerced automatically.
    **kwargs
        Forwarded to :func:`_feature_mutual_information_vector`.  Accepts
        ``reference``, ``reference_column``, ``random_state``, and
        ``discrete_reference``.

    Returns
    -------
    float
        Maximum mutual information value across all feature columns.
    """
    mi = _feature_mutual_information_vector(
        y_true=y_true,
        y_pred=y_pred,
        **kwargs,
    )
    return float(np.max(mi))


def data_empirical_cdf_function_score(y_true: Any, y_pred: Any, **kwargs: Any) -> Callable:
    """Return an empirical CDF function fitted to the reference vector.

    The returned callable accepts an array of query values and returns the
    fraction of reference samples that are less than or equal to each query
    value.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values used as the reference unless ``reference`` or
        ``reference_column`` is provided in *kwargs*.
    y_pred : array-like or pd.DataFrame
        Feature matrix (used only when resolving ``reference_column``).
    **kwargs
        Forwarded to :func:`_resolve_reference_vector`.  Accepts
        ``reference`` and ``reference_column``.

    Returns
    -------
    callable
        An ECDF function ``ecdf(x) -> np.ndarray`` with the same signature
        as :class:`scipy.stats.ecdf`.

    Raises
    ------
    ValueError
        If the resolved reference vector is empty after dropping NaN values.
    """
    X = _coerce_features_dataframe(y_pred)
    reference, _ = _resolve_reference_vector(y_true=y_true, X=X, **kwargs)
    ref = pd.Series(reference).dropna().astype(float).to_numpy()
    if ref.size == 0:
        raise ValueError("Reference vector is empty after dropping NaN values")

    sorted_vals = np.sort(ref)
    n = float(sorted_vals.size)

    def ecdf(x):
        values = np.asarray(x)
        return np.searchsorted(sorted_vals, values, side="right") / n

    return ecdf


@dataclass(eq=False)
class DefaultDataScoreConfig(_DataScorerMarker, _TaskAwareScorerMixin, ScorerDictConfig):
    """Default data-analysis scorer family with optional task inheritance."""

    classifier: Union[bool, str, None] = None
    scorers: Dict[str, Union[ScorerConfig, Dict[str, Any]]] = field(default_factory=dict)

    def _build_default_scorers(self, classifier: bool) -> Dict[str, Union[ScorerConfig, Dict[str, Any]]]:
        if classifier:
            return {
                "num_classes": ScorerConfig(
                    score_name="num_classes",
                    score_function="deckard.score.data.data_num_classes_score",
                ),
                "class_count_min": ScorerConfig(
                    score_name="class_count_min",
                    score_function="deckard.score.data.data_class_count_min_score",
                ),
                "class_count_max": ScorerConfig(
                    score_name="class_count_max",
                    score_function="deckard.score.data.data_class_count_max_score",
                ),
                "class_imbalance_ratio": ScorerConfig(
                    score_name="class_imbalance_ratio",
                    score_function="deckard.score.data.data_class_imbalance_ratio_score",
                    greater_is_better=False,
                ),
                "mutual_information_mean": ScorerConfig(
                    score_name="mutual_information_mean",
                    score_function="deckard.score.data.data_mutual_information_mean_score",
                ),
                "mutual_information_max": ScorerConfig(
                    score_name="mutual_information_max",
                    score_function="deckard.score.data.data_mutual_information_max_score",
                ),
            }
        return {
            "mutual_information_mean": ScorerConfig(
                score_name="mutual_information_mean",
                score_function="deckard.score.data.data_mutual_information_mean_score",
            ),
            "mutual_information_max": ScorerConfig(
                score_name="mutual_information_max",
                score_function="deckard.score.data.data_mutual_information_max_score",
            ),
            "empirical_cdf": ScorerConfig(
                score_name="empirical_cdf",
                score_function="deckard.score.data.data_empirical_cdf_function_score",
            ),
        }

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False)
class DefaultDataClassificationConfig(DefaultDataScoreConfig):
    """Default dataset-analysis scorers for classification datasets."""

    classifier: Union[bool, str, None] = True


@dataclass(eq=False)
class DefaultDataRegressionConfig(DefaultDataScoreConfig):
    """Default dataset-analysis scorers for regression datasets."""

    classifier: Union[bool, str, None] = False


safe_store(
    group="score",
    name="data-classification",
    node={"_target_": "deckard.score.data.DefaultDataScoreConfig", "classifier": True},
)
safe_store(
    group="score",
    name="data-regression",
    node={"_target_": "deckard.score.data.DefaultDataScoreConfig", "classifier": False},
)


__all__ = [
    "data_num_classes_score",
    "data_class_count_min_score",
    "data_class_count_max_score",
    "data_class_imbalance_ratio_score",
    "data_mutual_information_mean_score",
    "data_mutual_information_max_score",
    "data_empirical_cdf_function_score",
    "DefaultDataScoreConfig",
    "DefaultDataClassificationConfig",
    "DefaultDataRegressionConfig",
]
