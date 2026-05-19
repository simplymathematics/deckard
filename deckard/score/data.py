"""Dataset analysis scorers and ConfigStore registrations.

These scorers are intended for inspecting dataset properties without training a
model. They accept feature matrices through ``X`` and support optional
reference-column overrides so analysis can target a non-label column.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

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

LabelVector = Union[np.ndarray, pd.Series, Sequence[Union[int, float, str, bool]]]
FeatureMatrix = Union[np.ndarray, pd.Series, pd.DataFrame, Sequence[Sequence[float]]]
KwargMap = dict[str, Any]


def _coerce_features_dataframe(X: FeatureMatrix) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                codes, _ = pd.factorize(frame[column], sort=True)
                frame[column] = codes
        return frame
    if isinstance(X, pd.Series):
        if pd.api.types.is_numeric_dtype(X):
            return pd.DataFrame({X.name or "feature_0": X})
        codes, _ = pd.factorize(X, sort=True)
        return pd.DataFrame({X.name or "feature_0": codes})
    arr = np.asarray(X)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        frame = pd.DataFrame({"feature_0": arr})
        if not pd.api.types.is_numeric_dtype(frame["feature_0"]):
            frame["feature_0"] = pd.factorize(frame["feature_0"], sort=True)[0]
        return frame
    frame = pd.DataFrame(
        arr,
        columns=[f"feature_{i}" for i in range(arr.shape[1])],
    )
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            codes, _ = pd.factorize(frame[column], sort=True)
            frame[column] = codes
    return frame


def _resolve_reference_vector(
    y_true: LabelVector,
    X: pd.DataFrame,
    **kwargs: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
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


def _feature_mutual_information_vector(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> np.ndarray:
    X = _coerce_features_dataframe(X)
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


def data_num_classes_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> int:
    """Count the number of unique classes in ``y_true``.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    X : matrix-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    int
        Number of distinct classes in ``y_true`` including NaN if present.
    """
    _ = X, kwargs
    return int(pd.Series(y_true).nunique(dropna=False))


def data_class_count_min_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> int:
    """Return the count of the least-frequent class in ``y_true``.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    X : matrix-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    int
        Sample count of the rarest class, or 0 when ``y_true`` is empty.
    """
    _ = X, kwargs
    counts = pd.Series(y_true).value_counts(dropna=False)
    return int(counts.min()) if len(counts) else 0


def data_class_count_max_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> int:
    """Return the count of the most-frequent class in ``y_true``.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    X : matrix-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    int
        Sample count of the most common class, or 0 when ``y_true`` is empty.
    """
    _ = X, kwargs
    counts = pd.Series(y_true).value_counts(dropna=False)
    return int(counts.max()) if len(counts) else 0


def data_class_imbalance_ratio_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> float:
    """Return the ratio of the most-frequent to the least-frequent class.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target labels.
    X : matrix-like
        Predicted values (unused; present for scorer interface compatibility).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    float
        ``max_count / min_count``.  Returns ``0.0`` when ``y_true`` is empty
        and ``float('inf')`` when the minority class has zero samples.
    """
    _ = X, kwargs
    counts = pd.Series(y_true).value_counts(dropna=False)
    if len(counts) == 0:
        return 0.0
    min_count = float(counts.min())
    max_count = float(counts.max())
    if min_count == 0:
        return float("inf")
    return max_count / min_count


def data_mutual_information_mean_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> float:
    """Return the mean mutual information between features and the reference vector.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels used as the reference vector unless ``reference``
        or ``reference_column`` is supplied in *kwargs*.
    X : matrix-like or pd.DataFrame
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
        X=X,
        **kwargs,
    )
    return float(np.mean(mi))


def data_mutual_information_max_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> float:
    """Return the maximum mutual information between any single feature and the reference.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels used as the reference vector unless ``reference``
        or ``reference_column`` is supplied in *kwargs*.
    X : matrix-like or pd.DataFrame
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
        X=X,
        **kwargs,
    )
    return float(np.max(mi))


def data_empirical_cdf_function_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an empirical CDF function fitted to the reference vector.

    The returned callable accepts an array of query values and returns the
    fraction of reference samples that are less than or equal to each query
    value.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values used as the reference unless ``reference`` or
        ``reference_column`` is provided in *kwargs*.
    X : matrix-like or pd.DataFrame
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
    X = _coerce_features_dataframe(X)
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


@dataclass(eq=False, kw_only=True)
class DefaultDataScorerConfig(
    _DataScorerMarker,
    _TaskAwareScorerMixin,
    ScorerDictConfig,
):
    """Default data-analysis scorer family with optional task inheritance.

    Initialization parameters
    -------------------------
    classifier : bool | str | None
        Task type selector. Accepted values are ``True``, ``False``,
        ``"classifier"``, ``"regressor"``, or ``None``. When ``None``,
        task type is resolved from data/model/attack context or defaults to ``True``.
    scorers : dict[str, ScorerConfig | dict[str, Any]]
        Named scorer configurations. When empty, populated by ``_build_default_scorers``
        based on resolved task type (classification vs regression).

    Runtime parameters
    -------------------
    data : DataConfig
        Runtime data configuration containing train/test/val splits, feature matrices.
    model : Any
        Runtime model object (optional, used for context resolution).
    attack : Any
        Runtime attack object (optional, used for context resolution).

    Parameter layers
    ----------------
    1. Task awareness: ``classifier`` field enables task-specific scorer defaults
    2. Data properties: Scorers analyze feature matrices and label distributions
    3. Reference vector: Mutual information and ECDF scorers accept optional ``reference``
       or ``reference_column`` overrides for non-label analysis targets

    Family-specific parameter semantics
    -----------------------------------
    Data scorers operate on dataset properties without requiring trained models:

    - **Class distribution**: ``num_classes``, ``class_count_min/max``, ``class_imbalance_ratio``
      measure label-space characteristics (classification only).
    - **Feature informativeness**: ``mutual_information_mean/max`` quantify feature-label
      associations using information-theoretic measures.
    - **Empirical distribution**: ``empirical_cdf`` returns a calibrated CDF function for
      reference-vector percentile queries.

    Plugin pattern
    --------------
    This scorer inherits from ``_ScorerMixin`` semantics through ``ScorerDictConfig``.
    Plugins registered via ``ScorerTypePlugin`` contribute mixin-based runtime context
    for scope-based dispatch (e.g., ``scoring_type: "data"`` routes to this scorer).
    """

    classifier: Union[bool, str, None] = None
    scorers: dict[str, Union[ScorerConfig, KwargMap]] = field(default_factory=dict)

    def _build_default_scorers(
        self,
        classifier: bool,
    ) -> dict[str, Union[ScorerConfig, KwargMap]]:
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


@dataclass(eq=False, kw_only=True)
class DefaultDataClassificationConfig(DefaultDataScorerConfig):
    """Default dataset-analysis scorers for classification datasets.

    Initialization parameters
    -------------------------
    classifier : bool | str | None
        Fixed to ``True`` (classification mode). Class-distribution scorers
        (num_classes, class_count_min/max, class_imbalance_ratio) are included
        by default.

    Runtime behavior
    ----------------
    Inherits all runtime parameter resolution from ``DefaultDataScoreConfig``.
    Default scorers include both class-distribution and information-theoretic
    measures specific to classification analysis.
    """

    classifier: Union[bool, str, None] = True


@dataclass(eq=False, kw_only=True)
class DefaultDataRegressionConfig(DefaultDataScorerConfig):
    """Default dataset-analysis scorers for regression datasets.

    Initialization parameters
    -------------------------
    classifier : bool | str | None
        Fixed to ``False`` (regression mode). Excludes class-distribution scorers
        and includes only information-theoretic measures for continuous target analysis.

    Runtime behavior
    ----------------
    Inherits all runtime parameter resolution from ``DefaultDataScoreConfig``.
    Default scorers focus on feature-informativeness and empirical distribution
    measures suitable for regression analysis.
    """

    classifier: Union[bool, str, None] = False


def pytorch_split_count_score(
    y_true: LabelVector,
    X: FeatureMatrix,
    **kwargs: Any,
) -> int:
    """Return the number of samples available in the active split.

    Parameters
    ----------
    y_true : array-like
        Labels for the active split.
    X : matrix-like
        Feature matrix for the active split.
    **kwargs
        Additional scorer keyword arguments (unused).

    Returns
    -------
    int
        Number of samples in the active split.
    """
    _ = X, kwargs
    return int(len(np.asarray(y_true)))


@dataclass(eq=False, kw_only=True)
class DefaultPytorchDataScorerConfig(_TaskAwareScorerMixin, ScorerDictConfig):
    """Default tensor-aware data scorer family for PyTorch datasets.

    This scorer avoids feature-level mutual-information metrics, which are
    better suited to tabular sklearn-style data, and instead focuses on split
    counts plus task-appropriate dataset summaries.
    """

    classifier: Union[bool, str, None] = None
    scorers: dict[str, Union[ScorerConfig, KwargMap]] = field(default_factory=dict)

    def _build_default_scorers(
        self,
        classifier: bool,
    ) -> dict[str, Union[ScorerConfig, KwargMap]]:
        if classifier:
            return {
                "split_count": ScorerConfig(
                    score_name="split_count",
                    score_function="deckard.score.data.pytorch_split_count_score",
                ),
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
            }
        return {
            "split_count": ScorerConfig(
                score_name="split_count",
                score_function="deckard.score.data.pytorch_split_count_score",
            ),
            "empirical_cdf": ScorerConfig(
                score_name="empirical_cdf",
                score_function="deckard.score.data.data_empirical_cdf_function_score",
            ),
        }

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


safe_store(
    group="score",
    name="data-classification",
    node={
        "_target_": "deckard.score.data.DefaultDataScorerConfig",
        "classifier": True,
    },
)
safe_store(
    group="score",
    name="data-regression",
    node={
        "_target_": "deckard.score.data.DefaultDataScorerConfig",
        "classifier": False,
    },
)


__all__ = [
    "data_num_classes_score",
    "data_class_count_min_score",
    "data_class_count_max_score",
    "data_class_imbalance_ratio_score",
    "data_mutual_information_mean_score",
    "data_mutual_information_max_score",
    "data_empirical_cdf_function_score",
    "pytorch_split_count_score",
    "DefaultDataScorerConfig",
    "DefaultDataClassificationConfig",
    "DefaultDataRegressionConfig",
    "DefaultPytorchDataScorerConfig",
]
