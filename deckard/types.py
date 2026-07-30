"""Canonical runtime typing protocols for core and extension modules.

This module is the canonical home for framework-agnostic runtime typing
markers used across Deckard core, framework adapters, and plugin families.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeAlias, Any

import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_art_symbols() -> dict[str, Any]:
    from art.estimators.classification import PyTorchClassifier
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
    from art.estimators.regression import PyTorchRegressor
    from art.estimators.regression.scikitlearn import (
        ScikitlearnDecisionTreeRegressor,
        ScikitlearnRegressor,
    )

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
    return {
        "classifier_dict": classifier_dict,
        "regressor_dict": regressor_dict,
        "sklearn_dict": sklearn_dict,
        "sklearn_models": list(sklearn_dict.keys()),
        "torch_wrapper_types": (PyTorchClassifier, PyTorchRegressor),
        "torch_classifier": PyTorchClassifier,
        "torch_regressor": PyTorchRegressor,
    }


class RuntimeValue(Protocol):
    """Marker protocol for framework runtime payloads."""


class MatrixLike(Protocol):
    """Structural protocol for matrix-like payloads."""

    def __len__(self) -> int:
        """Return row or batch count when available."""
        ...

    def __iter__(self) -> object:
        """Yield rows, batches, or records."""
        ...


class ArrayLike(Protocol):
    """Structural protocol for array-like payloads."""

    def __len__(self) -> int:
        """Return element count."""
        ...

    def __iter__(self) -> object:
        """Yield elements, batches, or records."""
        ...


class SklearnModelLike(Protocol):
    def predict(self, X: Any) -> "ArrayLike": ...

    def __call__(self, *args: Any, **kwargs: Any) -> "EstimatorLike": ...

    def fit(self, X: Any, y: Any) -> "EstimatorLike": ...


class TorchModelLike(Protocol):
    def forward(self, X: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> "EstimatorLike": ...


class ARTEstimatorLike(Protocol):
    model: EstimatorLike
    _apply_fit: bool
    _apply_predict: bool

    def predict(self, X: Any) -> "ArrayLike": ...

    def __call__(self, *args: Any, **kwargs: Any) -> "EstimatorLike": ...

    def fit(self, X: Any, y: Any) -> "EstimatorLike": ...


EstimatorLike: TypeAlias = SklearnModelLike | TorchModelLike | ARTEstimatorLike


class AttackLike(Protocol):
    """Structural protocol for runtime attack objects."""

    def __len__(self) -> int:
        """Return attack size metadata when available."""
        ...


StringifiedClass: TypeAlias = str
DatasetLike: TypeAlias = str | Path


TabularLike: TypeAlias = pd.DataFrame | pd.Series
IndexLike: TypeAlias = "list[int]"


__all__ = [
    "RuntimeValue",
    "MatrixLike",
    "ArrayLike",
    "EstimatorLike",
    "AttackLike",
    "ARTEstimatorLike",
    "StringifiedClass",
    "DatasetLike",
    "TabularLike",
    "IndexLike",
]
