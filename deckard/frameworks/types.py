"""Shared framework typing protocols.

This module contains framework-agnostic runtime typing markers used across
deckard configs and framework integrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
    import pandas as pd


class RuntimeValue(Protocol):
    """Marker protocol for framework runtime payloads."""


class MatrixLike(Protocol):
    """Structural protocol for matrix-like payloads."""

    def __len__(self) -> int:
        """Return row or batch count when available."""

    def __iter__(self) -> object:
        """Yield rows, batches, or records."""


class ArrayLike(Protocol):
    """Structural protocol for array-like payloads."""

    def __len__(self) -> int:
        """Return element count."""

    def __iter__(self) -> object:
        """Yield elements, batches, or records."""


class EstimatorLike(Protocol):
    """Structural protocol for framework estimator runtime objects."""

    def __len__(self) -> int:
        """Return size metadata when available."""


if TYPE_CHECKING:
    TabularLike: TypeAlias = pd.DataFrame | pd.Series
else:
    TabularLike: TypeAlias = Any
IndexLike: TypeAlias = "list[int]"


__all__ = [
    "RuntimeValue",
    "MatrixLike",
    "ArrayLike",
    "EstimatorLike",
    "TabularLike",
    "IndexLike",
]
