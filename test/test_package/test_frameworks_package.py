"""Tests for the deckard.frameworks package public API."""

from deckard.frameworks import ArrayLike, EstimatorLike, MatrixLike, RuntimeValue
from deckard.frameworks.types import ArrayLike as _ArrayLike


def test_frameworks_types_are_importable():
    assert RuntimeValue is not None
    assert MatrixLike is not None
    assert ArrayLike is not None
    assert EstimatorLike is not None


def test_frameworks_types_re_exported_consistently():
    assert ArrayLike is _ArrayLike
