"""Tests for the deckard.frameworks package public API."""

from deckard.frameworks import ArrayLike, EstimatorLike, MatrixLike, RuntimeValue


def test_frameworks_package_re_exports_canonical_types():
    from deckard.types import ArrayLike as _ArrayLike

    assert RuntimeValue is not None
    assert MatrixLike is not None
    assert EstimatorLike is not None
    assert ArrayLike is _ArrayLike
