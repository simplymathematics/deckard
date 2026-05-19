from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "papers"
    / "compression_distance"
    / "classifier_refactor.py"
)


spec = importlib.util.spec_from_file_location("classifier_refactor", MODULE_PATH)
classifier_refactor = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(classifier_refactor)


StringDistanceTransformer = classifier_refactor.StringDistanceTransformer
KernelToDistanceTransformer = classifier_refactor.KernelToDistanceTransformer


def test_string_distance_transformer_basic_shapes():
    tr = StringDistanceTransformer(metric="gzip", algorithm="sort")
    X_train = np.array(["alpha", "beta", "gamma"])
    tr.fit(X_train)

    train_mtx = tr.transform(X_train)
    assert train_mtx.shape == (3, 3)

    X_test = np.array(["delta", "epsilon"])
    test_mtx = tr.transform(X_test)
    assert test_mtx.shape == (2, 3)


def test_string_distance_transformer_full_matrix_slicing(tmp_path: Path):
    full = np.arange(16, dtype=float).reshape(4, 4)
    full_path = tmp_path / "full.npz"
    np.savez_compressed(full_path, data=full)

    tr = StringDistanceTransformer(metric="gzip", distance_matrix_full=str(full_path))
    tr.set_split_indices(train_indices=[0, 2], test_indices=[1, 3])

    X_all = np.array(["a", "b", "c", "d"])
    tr.fit(X_all)

    expected_train = full[np.ix_([0, 2], [0, 2])]
    assert np.array_equal(tr.mtx_, expected_train)

    X_test = np.array(["b", "d"])
    expected_test = full[np.ix_([1, 3], [0, 2])]
    assert np.array_equal(tr.transform(X_test), expected_test)


def test_pre_sample_fit_generates_and_persists_full_matrix(
    tmp_path: Path,
    monkeypatch,
):
    matrix_path = tmp_path / "generated_full.npz"
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])

    def fake_calc(X, Y, **kwargs):
        assert len(X) == 2 and len(Y) == 2
        return expected

    monkeypatch.setattr(
        classifier_refactor,
        "calculate_rectangular_distance_matrix",
        fake_calc,
    )

    tr = StringDistanceTransformer(
        metric="gzip",
        distance_matrix_full=str(matrix_path),
    )
    tr.pre_sample_fit(np.array(["left", "right"]))

    assert matrix_path.exists()
    assert np.array_equal(tr._full_matrix, expected)

    reloaded = np.load(matrix_path)["data"]
    assert np.array_equal(reloaded, expected)


def test_kernel_to_distance_assume_unit_diagonal_transform():
    kernel = np.array([[1.0, 0.5], [0.5, 1.0]])
    tr = KernelToDistanceTransformer(form="exp_neg", assume_unit_diagonal=True)
    out = tr.transform(kernel)

    expected = 2 - 2 * kernel
    assert np.allclose(out, expected)


def test_string_distance_transformer_preserves_dataframe_row_count():
    tr = StringDistanceTransformer(metric="gzip", algorithm="sort")
    X_train = pd.DataFrame(
        {
            "protocol": ["tcp", "udp", "icmp"],
            "service": ["http", "dns", "echo"],
        },
    )
    tr.fit(X_train)
    assert tr.mtx_.shape == (3, 3)

    X_test = pd.DataFrame(
        {
            "protocol": ["tcp", "udp"],
            "service": ["ssh", "smtp"],
        },
    )
    out = tr.transform(X_test)
    assert out.shape == (2, 3)


def test_unknown_metric_raises_assertion():
    with pytest.raises(AssertionError):
        StringDistanceTransformer(metric="not-a-metric")
