import numpy as np
import pytest

from deckard.frameworks.pytorch.score import (
    normalize_scoring_mode,
    resolve_sensitive_features,
    resolve_split_arrays,
)
from deckard.plugins.fairlearn.score import fairness_stage_to_split_mode


class _DummyData:
    _sensitive_train = np.array([0, 1])
    _sensitive_test = np.array([1, 0])
    _sensitive_val = np.array([1, 1])
    _sensitive_all = np.array([0, 1, 1, 0])

    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[2.0], [3.0]])
    y_test = np.array([1, 0])
    X_val = np.array([[4.0], [5.0]])
    y_val = np.array([0, 0])
    _X = np.array([[9.0], [8.0]])
    _y = np.array([1, 1])


def test_resolve_sensitive_features_accepts_stage_aliases():
    data = _DummyData()
    stage_map = fairness_stage_to_split_mode("test")

    assert np.array_equal(
        resolve_sensitive_features(
            data,
            "post-defense",
            stage_to_split_mode=stage_map,
        ),
        data._sensitive_test,
    )
    assert np.array_equal(
        resolve_sensitive_features(
            data,
            "post-pipeline",
            stage_to_split_mode=stage_map,
        ),
        data._sensitive_test,
    )
    assert np.array_equal(
        resolve_sensitive_features(
            data,
            "post-sample",
            stage_to_split_mode=stage_map,
        ),
        data._sensitive_test,
    )
    assert np.array_equal(
        resolve_sensitive_features(
            data,
            "adversarial",
            stage_to_split_mode=stage_map,
        ),
        data._sensitive_test,
    )


def test_resolve_split_arrays_accepts_stage_aliases():
    data = _DummyData()
    stage_map = fairness_stage_to_split_mode("test")

    x, y = resolve_split_arrays(data, "post-defense", stage_to_split_mode=stage_map)
    assert np.array_equal(x, data.X_test)
    assert np.array_equal(y, data.y_test)

    x_adv, y_adv = resolve_split_arrays(data, "adversarial", stage_to_split_mode=stage_map)
    assert np.array_equal(x_adv, data.X_test)
    assert np.array_equal(y_adv, data.y_test)


def test_resolve_sensitive_features_rejects_unknown_mode():
    data = _DummyData()
    with pytest.raises(ValueError, match="Unsupported fairness scoring mode"):
        resolve_sensitive_features(data, "definitely-not-a-stage")


def test_normalize_scoring_mode_maps_attack_aliases_to_split_modes():
    stage_map = fairness_stage_to_split_mode("val")
    assert normalize_scoring_mode("attack", stage_to_split_mode=stage_map) == "val"
    assert normalize_scoring_mode("attack-val", stage_to_split_mode=stage_map) == "val"
    assert normalize_scoring_mode("adversarial", stage_to_split_mode=stage_map) == "val"


def test_normalize_scoring_mode_requires_mapping_for_stage_aliases():
    with pytest.raises(ValueError, match="Unsupported fairness scoring mode"):
        normalize_scoring_mode("adversarial")


def test_resolve_split_arrays_allows_attack_stage_override_to_val():
    data = _DummyData()
    x, y = resolve_split_arrays(
        data,
        "adversarial",
        stage_to_split_mode={"adversarial": "val"},
    )
    assert np.array_equal(x, data.X_val)
    assert np.array_equal(y, data.y_val)


def test_resolve_sensitive_features_allows_attack_stage_override_to_val():
    data = _DummyData()
    sensitive = resolve_sensitive_features(
        data,
        "adversarial",
        stage_to_split_mode={"adversarial": "val"},
    )
    assert np.array_equal(sensitive, data._sensitive_val)
