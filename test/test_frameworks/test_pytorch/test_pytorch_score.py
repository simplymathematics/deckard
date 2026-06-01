import numpy as np
import pytest

from deckard.frameworks.pytorch.score import (
    coerce_to_numpy,
    get_dataset_shape,
    is_dataloader_like,
    is_dataset_like,
    materialize_dataset,
    normalize_scoring_mode,
    resolve_sensitive_features,
    resolve_split_arrays,
    to_numpy,
    to_numpy_if_torch,
    validate_sensitive_features,
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

    x_adv, y_adv = resolve_split_arrays(
        data,
        "adversarial",
        stage_to_split_mode=stage_map,
    )
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
    assert (
        normalize_scoring_mode("adversarial", stage_to_split_mode=stage_map) == "val"
    )


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


def test_to_numpy_if_torch_recursively_converts_tensor_payloads():
    torch = pytest.importorskip("torch")

    payload = [torch.tensor([1, 2]), (torch.tensor([3]), "x")]
    converted = to_numpy_if_torch(payload)

    assert isinstance(converted[0], np.ndarray)
    assert converted[0].tolist() == [1, 2]
    assert isinstance(converted[1][0], np.ndarray)
    assert converted[1][1] == "x"


def test_to_numpy_and_coerce_to_numpy_cover_numpy_like_inputs():
    class _ArrayLike:
        def numpy(self):
            return np.array([1, 2, 3])

    class _DetachLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([4, 5, 6])

    assert np.array_equal(to_numpy(np.array([1, 2])), np.array([1, 2]))
    assert np.array_equal(to_numpy(_ArrayLike()), np.array([1, 2, 3]))
    assert coerce_to_numpy(None) is None
    assert np.array_equal(coerce_to_numpy([1, 2], dtype=float), np.array([1.0, 2.0]))
    assert np.array_equal(coerce_to_numpy(_DetachLike()), np.array([4, 5, 6]))


def test_dataset_detection_and_shape_helpers_cover_dataset_branches():
    class _Dataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return np.array([index, index + 1]), index

    class _DataLoader:
        batch_size = 2

        def __iter__(self):
            yield np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1])

    dataset = _Dataset()
    loader = _DataLoader()

    assert is_dataset_like(dataset) is True
    assert is_dataloader_like(loader) is True
    assert get_dataset_shape(dataset) == (3, 2)


def test_get_dataset_shape_supports_torch_subset_and_failure_path():
    torch = pytest.importorskip("torch")

    dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 3),
        torch.tensor([0, 1, 0, 1]),
    )
    subset = torch.utils.data.Subset(dataset, [0, 2])

    assert get_dataset_shape(subset) == (2, 3)

    with pytest.raises(AttributeError, match="has no determinable shape"):
        get_dataset_shape(object())


@pytest.mark.parametrize(
    ("sensitive", "y_true", "message"),
    [
        (None, [0], "Sensitive features are None"),
        ([], [0], "Sensitive features are empty"),
        ([None, None], [0, 1], "Sensitive features are all-null"),
        (["  ", ""], [0, 1], "Sensitive features are all-blank"),
        (
            [0],
            [0, 1],
            r"Sensitive features length \(1\) != y_true length \(2\)",
        ),
    ],
)
def test_validate_sensitive_features_rejects_invalid_inputs(
    sensitive,
    y_true,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_sensitive_features(sensitive, y_true, context="unit-test")


def test_validate_sensitive_features_accepts_aligned_values():
    sensitive = np.array([0, 1, 1])

    result = validate_sensitive_features(sensitive, [1, 0, 1], context="unit-test")

    assert result is sensitive


def test_materialize_dataset_supports_dataset_and_dataloader_inputs():
    class _Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return np.array([index, index + 1]), index

    class _DataLoader:
        batch_size = 2

        def __iter__(self):
            yield np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1])

    x_dataset, y_dataset = materialize_dataset(_Dataset())
    x_loader, y_loader = materialize_dataset(_DataLoader())

    assert x_dataset.shape == (2, 2)
    assert np.array_equal(y_dataset, np.array([0, 1]))
    assert x_loader.shape == (2, 2)
    assert np.array_equal(y_loader, np.array([0, 1]))


def test_materialize_dataset_rejects_unsupported_inputs():
    with pytest.raises(TypeError, match="Unsupported dataset-like input"):
        materialize_dataset(object())
