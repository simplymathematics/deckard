"""Optional torch helpers for scoring.

This module isolates torch-specific runtime conversions so the core scoring
module can stay importable when torch is not installed.

It also provides lightweight dataset/dataloader inspection utilities used by
the plot layer (``deckard.plot.yellowbrick_plots``) so they are available to
any module without importing plotting dependencies.
"""

from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Subset as TorchSubset

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    TorchSubset = None
    HAS_TORCH = False


def is_torch_tensor(value: Any) -> bool:
    return HAS_TORCH and isinstance(value, torch.Tensor)


def to_numpy_if_torch(value: Any) -> Any:
    """Recursively convert torch tensors to CPU numpy arrays."""
    if is_torch_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, list):
        return [to_numpy_if_torch(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_numpy_if_torch(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# Dataset / DataLoader inspection helpers
# ---------------------------------------------------------------------------


def to_numpy(value: Any) -> np.ndarray:
    """Convert any tensor-like, array-like, or scalar to a numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if HAS_TORCH:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def is_dataloader_like(obj: Any) -> bool:
    """Return True if *obj* looks like a DataLoader (has ``__iter__`` and ``batch_size``)."""
    return hasattr(obj, "__iter__") and hasattr(obj, "batch_size")


def is_dataset_like(obj: Any) -> bool:
    """Return True if *obj* looks like a map-style Dataset (has ``__len__`` and ``__getitem__``)."""
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


def get_dataset_shape(obj: Any) -> tuple:
    """Return the logical ``(N, *feature_dims)`` shape of a dataset or array.

    Works with numpy/torch arrays, ``torch.utils.data.Subset``,
    ``TensorDataset``-like objects, map-style datasets, and DataLoaders.
    Raises ``AttributeError`` when the shape cannot be determined.
    """
    if hasattr(obj, "shape"):
        return tuple(obj.shape)
    if hasattr(obj, "dataset") and hasattr(obj.dataset, "shape"):
        shape = obj.dataset.shape
        if hasattr(obj, "indices") and len(shape) > 0:
            return (len(obj.indices), *shape[1:])
        return shape
    if TorchSubset is not None and isinstance(obj, TorchSubset):
        base = obj.dataset
        # Recursive resolution via the same function
        try:
            shape = get_dataset_shape(base)
            if len(shape) > 0:
                return (len(obj.indices), *shape[1:])
            return shape
        except AttributeError:
            pass
        # TensorDataset-like (.tensors)
        if hasattr(base, "tensors") and base.tensors:
            tensor = base.tensors[0]
            if hasattr(tensor, "shape"):
                shape = tuple(tensor.shape)
                if len(shape) > 0:
                    return (len(obj.indices), *shape[1:])
                return shape
        if len(obj) > 0:
            first = obj[0]
            first_x = (
                first[0]
                if isinstance(first, (tuple, list)) and len(first) > 0
                else first
            )
            sample_shape = getattr(to_numpy(first_x), "shape", ())
            if len(sample_shape) > 0:
                return (len(obj), *sample_shape)
            return (len(obj),)
    if is_dataset_like(obj) and len(obj) > 0:
        first = obj[0]
        first_x = (
            first[0] if isinstance(first, (tuple, list)) and len(first) > 0 else first
        )
        sample_shape = getattr(to_numpy(first_x), "shape", ())
        if len(sample_shape) > 0:
            return (len(obj), *sample_shape)
        return (len(obj),)
    raise AttributeError(f"{type(obj).__name__} has no determinable shape")


# ---------------------------------------------------------------------------
# Fairness – sensitive-feature helpers shared by data, model, and score modules
# ---------------------------------------------------------------------------

# Canonical mapping: scoring mode -> sensitive-feature attribute name on data.
_SENSITIVE_ATTR: dict = {
    "train": "_sensitive_train",
    "test": "_sensitive_test",
    "attack": "_sensitive_test",  # attack uses test-split features
    "val": "_sensitive_val",
    "attack-val": "_sensitive_val",
    "all": "_sensitive_all",
    "pre-sample": "_sensitive_all",  # pre-sample falls back to all
}

# Canonical mapping: scoring mode -> data-split attribute names for (X, y).
_SPLIT_ATTRS: dict = {
    "train": ("X_train", "y_train"),
    "test": ("X_test", "y_test"),
    "attack": ("X_test", "y_test"),
    "val": ("X_val", "y_val"),
    "attack-val": ("X_val", "y_val"),
    "all": ("_X", "_y"),
    "pre-sample": ("_X", "_y"),
}


def resolve_sensitive_features(data: Any, mode: str) -> Optional[Any]:
    """Return the sensitive-feature array for *data* at *mode*, or ``None``.

    This is the single canonical lookup used by the score, data, and model
    fairness modules.  The mapping lives in ``_SENSITIVE_ATTR``.
    """
    if data is None:
        return None
    attr = _SENSITIVE_ATTR.get(mode)
    if attr is None:
        raise ValueError(
            f"Unsupported fairness scoring mode: '{mode}'. "
            f"Expected one of {list(_SENSITIVE_ATTR)}",
        )
    return getattr(data, attr, None)


def resolve_split_arrays(data: Any, mode: str) -> Tuple[Any, Any]:
    """Return the (X, y) arrays for *data* at *mode*.

    Uses ``_SPLIT_ATTRS`` to locate the right attribute names, then calls
    ``coerce_to_numpy`` on each so callers always receive plain numpy arrays
    or ``None``.
    """
    if data is None:
        return None, None
    x_attr, y_attr = _SPLIT_ATTRS.get(mode, ("X_test", "y_test"))
    return getattr(data, x_attr, None), getattr(data, y_attr, None)


def coerce_to_numpy(value: Any, dtype: Optional[Any] = None) -> Optional[np.ndarray]:
    """Best-effort conversion of any array-like to a plain numpy array.

    Handles torch tensors, objects with ``.numpy()``, ``.detach().cpu()``,
    and plain Python lists/arrays.  Returns ``None`` when *value* is ``None``.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(dtype) if dtype is not None else value
    if HAS_TORCH and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return arr.astype(dtype) if dtype is not None else arr
    if hasattr(value, "numpy"):
        arr = value.numpy()
        return arr.astype(dtype) if dtype is not None else arr
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
        return arr.astype(dtype) if dtype is not None else arr
    return np.asarray(value, dtype=dtype)


def validate_sensitive_features(sensitive: Any, y_true: Any, context: str) -> Any:
    """Validate that *sensitive* is a non-empty array aligned with *y_true*.

    Returns *sensitive* unchanged when valid.  Raises ``ValueError`` on any
    of the common failure modes (None, empty, all-null, wrong length).
    """
    import pandas as pd  # local import to keep torch module lightweight

    if sensitive is None:
        raise ValueError(
            f"Sensitive features are None during {context}; "
            "ensure the dataset exposes a '_sensitive' attribute.",
        )
    s = pd.Series(sensitive)
    if len(s) == 0:
        raise ValueError(f"Sensitive features are empty during {context}")
    if s.dropna().empty:
        raise ValueError(f"Sensitive features are all-null during {context}")
    if s.astype(str).str.strip().eq("").all():
        raise ValueError(f"Sensitive features are all-blank during {context}")
    if y_true is not None and hasattr(y_true, "__len__") and len(s) != len(y_true):
        raise ValueError(
            f"Sensitive features length ({len(s)}) != y_true length ({len(y_true)}) "
            f"during {context}",
        )
    return sensitive


def materialize_dataset(dataset_obj: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Materialise a Dataset or DataLoader into ``(X, y)`` numpy arrays.

    Iterates over every sample/batch and stacks feature rows.  Returns
    ``(X_array, y_array)`` where *y_array* is ``None`` when no labels are
    present.
    """
    x_rows = []
    y_rows = []

    if is_dataloader_like(dataset_obj):
        iterator = dataset_obj
    elif is_dataset_like(dataset_obj):
        iterator = (dataset_obj[i] for i in range(len(dataset_obj)))
    else:
        raise TypeError(f"Unsupported dataset-like input: {type(dataset_obj)}")

    for sample in iterator:
        if not isinstance(sample, (tuple, list)) or len(sample) < 1:
            continue
        # Handle batches from DataLoaders: flatten to individual samples
        x_batch = to_numpy(sample[0])
        if x_batch.ndim > 1:  # Batch tensor: shape (batch_size, *features)
            for i in range(x_batch.shape[0]):
                x_rows.append(x_batch[i])
        else:  # Single sample: shape (*features)
            x_rows.append(x_batch)

        if len(sample) >= 2:
            y_batch = np.asarray(to_numpy(sample[1])).reshape(-1)
            # Extend y_rows with individual labels from batch
            if y_batch.ndim > 1:  # Batch of labels (unlikely but handle it)
                y_rows.extend(y_batch.flatten().tolist())
            else:  # 1D array of labels
                y_rows.extend(y_batch.tolist())

    X = np.asarray(x_rows) if x_rows else np.empty((0,))
    y = np.asarray(y_rows) if y_rows else None
    return X, y
