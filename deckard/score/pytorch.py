"""Optional torch helpers for scoring.

This module isolates torch-specific runtime conversions so the core scoring
module can stay importable when torch is not installed.
"""

from typing import Any

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
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
