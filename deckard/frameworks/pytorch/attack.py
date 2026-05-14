from dataclasses import dataclass

import numpy as np
import pandas as pd

from art.config import ART_NUMPY_DTYPE
from ...score.attack import AttackScorerConfig  # noqa: F401
from ...attack.base import AttackConfig
from .torch_utils import is_tensor, tensor_to_numpy


@dataclass(eq=False)
class PytorchAttackConfig(AttackConfig):
    """Attack config variant that preserves torch tensors for attack execution.

    Scoring still uses the base normalization path (numpy arrays), but attack
    method calls keep tensor inputs when the upstream data pipeline is torch-native.
    """

    def _prepare_features_for_attack(self, value):
        if is_tensor(value):
            return value
        if isinstance(value, pd.DataFrame):
            return value.values.astype(ART_NUMPY_DTYPE)
        if isinstance(value, pd.Series):
            return value.values.astype(ART_NUMPY_DTYPE)
        return value

    def _prepare_labels_for_attack(self, value):
        if is_tensor(value):
            return value
        if isinstance(value, pd.DataFrame):
            return value.values
        if isinstance(value, pd.Series):
            return value.values
        return value

    def _prepare_features_for_art(self, value):
        """Torch-aware conversion used only at ART model/attack call boundaries."""
        if is_tensor(value):
            return tensor_to_numpy(value, dtype=ART_NUMPY_DTYPE)
        if isinstance(value, pd.DataFrame):
            return value.values.astype(ART_NUMPY_DTYPE)
        if isinstance(value, pd.Series):
            return value.values.astype(ART_NUMPY_DTYPE)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            return value.astype(ART_NUMPY_DTYPE, copy=False)
        return np.asarray(value)

    def _prepare_labels_for_art(self, value):
        """Torch-aware label conversion used only where ART requires numpy labels."""
        if is_tensor(value):
            return tensor_to_numpy(value)
        if isinstance(value, pd.DataFrame):
            return value.values
        if isinstance(value, pd.Series):
            return value.values
        return value
