from dataclasses import dataclass

import pandas as pd

from art.config import ART_NUMPY_DTYPE
from .base import AttackConfig
from .torch_utils import is_tensor


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
