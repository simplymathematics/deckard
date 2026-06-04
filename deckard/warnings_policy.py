"""Shared import-time warning policy for deckard."""

from __future__ import annotations

import warnings
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

_WARNING_POLICY_APPLIED = False


def apply_warning_policy() -> None:
    """Apply deckard's canonical warning filters once per interpreter session."""
    global _WARNING_POLICY_APPLIED
    if _WARNING_POLICY_APPLIED:
        return

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    warnings.filterwarnings("ignore", module=r"^sklearn(\.|$)")
    warnings.filterwarnings("ignore", module=r"^art(\.|$)")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"PyTorch not found\. Not importing DeepZ or Interval Bound Propagation functionality",
    )
    _WARNING_POLICY_APPLIED = True
