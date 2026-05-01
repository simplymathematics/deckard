"""Fairness-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
from typing import Dict

from . import ScorerConfig, ScorerDictConfig, safe_store

__all__ = [
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
    "DefaultFairnessConfig",
]


def _resolve_sensitive_features(data, y_true):
    if data is None:
        return None
    y_len = len(y_true)
    candidates = [
        getattr(data, "_sensitive_test", None),
        getattr(data, "_sensitive_train", None),
        getattr(data, "_sensitive_all", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if len(candidate) == y_len:
            return candidate
    return None


def fairness_demographic_parity_difference(y_true, y_pred, data=None, **kwargs):
    """Compute demographic parity difference for fairness-aware configurations."""
    try:
        from fairlearn.metrics import demographic_parity_difference
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        ) from exc

    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(data, y_true)
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")

    return float(
        demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    )


def fairness_equalized_odds_difference(y_true, y_pred, data=None, **kwargs):
    """Compute equalized odds difference for fairness-aware configurations."""
    try:
        from fairlearn.metrics import equalized_odds_difference
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        ) from exc

    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(data, y_true)
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")

    return float(
        equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    )


@dataclass
class DefaultFairnessConfig(ScorerDictConfig):
    """Default scorer set for fairness workflows."""

    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "demographic_parity_difference": ScorerConfig(
                score_name="demographic_parity_difference",
                score_function="deckard.score.fairness.fairness_demographic_parity_difference",
                greater_is_better=False,
            ),
            "equalized_odds_difference": ScorerConfig(
                score_name="equalized_odds_difference",
                score_function="deckard.score.fairness.fairness_equalized_odds_difference",
                greater_is_better=False,
            ),
        },
    )


safe_store(group="scorers", name="fairness", node=DefaultFairnessConfig)
