"""Survival-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
import math
from typing import Dict

from .base import ScorerConfig, ScorerDictConfig, safe_store

__all__ = [
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "DefaultLifelinesConfig",
]


def survival_concordance_score(y_true, y_pred, **kwargs):
    """Return survival concordance from a fitted lifelines model when available."""
    if hasattr(y_pred, "concordance_index_"):
        return float(y_pred.concordance_index_)
    raise ValueError(
        "y_pred must be a fitted survival model with concordance_index_"
    )


def survival_aic_score(y_true, y_pred, **kwargs):
    """Return survival AIC from a fitted lifelines model when available."""
    if hasattr(y_pred, "AIC_"):
        return float(y_pred.AIC_)
    if hasattr(y_pred, "partial_AIC_"):
        return float(y_pred.partial_AIC_)
    if hasattr(y_pred, "log_likelihood_"):
        k = None
        if hasattr(y_pred, "params_"):
            k = len(getattr(y_pred, "params_"))
        elif hasattr(y_pred, "params") and callable(getattr(y_pred, "params")):
            k = len(y_pred.params())
        if k is not None:
            return float(-2.0 * float(y_pred.log_likelihood_) + 2.0 * float(k))
    raise ValueError(
        "y_pred must expose AIC_ or enough information to compute AIC"
    )


def survival_bic_score(y_true, y_pred, **kwargs):
    """Return survival BIC from a fitted lifelines model when available."""
    if hasattr(y_pred, "BIC_"):
        return float(y_pred.BIC_)

    n = kwargs.get("n_samples")
    if n is None and y_true is not None:
        try:
            n = len(y_true)
        except TypeError:
            n = None

    if n is not None and hasattr(y_pred, "log_likelihood_"):
        k = None
        if hasattr(y_pred, "params_"):
            k = len(getattr(y_pred, "params_"))
        elif hasattr(y_pred, "params") and callable(getattr(y_pred, "params")):
            k = len(y_pred.params())
        if k is not None and n > 0:
            return float(
                -2.0 * float(y_pred.log_likelihood_) + float(k) * math.log(n)
            )

    raise ValueError(
        "y_pred must expose BIC_ or enough information to compute BIC"
    )


@dataclass(eq=False)
class DefaultLifelinesConfig(ScorerDictConfig):
    """Default scorer set for survival workflows."""

    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "concordance": ScorerConfig(
                score_name="concordance",
                score_function="deckard.score.survival.survival_concordance_score",
            ),
            "aic": ScorerConfig(
                score_name="aic",
                score_function="deckard.score.survival.survival_aic_score",
                greater_is_better=False,
            ),
            "bic": ScorerConfig(
                score_name="bic",
                score_function="deckard.score.survival.survival_bic_score",
                greater_is_better=False,
            ),
        },
    )


safe_store(group="score", name="lifelines", node=DefaultLifelinesConfig)
