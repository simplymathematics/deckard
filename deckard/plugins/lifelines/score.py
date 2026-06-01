"""Survival-specific scoring helpers and default scorer configuration."""

import math
from dataclasses import dataclass, field
from typing import Any, Union

from ...score.base import ScorerConfig, ScorerDictConfig, safe_store

__all__ = [
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "DefaultLifelinesConfig",
]


def survival_concordance_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Return survival concordance from a fitted lifelines model when available."""
    _ = y_true, kwargs
    if hasattr(y_pred, "concordance_index_"):
        return float(y_pred.concordance_index_)
    raise ValueError(
        "y_pred must be a fitted survival model with concordance_index_",
    )


def survival_aic_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Return survival AIC from a fitted lifelines model when available."""
    _ = y_true, kwargs
    if hasattr(y_pred, "AIC_"):
        return float(y_pred.AIC_)
    if hasattr(y_pred, "partial_AIC_"):
        return float(y_pred.partial_AIC_)
    raise ValueError("y_pred must expose AIC_ or partial_AIC_")


def survival_bic_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Return survival BIC from a fitted lifelines model when available."""
    if hasattr(y_pred, "BIC_"):
        return float(y_pred.BIC_)

    n = kwargs.get("n_samples")
    if n is None and y_true is not None:
        try:
            n = len(y_true)
        except TypeError:
            n = None

    if hasattr(y_pred, "log_likelihood_"):
        k = None
        if hasattr(y_pred, "params_"):
            k = len(getattr(y_pred, "params_"))
        elif hasattr(y_pred, "params") and callable(getattr(y_pred, "params")):
            k = len(y_pred.params())
        if k is not None and n is not None and n > 0:
            return float(-2.0 * float(y_pred.log_likelihood_) + float(k) * math.log(n))

    raise ValueError("y_pred must expose BIC_ or enough information to compute BIC")


@dataclass(eq=False, kw_only=True)
class DefaultLifelinesConfig(ScorerDictConfig):
    """Default scorer set for survival workflows.

    This config composes survival-model ``ScorerConfig`` objects into one
    ``ScorerDictConfig`` that emits a ``ScoreDict`` for fitted lifelines
    models. The default scorer set covers concordance, AIC, and BIC by
    inspecting standard lifelines fitter attributes.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    scorers: dict[str, Union[ScorerConfig, dict[str, Any]]] = field(default_factory=lambda : {'concordance': ScorerConfig(score_name='concordance', score_function='deckard.plugins.lifelines.score.survival_concordance_score'), 'aic': ScorerConfig(score_name='aic', score_function='deckard.plugins.lifelines.score.survival_aic_score', greater_is_better=False), 'bic': ScorerConfig(score_name='bic', score_function='deckard.plugins.lifelines.score.survival_bic_score', greater_is_better=False)}, metadata={'help': 'Configuration field: scorers.'})


safe_store(group="score", name="lifelines", node=DefaultLifelinesConfig)
