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
    if hasattr(y_pred, "concordance_index_"):
        return float(y_pred.concordance_index_)
    raise ValueError(
        "y_pred must be a fitted survival model with concordance_index_",
    )


def survival_aic_score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
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
        "y_pred must expose AIC_ or enough information to compute AIC",
    )


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

    if n is not None and hasattr(y_pred, "log_likelihood_"):
        k = None
        if hasattr(y_pred, "params_"):
            k = len(getattr(y_pred, "params_"))
        elif hasattr(y_pred, "params") and callable(getattr(y_pred, "params")):
            k = len(y_pred.params())
        if k is not None and n > 0:
            return float(
                -2.0 * float(y_pred.log_likelihood_) + float(k) * math.log(n),
            )

    raise ValueError(
        "y_pred must expose BIC_ or enough information to compute BIC",
    )


@dataclass(eq=False, kw_only=True)
class DefaultLifelinesConfig(ScorerDictConfig):
    """Default scorer set for survival workflows.

    Initialization parameters
    -------------------------
    scorers : dict[str, ScorerConfig | dict[str, Any]]
        Named scorer configurations extracted from fitted lifelines survival models.
        Default set includes concordance, AIC, and BIC metrics.

    Runtime parameters
    -------------------
    y_pred : Any
        Fitted lifelines survival model instance (e.g., KaplanMeierFitter,
        CoxPHFitter, WeibullAFTFitter). Must expose relevant survival metrics
        as attributes (concordance_index_, AIC_, BIC_, log_likelihood_).
    y_true : Any
        Optional event data or durations (used for BIC calculation when
        n_samples is not explicitly provided).
    n_samples : int
        Optional sample count for BIC computation. If not provided, inferred
        from y_true length.

    Parameter layers
    ----------------
    1. Model introspection: Scorers extract fitted model attributes
    2. Information criteria: AIC and BIC support model comparison
    3. Predictive performance: Concordance measures prediction accuracy

    Family-specific parameter semantics
    -----------------------------------
    Survival scorers operate on fitted lifelines models:

    - **concordance_index_**: C-statistic measuring ranking accuracy of predictions.
    - **AIC_** / **partial_AIC_**: Akaike Information Criterion for model comparison.
    - **BIC_**: Bayesian Information Criterion (computed from log_likelihood_ if BIC_ unavailable).

    Plugin pattern
    --------------
    This scorer inherits from ``_ScorerMixin`` semantics through ``ScorerDictConfig``.
    Plugins registered via ``ScorerTypePlugin`` contribute mixin-based runtime context
    for survival-specific dispatch (e.g., ``scoring_subtype: "survival"`` routes to this scorer).
    """

    scorers: dict[str, Union[ScorerConfig, dict[str, Any]]] = field(
        default_factory=lambda: {
            "concordance": ScorerConfig(
                score_name="concordance",
                score_function="deckard.plugins.lifelines.score.survival_concordance_score",
            ),
            "aic": ScorerConfig(
                score_name="aic",
                score_function="deckard.plugins.lifelines.score.survival_aic_score",
                greater_is_better=False,
            ),
            "bic": ScorerConfig(
                score_name="bic",
                score_function="deckard.plugins.lifelines.score.survival_bic_score",
                greater_is_better=False,
            ),
        },
    )


safe_store(group="score", name="lifelines", node=DefaultLifelinesConfig)
