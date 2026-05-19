"""Scoring configuration exports and lazy optional plugin symbol loading."""

from __future__ import annotations

import importlib.util

from .attack import (
    AttackScorerConfig,
    DefaultAttributeInferenceAttackScorerConfig,
    DefaultAttributeInferenceRegressionAttackScorerConfig,
    DefaultEvasionAttackScorerConfig,
    DefaultEvasionRegressionAttackScorerConfig,
    DefaultMembershipInferenceAttackScorerConfig,
    FairlearnAttackScorerConfig,
)
from .base import (
    DefaultClassifierConfig,
    DefaultModelScorerConfig,
    DefaultPytorchClassifierConfig,
    DefaultPytorchRegressorConfig,
    DefaultPytorchScorerConfig,
    DefaultRegressorConfig,
    ScorerConfig,
    ScorerDictConfig,
    build_scorer,
    build_scorer_dict,
)
from .data import (
    DefaultDataClassificationConfig,
    DefaultDataRegressionConfig,
    DefaultDataScorerConfig,
    DefaultPytorchDataScorerConfig,
    data_class_count_max_score,
    data_class_count_min_score,
    data_class_imbalance_ratio_score,
    data_empirical_cdf_function_score,
    data_mutual_information_max_score,
    data_mutual_information_mean_score,
    data_num_classes_score,
)


def _is_available(module_name: str) -> bool:
    """Return ``True`` when an optional dependency appears installed."""
    return importlib.util.find_spec(module_name) is not None


def _load_fairlearn_score_symbols() -> bool:
    try:
        from ..plugins.fairlearn.score import (
            DefaultFairlearnClassificationConfig,
            DefaultFairlearnDataScorerConfig,
            DefaultFairlearnRegressionConfig,
            DefaultFairlearnScorerConfig,
            FairlearnScoreDictConfig,
        )
    except Exception:  # pragma: no cover
        return False

    globals().update(
        {
            "DefaultFairlearnClassificationConfig": DefaultFairlearnClassificationConfig,
            "DefaultFairlearnDataScorerConfig": DefaultFairlearnDataScorerConfig,
            "DefaultFairlearnRegressionConfig": DefaultFairlearnRegressionConfig,
            "DefaultFairlearnScorerConfig": DefaultFairlearnScorerConfig,
            "FairlearnScoreDictConfig": FairlearnScoreDictConfig,
        },
    )
    return True


def _load_anjana_score_symbols() -> bool:
    try:
        from ..plugins.anjana.score import (
            DefaultAnjanaDataScorerConfig,
            DefaultAnjanaModelScorerConfig,
            DefaultAnjanaScorerConfig,
        )
    except Exception:  # pragma: no cover
        return False

    globals().update(
        {
            "DefaultAnjanaScorerConfig": DefaultAnjanaScorerConfig,
            "DefaultAnjanaDataScorerConfig": DefaultAnjanaDataScorerConfig,
            "DefaultAnjanaModelScorerConfig": DefaultAnjanaModelScorerConfig,
        },
    )
    return True


def _load_lifelines_score_symbols() -> bool:
    try:
        from ..plugins.lifelines.score import DefaultLifelinesConfig
    except Exception:  # pragma: no cover
        return False

    globals().update({"DefaultLifelinesConfig": DefaultLifelinesConfig})
    return True


def _load_fairlearn_score_symbol(symbol_name: str):
    from ..plugins.fairlearn.score import (
        fairness_demographic_parity_difference as _fairness_demographic_parity_difference,
        fairness_equalized_odds_difference as _fairness_equalized_odds_difference,
        fairness_group_mae_difference as _fairness_group_mae_difference,
        fairness_group_mean_prediction_difference as _fairness_group_mean_prediction_difference,
        fairness_group_mse_difference as _fairness_group_mse_difference,
    )

    symbols = {
        "fairness_demographic_parity_difference": _fairness_demographic_parity_difference,
        "fairness_equalized_odds_difference": _fairness_equalized_odds_difference,
        "fairness_group_mean_prediction_difference": _fairness_group_mean_prediction_difference,
        "fairness_group_mae_difference": _fairness_group_mae_difference,
        "fairness_group_mse_difference": _fairness_group_mse_difference,
    }
    return symbols[symbol_name]


def _load_anjana_score_symbol(symbol_name: str):
    from ..plugins.anjana.score import (
        anjana_k_anonymity_score as _anjana_k_anonymity_score,
        anjana_l_diversity_score as _anjana_l_diversity_score,
        anjana_t_closeness_score as _anjana_t_closeness_score,
    )

    symbols = {
        "anjana_k_anonymity_score": _anjana_k_anonymity_score,
        "anjana_l_diversity_score": _anjana_l_diversity_score,
        "anjana_t_closeness_score": _anjana_t_closeness_score,
    }
    return symbols[symbol_name]


def _load_lifelines_score_symbol(symbol_name: str):
    from ..plugins.lifelines.score import (
        survival_aic_score as _survival_aic_score,
        survival_bic_score as _survival_bic_score,
        survival_concordance_score as _survival_concordance_score,
    )

    symbols = {
        "survival_concordance_score": _survival_concordance_score,
        "survival_aic_score": _survival_aic_score,
        "survival_bic_score": _survival_bic_score,
    }
    return symbols[symbol_name]


def fairness_demographic_parity_difference(*args, **kwargs):
    return _load_fairlearn_score_symbol("fairness_demographic_parity_difference")(
        *args,
        **kwargs,
    )


def fairness_equalized_odds_difference(*args, **kwargs):
    return _load_fairlearn_score_symbol("fairness_equalized_odds_difference")(
        *args,
        **kwargs,
    )


def fairness_group_mean_prediction_difference(*args, **kwargs):
    return _load_fairlearn_score_symbol("fairness_group_mean_prediction_difference")(
        *args,
        **kwargs,
    )


def fairness_group_mae_difference(*args, **kwargs):
    return _load_fairlearn_score_symbol("fairness_group_mae_difference")(
        *args,
        **kwargs,
    )


def fairness_group_mse_difference(*args, **kwargs):
    return _load_fairlearn_score_symbol("fairness_group_mse_difference")(
        *args,
        **kwargs,
    )


def anjana_k_anonymity_score(*args, **kwargs):
    return _load_anjana_score_symbol("anjana_k_anonymity_score")(*args, **kwargs)


def anjana_l_diversity_score(*args, **kwargs):
    return _load_anjana_score_symbol("anjana_l_diversity_score")(*args, **kwargs)


def anjana_t_closeness_score(*args, **kwargs):
    return _load_anjana_score_symbol("anjana_t_closeness_score")(*args, **kwargs)


def survival_concordance_score(*args, **kwargs):
    return _load_lifelines_score_symbol("survival_concordance_score")(*args, **kwargs)


def survival_aic_score(*args, **kwargs):
    return _load_lifelines_score_symbol("survival_aic_score")(*args, **kwargs)


def survival_bic_score(*args, **kwargs):
    return _load_lifelines_score_symbol("survival_bic_score")(*args, **kwargs)


__all__ = [
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultModelScorerConfig",
    "DefaultClassifierConfig",
    "DefaultPytorchScorerConfig",
    "DefaultPytorchClassifierConfig",
    "DefaultPytorchRegressorConfig",
    "DefaultRegressorConfig",
    "AttackScorerConfig",
    "FairlearnAttackScorerConfig",
    "DefaultEvasionAttackScorerConfig",
    "DefaultEvasionRegressionAttackScorerConfig",
    "DefaultMembershipInferenceAttackScorerConfig",
    "DefaultAttributeInferenceAttackScorerConfig",
    "DefaultAttributeInferenceRegressionAttackScorerConfig",
    "DefaultDataScorerConfig",
    "DefaultDataClassificationConfig",
    "DefaultDataRegressionConfig",
    "DefaultPytorchDataScorerConfig",
    "data_num_classes_score",
    "data_class_count_min_score",
    "data_class_count_max_score",
    "data_class_imbalance_ratio_score",
    "data_mutual_information_mean_score",
    "data_mutual_information_max_score",
    "data_empirical_cdf_function_score",
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
    "fairness_group_mean_prediction_difference",
    "fairness_group_mae_difference",
    "fairness_group_mse_difference",
    "anjana_k_anonymity_score",
    "anjana_l_diversity_score",
    "anjana_t_closeness_score",
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "build_scorer",
    "build_scorer_dict",
]


if _is_available("fairlearn"):
    __all__.extend(
        [
            "DefaultFairlearnScorerConfig",
            "DefaultFairlearnClassificationConfig",
            "DefaultFairlearnRegressionConfig",
            "DefaultFairlearnDataScorerConfig",
            "FairlearnScoreDictConfig",
        ],
    )

if _is_available("pycanon"):
    __all__.extend(
        [
            "DefaultAnjanaScorerConfig",
            "DefaultAnjanaDataScorerConfig",
            "DefaultAnjanaModelScorerConfig",
        ],
    )

if _is_available("lifelines"):
    __all__.append("DefaultLifelinesConfig")


def __getattr__(name: str):
    fairlearn_symbols = {
        "DefaultFairlearnScorerConfig",
        "DefaultFairlearnClassificationConfig",
        "DefaultFairlearnRegressionConfig",
        "DefaultFairlearnDataScorerConfig",
        "FairlearnScoreDictConfig",
    }
    anjana_symbols = {
        "DefaultAnjanaScorerConfig",
        "DefaultAnjanaDataScorerConfig",
        "DefaultAnjanaModelScorerConfig",
    }
    lifelines_symbols = {"DefaultLifelinesConfig"}

    if name in fairlearn_symbols and _load_fairlearn_score_symbols():
        return globals()[name]
    if name in anjana_symbols and _load_anjana_score_symbols():
        return globals()[name]
    if name in lifelines_symbols and _load_lifelines_score_symbols():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
