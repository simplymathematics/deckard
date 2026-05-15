"""Scoring configuration exports and Hydra registrations."""

import logging

from .base import (  # noqa: F401
    DefaultModelScorerConfig,
    DefaultClassifierConfig,
    DefaultPytorchScorerConfig,
    DefaultPytorchClassifierConfig,
    DefaultPytorchRegressorConfig,
    DefaultRegressorConfig,
    ScorerConfig,
    ScorerDictConfig,
    build_scorer,
    build_scorer_dict,
)

from .attack import (  # noqa: E402
    AttackScorerConfig,
    FairlearnAttackScorerConfig,
    DefaultEvasionAttackScorerConfig,
    DefaultEvasionRegressionAttackScorerConfig,
    DefaultMembershipInferenceAttackScorerConfig,
    DefaultAttributeInferenceAttackScorerConfig,
    DefaultAttributeInferenceRegressionAttackScorerConfig,
)
from .data import (  # noqa: E402
    DefaultDataScorerConfig,
    DefaultDataClassificationConfig,
    DefaultDataRegressionConfig,
    data_num_classes_score,
    data_class_count_min_score,
    data_class_count_max_score,
    data_class_imbalance_ratio_score,
    data_mutual_information_mean_score,
    data_mutual_information_max_score,
    data_empirical_cdf_function_score,
)

logger = logging.getLogger(__name__)

survival_concordance_score = None
survival_aic_score = None
survival_bic_score = None


def _load_lifelines_score_symbol(symbol_name):
    from ..plugins.lifelines.score import (  # noqa: WPS433
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


if survival_concordance_score is None:

    def survival_concordance_score(*args, **kwargs):
        return _load_lifelines_score_symbol("survival_concordance_score")(
            *args,
            **kwargs,
        )


if survival_aic_score is None:

    def survival_aic_score(*args, **kwargs):
        return _load_lifelines_score_symbol("survival_aic_score")(
            *args,
            **kwargs,
        )


if survival_bic_score is None:

    def survival_bic_score(*args, **kwargs):
        return _load_lifelines_score_symbol("survival_bic_score")(
            *args,
            **kwargs,
        )

from .declarations import (  # noqa: E402
    SCORER_PLUGIN_MODEL_BASE,
    SCORER_PLUGIN_MODEL_CLASSIFIER,
    SCORER_PLUGIN_MODEL_REGRESSOR,
    SCORER_PLUGIN_DATA_BASE,
    SCORER_PLUGIN_DATA_CLASSIFIER,
    SCORER_PLUGIN_DATA_REGRESSOR,
)

try:
    from ..plugins.fairlearn.score import (  # noqa: E402
        DefaultFairlearnClassificationConfig,
        DefaultFairlearnRegressionConfig,
        DefaultFairlearnDataScorerConfig,
        FairlearnScoreDictConfig,
        fairness_demographic_parity_difference,
        fairness_equalized_odds_difference,
        fairness_group_mae_difference,
        fairness_group_mean_prediction_difference,
        fairness_group_mse_difference,
    )

    _ = (
        DefaultFairlearnClassificationConfig,
        DefaultFairlearnRegressionConfig,
        DefaultFairlearnDataScorerConfig,
        FairlearnScoreDictConfig,
        fairness_demographic_parity_difference,
        fairness_equalized_odds_difference,
        fairness_group_mae_difference,
        fairness_group_mean_prediction_difference,
        fairness_group_mse_difference,
    )
except ImportError:  # pragma: no cover - optional dependency
    DefaultFairlearnClassificationConfig = None
    DefaultFairlearnRegressionConfig = None
    DefaultFairlearnDataScorerConfig = None
    FairlearnScoreDictConfig = None
    fairness_demographic_parity_difference = None
    fairness_equalized_odds_difference = None
    fairness_group_mae_difference = None
    fairness_group_mean_prediction_difference = None
    fairness_group_mse_difference = None
    logger.debug("Fairlearn not found. Fairness score configs are unavailable.")


def _load_fairlearn_score_symbol(symbol_name):
    from ..plugins.fairlearn.score import (  # noqa: WPS433
        fairness_demographic_parity_difference as _fairness_demographic_parity_difference,
        fairness_equalized_odds_difference as _fairness_equalized_odds_difference,
        fairness_group_mae_difference as _fairness_group_mae_difference,
        fairness_group_mean_prediction_difference as _fairness_group_mean_prediction_difference,
        fairness_group_mse_difference as _fairness_group_mse_difference,
    )

    symbols = {
        "fairness_demographic_parity_difference": _fairness_demographic_parity_difference,
        "fairness_equalized_odds_difference": _fairness_equalized_odds_difference,
        "fairness_group_mae_difference": _fairness_group_mae_difference,
        "fairness_group_mean_prediction_difference": _fairness_group_mean_prediction_difference,
        "fairness_group_mse_difference": _fairness_group_mse_difference,
    }
    return symbols[symbol_name]


if fairness_demographic_parity_difference is None:

    def fairness_demographic_parity_difference(*args, **kwargs):
        return _load_fairlearn_score_symbol("fairness_demographic_parity_difference")(
            *args,
            **kwargs,
        )


if fairness_equalized_odds_difference is None:

    def fairness_equalized_odds_difference(*args, **kwargs):
        return _load_fairlearn_score_symbol("fairness_equalized_odds_difference")(
            *args,
            **kwargs,
        )


if fairness_group_mae_difference is None:

    def fairness_group_mae_difference(*args, **kwargs):
        return _load_fairlearn_score_symbol("fairness_group_mae_difference")(
            *args,
            **kwargs,
        )


if fairness_group_mean_prediction_difference is None:

    def fairness_group_mean_prediction_difference(*args, **kwargs):
        return _load_fairlearn_score_symbol(
            "fairness_group_mean_prediction_difference",
        )(*args, **kwargs)


if fairness_group_mse_difference is None:

    def fairness_group_mse_difference(*args, **kwargs):
        return _load_fairlearn_score_symbol("fairness_group_mse_difference")(
            *args,
            **kwargs,
        )

try:
    from ..plugins.anjana.score import (  # noqa: E402
        DefaultAnjanaScorerConfig,
        DefaultAnjanaDataScorerConfig,
        DefaultAnjanaModelScorerConfig,
        anjana_k_anonymity_score,
        anjana_l_diversity_score,
        anjana_t_closeness_score,
    )

    _ = (
        DefaultAnjanaScorerConfig,
        DefaultAnjanaDataScorerConfig,
        DefaultAnjanaModelScorerConfig,
        anjana_k_anonymity_score,
        anjana_l_diversity_score,
        anjana_t_closeness_score,
    )
except ImportError:  # pragma: no cover - optional dependency
    DefaultAnjanaScorerConfig = None
    DefaultAnjanaDataScorerConfig = None
    DefaultAnjanaModelScorerConfig = None
    anjana_k_anonymity_score = None
    anjana_l_diversity_score = None
    anjana_t_closeness_score = None
    logger.debug("Anjana not found. Anjana score configs are unavailable.")

try:
    from ..plugins.lifelines.score import (  # noqa: E402
        DefaultLifelinesConfig,
        survival_aic_score,
        survival_bic_score,
        survival_concordance_score,
    )

    _ = (
        DefaultLifelinesConfig,
        survival_aic_score,
        survival_bic_score,
        survival_concordance_score,
    )
except ImportError:  # pragma: no cover - optional dependency
    DefaultLifelinesConfig = None
    logger.debug("Lifelines not found. Survival score configs are unavailable.")

if "FairlearnScoreDictConfig" in globals() and FairlearnScoreDictConfig is not None:
    pass

if "DefaultLifelinesConfig" in globals() and DefaultLifelinesConfig is not None:
    pass


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
    "data_num_classes_score",
    "data_class_count_min_score",
    "data_class_count_max_score",
    "data_class_imbalance_ratio_score",
    "data_mutual_information_mean_score",
    "data_mutual_information_max_score",
    "data_empirical_cdf_function_score",
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "build_scorer",
    "build_scorer_dict",
]

if "FairlearnScoreDictConfig" in globals() and FairlearnScoreDictConfig is not None:
    __all__.extend(
        [
            "DefaultFairlearnClassificationConfig",
            "DefaultFairlearnScorerConfig",
            "DefaultFairlearnDataScorerConfig",
            "DefaultFairlearnRegressionConfig",
            "FairlearnScoreDictConfig",
            "fairness_demographic_parity_difference",
            "fairness_equalized_odds_difference",
            "fairness_group_mean_prediction_difference",
            "fairness_group_mae_difference",
            "fairness_group_mse_difference",
        ],
    )

if (
    "DefaultAnjanaDataScorerConfig" in globals()
    and DefaultAnjanaDataScorerConfig is not None
):
    __all__.extend(
        [
            "DefaultAnjanaScorerConfig",
            "DefaultAnjanaDataScorerConfig",
            "DefaultAnjanaModelScorerConfig",
            "anjana_k_anonymity_score",
            "anjana_l_diversity_score",
            "anjana_t_closeness_score",
        ],
    )

if "DefaultLifelinesConfig" in globals() and DefaultLifelinesConfig is not None:
    __all__.extend(
        [
            "DefaultLifelinesConfig",
            "survival_concordance_score",
            "survival_aic_score",
            "survival_bic_score",
        ],
    )
