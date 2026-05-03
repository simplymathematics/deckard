"""Scoring configuration exports and Hydra registrations."""

import logging

from .base import (  # noqa: F401
    DefaultClassifierConfig,
    DefaultPytorchClassifierConfig,
    DefaultPytorchRegressorConfig,
    DefaultRegressorConfig,
    ScorerConfig,
    ScorerDictConfig,
    build_scorer,
    build_scorer_dict,
    safe_store,
)

from .attack import (  # noqa: E402
    AttackScorerConfig,
    DefaultEvasionAttackScorerConfig,
    DefaultEvasionRegressionAttackScorerConfig,
    DefaultMembershipInferenceAttackScorerConfig,
    DefaultAttributeInferenceAttackScorerConfig,
    DefaultAttributeInferenceRegressionAttackScorerConfig,
)
from .data import (  # noqa: E402
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

from .declarations import (  # noqa: E402
    DefaultClassifierDict,
    DefaultDataClassificationDict,
    DefaultDataRegressionDict,
    DefaultRegressorDict,
)

try:
    from .fairness import (  # noqa: E402
        DefaultFairlearnClassificationConfig,
        DefaultFairlearnConfig,
        DefaultFairlearnRegressionConfig,
        FairlearnScoreDictConfig,
        fairness_demographic_parity_difference,
        fairness_equalized_odds_difference,
        fairness_group_mae_difference,
        fairness_group_mean_prediction_difference,
        fairness_group_mse_difference,
    )
except ImportError:  # pragma: no cover - optional dependency
    logger.debug("Fairlearn not found. Fairness score configs are unavailable.")

try:
    from .anjana import (  # noqa: E402
        DefaultAnjanaDataScoreConfig,
        DefaultAnjanaModelScoreConfig,
        anjana_k_anonymity_score,
        anjana_l_diversity_score,
        anjana_t_closeness_score,
    )
except ImportError:  # pragma: no cover - optional dependency
    logger.debug("Anjana not found. Anjana score configs are unavailable.")

try:
    from .survival import (  # noqa: E402
        DefaultLifelinesConfig,
        survival_aic_score,
        survival_bic_score,
        survival_concordance_score,
    )
except ImportError:  # pragma: no cover - optional dependency
    logger.debug("Lifelines not found. Survival score configs are unavailable.")

if "DefaultFairlearnConfig" in globals():
    from .declarations_fairness import (  # noqa: E402
        DefaultFairlearnClassificationDict,
        DefaultFairlearnDict,
        DefaultFairlearnRegressionDict,
    )

if "DefaultLifelinesConfig" in globals():
    from .declarations_survival import DefaultLifelinesDict  # noqa: E402


__all__ = [
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultClassifierConfig",
    "DefaultPytorchClassifierConfig",
    "DefaultPytorchRegressorConfig",
    "DefaultRegressorConfig",
    "AttackScorerConfig",
    "DefaultEvasionAttackScorerConfig",
    "DefaultEvasionRegressionAttackScorerConfig",
    "DefaultMembershipInferenceAttackScorerConfig",
    "DefaultAttributeInferenceAttackScorerConfig",
    "DefaultAttributeInferenceRegressionAttackScorerConfig",
    "DefaultClassifierDict",
    "DefaultRegressorDict",
    "DefaultDataClassificationConfig",
    "DefaultDataRegressionConfig",
    "DefaultDataClassificationDict",
    "DefaultDataRegressionDict",
    "data_num_classes_score",
    "data_class_count_min_score",
    "data_class_count_max_score",
    "data_class_imbalance_ratio_score",
    "data_mutual_information_mean_score",
    "data_mutual_information_max_score",
    "data_empirical_cdf_function_score",
    "build_scorer",
    "build_scorer_dict",
]

if "DefaultFairlearnConfig" in globals():
    __all__.extend(
        [
            "DefaultFairlearnClassificationConfig",
            "DefaultFairlearnConfig",
            "DefaultFairlearnRegressionConfig",
            "FairlearnScoreDictConfig",
            "DefaultFairlearnDict",
            "DefaultFairlearnClassificationDict",
            "DefaultFairlearnRegressionDict",
            "fairness_demographic_parity_difference",
            "fairness_equalized_odds_difference",
            "fairness_group_mean_prediction_difference",
            "fairness_group_mae_difference",
            "fairness_group_mse_difference",
        ]
    )

if "DefaultAnjanaDataScoreConfig" in globals():
    __all__.extend(
        [
            "DefaultAnjanaDataScoreConfig",
            "DefaultAnjanaModelScoreConfig",
            "anjana_k_anonymity_score",
            "anjana_l_diversity_score",
            "anjana_t_closeness_score",
        ]
    )

if "DefaultLifelinesConfig" in globals():
    __all__.extend(
        [
            "DefaultLifelinesConfig",
            "DefaultLifelinesDict",
            "survival_concordance_score",
            "survival_aic_score",
            "survival_bic_score",
        ]
    )
