"""Scoring configuration exports and Hydra registrations."""

import logging

from .base import (  # noqa: F401
    DefaultModelScoreConfig,
    DefaultClassifierConfig,
    DefaultPytorchScoreConfig,
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
    DefaultEvasionScoreConfig,
    DefaultEvasionAttackScorerConfig,
    DefaultEvasionRegressionAttackScorerConfig,
    DefaultMembershipInferenceScoreConfig,
    DefaultMembershipInferenceAttackScorerConfig,
    DefaultAttributeInferenceScoreConfig,
    DefaultAttributeInferenceAttackScorerConfig,
    DefaultAttributeInferenceRegressionAttackScorerConfig,
)
from .data import (  # noqa: E402
    DefaultDataScoreConfig,
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
        DefaultFairlearnScoreDictConfig,
        DefaultFairlearnRegressionConfig,
        FairlearnScoreDictConfig,
        fairness_demographic_parity_difference,
        fairness_equalized_odds_difference,
        fairness_group_mae_difference,
        fairness_group_mean_prediction_difference,
        fairness_group_mse_difference,
    )

    _ = (
        DefaultFairlearnClassificationConfig,
        DefaultFairlearnScoreDictConfig,
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
        DefaultAnjanaScoreConfig,
        DefaultAnjanaDataScoreConfig,
        DefaultAnjanaModelScoreConfig,
        anjana_k_anonymity_score,
        anjana_l_diversity_score,
        anjana_t_closeness_score,
    )

    _ = (
        DefaultAnjanaScoreConfig,
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

    _ = (
        DefaultLifelinesConfig,
        survival_aic_score,
        survival_bic_score,
        survival_concordance_score,
    )
except ImportError:  # pragma: no cover - optional dependency
    logger.debug("Lifelines not found. Survival score configs are unavailable.")

if "DefaultFairlearnScoreDictConfig" in globals():
    from .declarations_fairness import (  # noqa: E402
        DefaultFairlearnClassificationDict,
        DefaultFairlearnScoreDict,
        DefaultFairlearnRegressionDict,
    )

    _ = (
        DefaultFairlearnClassificationDict,
        DefaultFairlearnScoreDict,
        DefaultFairlearnRegressionDict,
    )

if "DefaultLifelinesConfig" in globals():
    from .declarations_survival import DefaultLifelinesDict  # noqa: E402

    _ = DefaultLifelinesDict


__all__ = [
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultModelScoreConfig",
    "DefaultClassifierConfig",
    "DefaultPytorchScoreConfig",
    "DefaultPytorchClassifierConfig",
    "DefaultPytorchRegressorConfig",
    "DefaultRegressorConfig",
    "AttackScorerConfig",
    "FairlearnAttackScorerConfig",
    "DefaultEvasionScoreConfig",
    "DefaultEvasionAttackScorerConfig",
    "DefaultEvasionRegressionAttackScorerConfig",
    "DefaultMembershipInferenceScoreConfig",
    "DefaultMembershipInferenceAttackScorerConfig",
    "DefaultAttributeInferenceScoreConfig",
    "DefaultAttributeInferenceAttackScorerConfig",
    "DefaultAttributeInferenceRegressionAttackScorerConfig",
    "DefaultDataScoreConfig",
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

if "DefaultFairlearnScoreDictConfig" in globals():
    __all__.extend(
        [
            "DefaultFairlearnClassificationConfig",
            "DefaultFairlearnScoreDictConfig",
            "DefaultFairlearnScoreDictConfig",
            "DefaultFairlearnRegressionConfig",
            "FairlearnScoreDictConfig",
            "DefaultFairlearnScoreDict",
            "DefaultFairlearnClassificationDict",
            "DefaultFairlearnRegressionDict",
            "fairness_demographic_parity_difference",
            "fairness_equalized_odds_difference",
            "fairness_group_mean_prediction_difference",
            "fairness_group_mae_difference",
            "fairness_group_mse_difference",
        ],
    )

if "DefaultAnjanaDataScoreConfig" in globals():
    __all__.extend(
        [
            "DefaultAnjanaScoreConfig",
            "DefaultAnjanaDataScoreConfig",
            "DefaultAnjanaModelScoreConfig",
            "anjana_k_anonymity_score",
            "anjana_l_diversity_score",
            "anjana_t_closeness_score",
        ],
    )

if "DefaultLifelinesConfig" in globals():
    __all__.extend(
        [
            "DefaultLifelinesConfig",
            "DefaultLifelinesDict",
            "survival_concordance_score",
            "survival_aic_score",
            "survival_bic_score",
        ],
    )
