"""Scoring configuration exports and Hydra registrations."""

from .base import (  # noqa: F401
    DefaultClassifierConfig,
    DefaultRegressorConfig,
    ScorerConfig,
    ScorerDictConfig,
    build_scorer,
    build_scorer_dict,
    safe_store,
)


from .fairness import (  # noqa: E402
    DefaultFairnessClassificationConfig,
    DefaultFairnessConfig,
    DefaultFairnessRegressionConfig,
    FairnessScoreDictConfig,
    fairness_demographic_parity_difference,
    fairness_equalized_odds_difference,
    fairness_group_mae_difference,
    fairness_group_mean_prediction_difference,
    fairness_group_mse_difference,
)
from .survival import (  # noqa: E402
    DefaultSurvivalConfig,
    survival_aic_score,
    survival_bic_score,
    survival_concordance_score,
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


class DefaultClassifierDict:
    scorers = DefaultClassifierConfig()


class DefaultRegressorDict:
    scorers = DefaultRegressorConfig()


class DefaultFairnessDict:
    scorers = DefaultFairnessConfig()


class DefaultFairnessClassificationDict:
    scorers = DefaultFairnessClassificationConfig()


class DefaultFairnessRegressionDict:
    scorers = DefaultFairnessRegressionConfig()


class DefaultSurvivalDict:
    scorers = DefaultSurvivalConfig()


class DefaultDataClassificationDict:
    scorers = DefaultDataClassificationConfig()


class DefaultDataRegressionDict:
    scorers = DefaultDataRegressionConfig()


safe_store(group="score", name="fairness-classification", node=DefaultFairnessClassificationConfig)
safe_store(group="score", name="fairness-regression", node=DefaultFairnessRegressionConfig)
safe_store(group="score", name="survival", node=DefaultSurvivalConfig)


__all__ = [
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultClassifierConfig",
    "DefaultRegressorConfig",
    "DefaultFairnessClassificationConfig",
    "DefaultFairnessConfig",
    "DefaultFairnessRegressionConfig",
    "FairnessScoreDictConfig",
    "DefaultSurvivalConfig",
    "AttackScorerConfig",
    "DefaultEvasionAttackScorerConfig",
    "DefaultEvasionRegressionAttackScorerConfig",
    "DefaultMembershipInferenceAttackScorerConfig",
    "DefaultAttributeInferenceAttackScorerConfig",
    "DefaultAttributeInferenceRegressionAttackScorerConfig",
    "DefaultClassifierDict",
    "DefaultRegressorDict",
    "DefaultFairnessDict",
    "DefaultFairnessClassificationDict",
    "DefaultFairnessRegressionDict",
    "DefaultSurvivalDict",
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
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
    "fairness_group_mean_prediction_difference",
    "fairness_group_mae_difference",
    "fairness_group_mse_difference",
]
