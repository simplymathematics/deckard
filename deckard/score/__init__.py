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
    DefaultFairnessConfig,
    fairness_demographic_parity_difference,
    fairness_equalized_odds_difference,
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


class DefaultClassifierDict:
    scorers = DefaultClassifierConfig()


class DefaultRegressorDict:
    scorers = DefaultRegressorConfig()


class DefaultFairnessDict:
    scorers = DefaultFairnessConfig()


class DefaultSurvivalDict:
    scorers = DefaultSurvivalConfig()


safe_store(group="scorers", name="fairness", node=DefaultFairnessConfig)
safe_store(group="scorers", name="survival", node=DefaultSurvivalConfig)


__all__ = [
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultClassifierConfig",
    "DefaultRegressorConfig",
    "DefaultFairnessConfig",
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
    "DefaultSurvivalDict",
    "build_scorer",
    "build_scorer_dict",
    "survival_concordance_score",
    "survival_aic_score",
    "survival_bic_score",
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
]
