"""Cluster-scoring defaults for unsupervised model profiles."""

from dataclasses import dataclass, field

from .base import ScorerConfig, ScorerDictConfig, safe_store


@dataclass(eq=False, kw_only=True)
class DefaultClusterScorerDictConfig(ScorerDictConfig):
    """Default scorer profile for clustering models.

    These scorers compare predicted cluster assignments against available
    reference labels when present (for example synthetic benchmark datasets).

    Attributes:
        scoring_type: Scoring family identifier for model outputs.
        scorers: Metric configurations keyed by score name.
    """

    scoring_type: str = "model"
    scorers: dict[str, ScorerConfig] = field(
        default_factory=dict,
        metadata={
            "help": "Optional clustering scorer overrides keyed by metric name.",
        },
    )

    def __post_init__(self):
        if not self.scorers:
            self.scorers = {
                "adjusted_rand": ScorerConfig(
                    score_name="adjusted_rand",
                    score_function="sklearn.metrics.adjusted_rand_score",
                ),
                "normalized_mutual_info": ScorerConfig(
                    score_name="normalized_mutual_info",
                    score_function="sklearn.metrics.normalized_mutual_info_score",
                ),
                "homogeneity": ScorerConfig(
                    score_name="homogeneity",
                    score_function="sklearn.metrics.homogeneity_score",
                ),
                "completeness": ScorerConfig(
                    score_name="completeness",
                    score_function="sklearn.metrics.completeness_score",
                ),
                "v_measure": ScorerConfig(
                    score_name="v_measure",
                    score_function="sklearn.metrics.v_measure_score",
                ),
            }
        super().__post_init__()


safe_store(
    group="score",
    name="cluster",
    node={
        "_target_": "deckard.score.cluster.DefaultClusterScorerDictConfig",
    },
)


__all__ = ["DefaultClusterScorerDictConfig"]
