from deckard.score.base import ScorerConfig
from deckard.score.cluster import DefaultClusterScorerDictConfig


def test_default_cluster_scorer_dict_populates_expected_metrics():
    cfg = DefaultClusterScorerDictConfig()

    assert cfg.scoring_type == "model"
    assert set(cfg.scorers.keys()) == {
        "adjusted_rand",
        "normalized_mutual_info",
        "homogeneity",
        "completeness",
        "v_measure",
    }
    adjusted_rand = cfg.scorers["adjusted_rand"].score_function
    assert callable(adjusted_rand)
    assert adjusted_rand.__name__ == "adjusted_rand_score"


def test_default_cluster_scorer_dict_respects_provided_scorers():
    custom = {
        "custom_metric": ScorerConfig(
            score_name="custom_metric",
            score_function="sklearn.metrics.adjusted_rand_score",
        ),
    }

    cfg = DefaultClusterScorerDictConfig(scorers=custom)

    assert set(cfg.scorers.keys()) == {"custom_metric"}
    assert cfg.scorers["custom_metric"].score_name == "custom_metric"
