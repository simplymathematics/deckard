import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error

from deckard.plugins.fairlearn.score import (
    DefaultFairlearnClassificationScorerDictConfig,
    DefaultFairlearnRegressionScorerDictConfig,
    FairlearnScorerDictConfig,
)
from deckard.score.attack import FairlearnAttackScorerConfig
from deckard.score import ScorerConfig, ScorerDictConfig


class TestFairnessScorers:
    def test_fairness_classification_and_regression_profiles_are_distinct(self):
        classification = DefaultFairlearnClassificationScorerDictConfig()
        regression = DefaultFairlearnRegressionScorerDictConfig()
        assert "accuracy" in classification.scorers.keys()
        assert "mse" in regression.scorers.keys()
        assert "demographic_parity_difference" in classification.scorers.keys()
        assert "equalized_odds_difference" in classification.scorers.keys()
        # Regression group metrics should only be in group_scorers, not main scorers
        assert "group_mae_difference" in regression.scorers.keys()
        assert "group_mse_difference" in regression.scorers.keys()
        # group_mean_prediction_difference is only in group_scorers for regression
        assert "group_mean_prediction_difference" not in regression.scorers.keys()
        assert "group_mae_difference" in regression.group_scorers
        assert "group_mse_difference" in regression.group_scorers
        assert "group_mean_prediction_difference" not in regression.group_scorers
        assert "group_mae_difference" in regression.scorers.keys()

    def test_fairness_regression_scores(self):
        scorer = DefaultFairlearnRegressionScorerDictConfig()
        y_true = np.array([1.0, 1.8, 3.2, 4.0])
        y_pred = np.array([1.1, 1.6, 2.8, 4.3])
        sensitive = np.array([0, 0, 1, 1])

        scores = scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            sensitive_features=sensitive,
        )

        # Only expect per-group keys for group metrics that are present
        assert "0_group_mean_prediction_difference" not in scores
        assert "1_group_mean_prediction_difference" not in scores
        assert "0_group_mae_difference" in scores
        assert "1_group_mae_difference" in scores
        assert "0_group_mse_difference" in scores
        assert "1_group_mse_difference" in scores

    def test_score_profile_classes_available(self):
        assert isinstance(
            DefaultFairlearnClassificationScorerDictConfig(),
            FairlearnScorerDictConfig,
        )
        assert isinstance(
            DefaultFairlearnRegressionScorerDictConfig(),
            FairlearnScorerDictConfig,
        )

    def test_metric_frame_fairness_score_dict_classification(self):
        scorer = FairlearnScorerDictConfig(
            group_scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                ),
            },
            group_reduction="difference",
            include_group_by_group=True,
            include_group_overall=True,
        )
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        sensitive = np.array(["A", "A", "B", "B"])

        scores = scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            sensitive_features=sensitive,
        )
        assert "A_accuracy" in scores
        assert "B_accuracy" in scores
        assert "accuracy_overall" in scores
        assert "accuracy_difference" in scores

    def test_metric_frame_fairness_score_dict_regression(self):
        scorer = FairlearnScorerDictConfig(
            group_scorers={
                "mse": ScorerConfig(
                    score_name="mse",
                    score_function=mean_squared_error,
                ),
            },
            group_reduction="ratio",
            include_group_by_group=True,
            include_group_overall=False,
        )
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 2.8, 4.2])
        sensitive = np.array([0, 0, 1, 1])

        scores = scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            sensitive_features=sensitive,
        )
        assert "0_mse" in scores
        assert "1_mse" in scores
        assert "mse_ratio" in scores

    def test_metric_frame_fairness_score_dict_supports_full_metricframe_kwargs(
        self,
    ):
        scorer = FairlearnScorerDictConfig(
            group_scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                ),
            },
            group_reduction="difference",
            include_group_by_group=True,
            include_group_overall=True,
            n_boot=5,
            ci_quantiles=[0.25, 0.75],
            random_state=7,
        )
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "B", "B", "A", "B"])
        control = np.array(["X", "X", "X", "Y", "Y", "Y"])
        sample_weight = np.array([1.0, 2.0, 1.0, 1.0, 0.5, 1.5])

        scores = scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            sensitive_features=sensitive,
            control_features=control,
            sample_params={"sample_weight": sample_weight},
        )
        assert "X_A_accuracy" in scores
        assert "X_B_accuracy" in scores
        assert "Y_A_accuracy" in scores
        assert "Y_B_accuracy" in scores
        assert "X_accuracy_overall" in scores
        assert "Y_accuracy_overall" in scores
        assert "X_accuracy_difference" in scores
        assert "Y_accuracy_difference" in scores

        assert isinstance(scores, dict)
        assert any(key.endswith("accuracy_difference") for key in scores)

    def test_fairlearn_attack_scorer_accepts_plain_scorerdict_profiles(self):
        pytest.importorskip("fairlearn")

        profile = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function="sklearn.metrics.accuracy_score",
                ),
            },
        )
        scorer = FairlearnAttackScorerConfig(evasion=profile)
        assert scorer.evasion is not None
