import unittest

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error

from deckard.plugins.fairlearn.score import (
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnRegressionConfig,
    FairlearnScoreDictConfig,
)
from deckard.score.attack import FairlearnAttackScorerConfig
from deckard.score import ScorerConfig, ScorerDictConfig


class TestFairnessScorers(unittest.TestCase):
    def test_fairness_classification_and_regression_profiles_are_distinct(self):
        classification = DefaultFairlearnClassificationConfig()
        regression = DefaultFairlearnRegressionConfig()
        self.assertIn("accuracy", classification.scorers.keys())
        self.assertIn("mse", regression.scorers.keys())
        self.assertIn("demographic_parity_difference", classification.scorers.keys())
        self.assertIn("equalized_odds_difference", classification.scorers.keys())
        # Regression group metrics should only be in group_scorers, not main scorers
        self.assertIn("group_mae_difference", regression.scorers.keys())
        self.assertIn("group_mse_difference", regression.scorers.keys())
        # group_mean_prediction_difference is only in group_scorers for regression
        self.assertNotIn("group_mean_prediction_difference", regression.scorers.keys())
        self.assertIn("group_mae_difference", regression.group_scorers)
        self.assertIn("group_mse_difference", regression.group_scorers)
        self.assertNotIn("group_mean_prediction_difference", regression.group_scorers)
        self.assertIn("group_mae_difference", regression.scorers.keys())

    def test_fairness_regression_scores(self):
        scorer = DefaultFairlearnRegressionConfig()
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
        self.assertNotIn("0_group_mean_prediction_difference", scores)
        self.assertNotIn("1_group_mean_prediction_difference", scores)
        self.assertIn("0_group_mae_difference", scores)
        self.assertIn("1_group_mae_difference", scores)
        self.assertIn("0_group_mse_difference", scores)
        self.assertIn("1_group_mse_difference", scores)

    def test_score_profile_classes_available(self):
        self.assertIsInstance(
            DefaultFairlearnClassificationConfig(),
            FairlearnScoreDictConfig,
        )
        self.assertIsInstance(
            DefaultFairlearnRegressionConfig(),
            FairlearnScoreDictConfig,
        )

    def test_metric_frame_fairness_score_dict_classification(self):
        scorer = FairlearnScoreDictConfig(
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
        self.assertIn("A_accuracy", scores)
        self.assertIn("B_accuracy", scores)
        self.assertIn("accuracy_overall", scores)
        self.assertIn("accuracy_difference", scores)

    def test_metric_frame_fairness_score_dict_regression(self):
        scorer = FairlearnScoreDictConfig(
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
        self.assertIn("0_mse", scores)
        self.assertIn("1_mse", scores)
        self.assertIn("mse_ratio", scores)

    def test_metric_frame_fairness_score_dict_supports_full_metricframe_kwargs(
        self,
    ):
        scorer = FairlearnScoreDictConfig(
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
        self.assertIn("X_A_accuracy", scores)
        self.assertIn("X_B_accuracy", scores)
        self.assertIn("Y_A_accuracy", scores)
        self.assertIn("Y_B_accuracy", scores)
        self.assertIn("X_accuracy_overall", scores)
        self.assertIn("Y_accuracy_overall", scores)
        self.assertIn("X_accuracy_difference", scores)
        self.assertIn("Y_accuracy_difference", scores)

        self.assertIsInstance(scores, dict)
        self.assertTrue(
            any(key.endswith("accuracy_difference") for key in scores),
        )

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
        self.assertIsNotNone(scorer.evasion)
