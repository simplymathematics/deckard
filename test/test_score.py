import unittest
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from deckard.score import (
    ScorerConfig,
    ScorerDictConfig,
    DefaultClassifierDict,
    DefaultRegressorDict,
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnRegressionConfig,
    FairlearnScoreDictConfig,
    AttackScorerConfig,
    DefaultDataClassificationDict,
    DefaultDataRegressionDict,
    DefaultDataClassificationConfig,
    DefaultDataRegressionConfig,
    survival_concordance_score,
    survival_aic_score,
    survival_bic_score,
)
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score


class TestScorerDictConfigMerge(unittest.TestCase):
    """ScorerDictConfig.merge() should union scorer dicts from multiple specs."""

    acc_dict = {
        "accuracy": {"score_function": "sklearn.metrics.accuracy_score"}
    }
    prec_dict = {
        "precision": {
            "score_function": "sklearn.metrics.precision_score",
            "score_params": {"average": "weighted", "zero_division": 0},
        },
    }
    f1_dict = {
        "f1": {
            "score_function": "sklearn.metrics.f1_score",
            "score_params": {"average": "weighted", "zero_division": 0},
        },
    }

    def test_merge_two_bare_dicts(self):
        result = ScorerDictConfig.merge([self.acc_dict, self.prec_dict])
        self.assertIsInstance(result, ScorerDictConfig)
        self.assertIn("accuracy", result.scorers)
        self.assertIn("precision", result.scorers)

    def test_merge_three_dicts(self):
        result = ScorerDictConfig.merge(
            [self.acc_dict, self.prec_dict, self.f1_dict]
        )
        self.assertEqual(
            set(result.scorers.keys()), {"accuracy", "precision", "f1"}
        )

    def test_merge_scorer_dict_config_instances(self):
        a = ScorerDictConfig(scorers=self.acc_dict)
        b = ScorerDictConfig(scorers=self.prec_dict)
        result = ScorerDictConfig.merge([a, b])
        self.assertIn("accuracy", result.scorers)
        self.assertIn("precision", result.scorers)

    def test_merge_later_wins_on_key_conflict(self):
        override = {
            "accuracy": {
                "score_function": "sklearn.metrics.balanced_accuracy_score"
            },
        }
        result = ScorerDictConfig.merge([self.acc_dict, override])
        self.assertEqual(
            result.scorers["accuracy"].score_function.__name__,
            "balanced_accuracy_score",
        )

    def test_merge_dict_with_scorers_key(self):
        wrapped = {"scorers": self.acc_dict}
        result = ScorerDictConfig.merge([wrapped, self.prec_dict])
        self.assertIn("accuracy", result.scorers)
        self.assertIn("precision", result.scorers)

    def test_merge_single_element_list(self):
        result = ScorerDictConfig.merge([self.acc_dict])
        self.assertIn("accuracy", result.scorers)


class TestScorerConfig(unittest.TestCase):
    def test_scorer_config_initialization(self):
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={"normalize": True},
        )
        self.assertEqual(config.score_name, "accuracy")
        self.assertTrue(callable(config.score_function))

    def test_scorer_config_callable(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, accuracy_score(y_true, y_pred))

    def test_scorer_config_swap(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="accuracy",
            score_function=accuracy_score,
            score_params={},
        )
        score_swap = config(y_true=y_true, y_pred=y_pred, swap=True)
        score_normal = config(y_true=y_pred, y_pred=y_true)
        self.assertEqual(score_swap, score_normal)

    def test_scorer_config_with_additional_params(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        config = ScorerConfig(
            score_name="precision",
            score_function=precision_score,
            score_params={"average": "binary", "zero_division": 0},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        self.assertEqual(
            score,
            precision_score(y_true, y_pred, average="binary", zero_division=0),
        )

    def test_scorer_config_accepts_torch_tensors_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional and not installed")

        y_true = torch.tensor([1, 0, 1, 1])
        y_pred = torch.tensor([1, 0, 0, 1])
        config = ScorerConfig(
            score_name="accuracy",
            score_function="sklearn.metrics.accuracy_score",
            score_params={},
        )
        score = config(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 0.75)


class TestScorerDictConfig(unittest.TestCase):
    def test_scorer_dict_config_initialization_and_call(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        scorer_dict = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                    score_params={},
                ),
                "mse": ScorerConfig(
                    score_name="mse",
                    score_function=mean_squared_error,
                    score_params={},
                ),
            },
        )
        scores = scorer_dict(y_true=y_true, y_pred=y_pred)
        self.assertIn("accuracy", scores)
        self.assertIn("mse", scores)
        self.assertEqual(scores["accuracy"], accuracy_score(y_true, y_pred))
        self.assertEqual(scores["mse"], mean_squared_error(y_true, y_pred))

    def test_scorer_dict_config_get_callables(self):
        scorer_dict = ScorerDictConfig(
            scorers={
                "accuracy": ScorerConfig(
                    score_name="accuracy",
                    score_function=accuracy_score,
                    score_params={},
                ),
            },
        )
        callables = scorer_dict.get_callables()
        self.assertIn("accuracy", callables)
        self.assertTrue(callable(callables["accuracy"]))


class TestDefaultScorerDicts(unittest.TestCase):
    def test_default_classifier_dict(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        y_proba = [0.9, 0.1, 0.3, 0.8]
        scores = DefaultClassifierDict.scorers(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )
        self.assertIn("accuracy", scores)
        self.assertIn("precision", scores)
        self.assertIn("recall", scores)
        self.assertIn("f1", scores)
        self.assertIn("roc_auc", scores)
        self.assertIn("log_loss", scores)

    def test_default_classifier_dict_requires_probabilities_for_probability_metrics(
        self,
    ):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        with self.assertRaises(ValueError):
            DefaultClassifierDict.scorers(y_true=y_true, y_pred=y_pred)

    def test_default_regressor_dict(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        scores = DefaultRegressorDict.scorers(y_true=y_true, y_pred=y_pred)
        self.assertIn("mse", scores)
        self.assertIn("mae", scores)
        self.assertIn("r2", scores)

    def test_default_classifier_dict_with_empty_predictions(self):
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            DefaultClassifierDict.scorers(y_true=y_true, y_pred=y_pred)

    def test_default_regressor_dict_with_empty_predictions(self):
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            DefaultRegressorDict.scorers(y_true=y_true, y_pred=y_pred)


class TestSurvivalScorers(unittest.TestCase):
    class _MockFitter:
        def __init__(self):
            self.concordance_index_ = 0.71
            self.AIC_ = 123.4
            self.log_likelihood_ = -50.0
            self.params_ = [1.0, 2.0, 3.0]

    def test_survival_concordance_score(self):
        fitter = self._MockFitter()
        score = survival_concordance_score(y_true=[1, 2, 3], y_pred=fitter)
        self.assertEqual(score, fitter.concordance_index_)

    def test_survival_aic_score(self):
        fitter = self._MockFitter()
        score = survival_aic_score(y_true=[1, 2, 3], y_pred=fitter)
        self.assertEqual(score, fitter.AIC_)

    def test_survival_bic_score_computed(self):
        fitter = self._MockFitter()
        score = survival_bic_score(y_true=[1, 2, 3, 4, 5], y_pred=fitter)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)


class TestFairnessScorers(unittest.TestCase):
    def test_fairness_classification_and_regression_profiles_are_distinct(self):
        classification = DefaultFairlearnClassificationConfig()
        regression = DefaultFairlearnRegressionConfig()

        self.assertIn("demographic_parity_difference", classification.scorers)
        self.assertIn("equalized_odds_difference", classification.scorers)
        self.assertIn("group_mae_difference", regression.scorers)
        self.assertIn("group_mse_difference", regression.scorers)
        self.assertIn("group_mean_prediction_difference", regression.scorers)
        self.assertNotIn("group_mae_difference", classification.scorers)

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

        self.assertIn("group_mean_prediction_difference", scores)
        self.assertIn("group_mae_difference", scores)
        self.assertIn("group_mse_difference", scores)
        self.assertGreaterEqual(scores["group_mean_prediction_difference"], 0.0)
        self.assertGreaterEqual(scores["group_mae_difference"], 0.0)
        self.assertGreaterEqual(scores["group_mse_difference"], 0.0)

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

        self.assertIsInstance(scores, dict)
        self.assertTrue(
            any(key.endswith("accuracy_difference") for key in scores),
        )


class TestAttackScorers(unittest.TestCase):
    def test_attack_scorer_evasion(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_evasion(
            ben_pred_labels=[0, 1, 0, 1],
            adv_pred_labels=[0, 0, 0, 1],
            y_true=[0, 1, 0, 1],
            attack_size=4,
        )
        self.assertIn("evasion_accuracy", scores)
        self.assertIn("evasion_success", scores)
        self.assertIn("attack_score_time", scores)

    def test_attack_scorer_evasion_regression(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_evasion(
            ben_pred_labels=[1.0, 2.0, 3.0, 4.0],
            adv_pred_labels=[1.1, 1.9, 3.2, 3.8],
            y_true=[1.0, 2.0, 3.0, 4.0],
            attack_size=4,
            is_classification=False,
        )
        self.assertIn("evasion_mse", scores)
        self.assertIn("evasion_mae", scores)
        self.assertIn("evasion_r2", scores)
        self.assertNotIn("evasion_success", scores)
        self.assertIn("attack_score_time", scores)

    def test_attack_scorer_membership(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_membership(
            labels=[1, 1, 0, 0],
            inferred=[1, 0, 0, 0],
            attack_size=4,
        )
        self.assertIn("membership_inference_accuracy", scores)
        self.assertIn("membership_inference_precision", scores)
        self.assertIn("attack_score_time", scores)

    def test_attack_scorer_attribute_classification(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_attribute(
            target=[1, 0, 1, 0],
            inferred=[1, 1, 1, 0],
            attack_size=4,
            targeted_attribute="age",
            is_classification=True,
            attack_generation_time=0.1,
        )
        self.assertIn("inferred_age_accuracy", scores)
        self.assertIn("inferred_age_f1", scores)
        self.assertIn("attack_generation_time", scores)

    def test_attack_scorer_attribute_regression(self):
        scorer = AttackScorerConfig()
        scores = scorer.score_attribute(
            target=[1.0, 2.0, 3.0, 4.0],
            inferred=[1.1, 1.9, 3.2, 3.8],
            attack_size=4,
            targeted_attribute="income",
            is_classification=False,
        )
        self.assertIn("inferred_income_mse", scores)
        self.assertIn("inferred_income_r2", scores)
        self.assertIn("attack_score_time", scores)

    def test_attack_score_configstores_registered(self):
        scorer = AttackScorerConfig()
        cs = ConfigStore.instance()
        self.assertIsNotNone(cs)
        self.assertIsNotNone(scorer.evasion)
        self.assertIsNotNone(scorer.membership_inference)
        self.assertIsNotNone(scorer.attribute_inference)


class TestDataInspectionScorers(unittest.TestCase):
    def test_data_classification_default_scores(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        X = pd.DataFrame(
            {
                "feature_0": [0, 1, 0, 1, 1, 0, 1, 0],
                "feature_1": [1, 2, 1, 2, 2, 1, 2, 1],
            },
        )
        scores = DefaultDataClassificationDict.scorers(
            y_true=y_true,
            y_pred=X,
            mode=None,
        )
        self.assertIn("num_classes", scores)
        self.assertIn("class_count_min", scores)
        self.assertIn("class_count_max", scores)
        self.assertIn("class_imbalance_ratio", scores)
        self.assertIn("mutual_information_mean", scores)
        self.assertIn("mutual_information_max", scores)
        self.assertEqual(scores["num_classes"], 2)

    def test_data_classification_reference_column_override(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        X = pd.DataFrame(
            {
                "age": [20, 24, 41, 45, 44, 23, 43, 21],
                "income_proxy": [1.2, 1.5, 2.8, 3.0, 2.9, 1.4, 3.1, 1.3],
            },
        )
        scores = DefaultDataClassificationDict.scorers(
            y_true=y_true,
            y_pred=X,
            mode=None,
            reference_column="age",
        )
        self.assertIn("mutual_information_mean", scores)
        self.assertGreaterEqual(scores["mutual_information_mean"], 0.0)

    def test_data_regression_default_scores_include_ecdf(self):
        y_true = np.array([2.1, 2.5, 3.0, 3.4, 3.6, 4.0])
        X = pd.DataFrame(
            {
                "feature_0": [0.1, 0.2, 0.4, 0.7, 0.8, 1.0],
                "feature_1": [10, 12, 14, 18, 19, 22],
            },
        )
        scores = DefaultDataRegressionDict.scorers(
            y_true=y_true, y_pred=X, mode=None
        )
        self.assertIn("mutual_information_mean", scores)
        self.assertIn("mutual_information_max", scores)
        self.assertIn("empirical_cdf", scores)
        self.assertTrue(callable(scores["empirical_cdf"]))

        ecdf = scores["empirical_cdf"]
        values = ecdf(np.array([2.1, 3.0, 4.0]))
        self.assertTrue(np.all(values[:-1] <= values[1:]))

    def test_data_scorer_configstores_registered(self):
        cs = ConfigStore.instance()
        self.assertIsNotNone(cs)
        self.assertIsInstance(
            DefaultDataClassificationConfig(),
            DefaultDataClassificationConfig,
        )
        self.assertIsInstance(
            DefaultDataRegressionConfig(),
            DefaultDataRegressionConfig,
        )

    def test_score_profile_classes_available(self):
        self.assertIsInstance(DefaultClassifierDict.scorers, ScorerDictConfig)
        self.assertIsInstance(DefaultRegressorDict.scorers, ScorerDictConfig)
        self.assertIsInstance(
            DefaultFairlearnClassificationConfig(), ScorerDictConfig
        )
        self.assertIsInstance(
            DefaultFairlearnRegressionConfig(), ScorerDictConfig
        )


if __name__ == "__main__":
    unittest.main()
