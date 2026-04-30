import unittest
import pandas as pd
import tempfile
import shutil
from unittest.mock import Mock
import pytest

pytest.importorskip("fairlearn")

from deckard.model.fairness import FairnessModelConfig  # NOQA E402
from deckard.data.fairness import FairnessDataConfig  # NOQA E402
from deckard.model.fairness import FairnessDefenseConfig  # NOQA E402


class TestFairnessModelConfig(unittest.TestCase):
    def setUp(self):
        # Create sample data with group information
        self.X_train = pd.DataFrame(
            {
                "feature1": [0, 1, 2, 3, 4, 5],
                "feature2": [1, 2, 3, 4, 5, 6],
                "group": ["A", "B", "A", "B", "A", "B"],
            },
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 1])

        self.X_test = pd.DataFrame(
            {
                "feature1": [6, 7, 8, 9],
                "feature2": [7, 8, 9, 10],
                "group": ["A", "B", "A", "B"],
            },
        )
        self.y_test = pd.Series([1, 0, 1, 0])

        # Create sensitive feature series for fairness evaluation
        self.sensitive_test = pd.Series(["A", "B", "A", "B"], index=self.y_test.index)

        self.model_type = "sklearn.ensemble.RandomForestClassifier"
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_fairness_model_config_initialization(self):
        """Test FairnessModelConfig can be initialized."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.data, fairness_data)

    def test_fairness_model_config_initialization_without_data(self):
        """Test FairnessModelConfig can be initialized without fairness data."""
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )

        self.assertIsNotNone(model)
        self.assertIsNone(model.data)

    def test_classification_scores_without_fairness_data(self):
        """Test classification scores when fairness_data is None."""
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )

        scores = model._classification_scores(self.y_test, self.y_test)

        self.assertIsInstance(scores, dict)
        self.assertIn("accuracy", scores)
        # Should not contain group-specific scores
        self.assertNotIn("A_accuracy", scores)
        self.assertNotIn("B_accuracy", scores)

    def test_classification_scores_with_fairness_data(self):
        """Test classification scores includes group fairness metrics."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._classification_scores(self.y_test, self.y_test)

        self.assertIsInstance(scores, dict)
        self.assertIn("accuracy", scores)
        # Should contain group-specific scores
        self.assertIn("A_accuracy", scores)
        self.assertIn("B_accuracy", scores)

    def test_regression_scores_without_fairness_data(self):
        """Test regression scores when fairness_data is None."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8])

        model = FairnessModelConfig(
            model_type="sklearn.linear_model.LinearRegression",
            classifier=False,
            data=None,
        )

        scores = model._regression_scores(y_true, y_pred)

        self.assertIsInstance(scores, dict)
        self.assertIn("mse", scores)
        # Should not contain group-specific scores
        self.assertNotIn("A_mse", scores)
        self.assertNotIn("B_mse", scores)

    def test_regression_scores_with_fairness_data(self):
        """Test regression scores includes group fairness metrics."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0], index=self.sensitive_test.index)
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8], index=self.sensitive_test.index)

        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type="sklearn.linear_model.LinearRegression",
            classifier=False,
            data=fairness_data,
        )

        scores = model._regression_scores(y_true, y_pred)

        self.assertIsInstance(scores, dict)
        self.assertIn("mse", scores)
        # Should contain group-specific scores
        self.assertIn("A_mse", scores)
        self.assertIn("B_mse", scores)

    def test_compute_group_fairness_scores_no_sensitive_features(self):
        """Test group fairness scores error when sensitive features are missing."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = None

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        with self.assertRaises(ValueError):
            model._compute_group_fairness_scores(self.y_test, self.y_test)

    def test_compute_group_fairness_scores_none_sensitive_features(self):
        """Test group fairness scores errors when sensitive features are all None."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = None
        fairness_data.sensitive_train_ = None
        fairness_data.sensitive_all_ = None

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        with self.assertRaises(ValueError):
            model._compute_group_fairness_scores(self.y_test, self.y_test)

    def test_compute_group_fairness_scores_empty_group(self):
        """Test group fairness scores skips empty groups."""
        # Create groups with different sizes
        sensitive_test = pd.Series(["A", "A", "A", "B"], index=self.y_test.index)

        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = sensitive_test

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)

        self.assertIsInstance(scores, dict)
        # Should have scores for both groups
        self.assertTrue(any("A_" in key for key in scores.keys()))
        self.assertTrue(any("B_" in key for key in scores.keys()))

    def test_compute_group_fairness_scores_classification(self):
        """Test group fairness scores for classification task."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)

        self.assertIsInstance(scores, dict)
        # Check for classification metrics per group
        for metric in ["accuracy", "precision", "recall", "f1-score"]:
            self.assertTrue(
                any(
                    f"A_{metric}" in key or f"B_{metric}" in key
                    for key in scores.keys()
                ),
            )

    def test_compute_group_fairness_scores_regression(self):
        """Test group fairness scores for regression task."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0], index=self.sensitive_test.index)
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8], index=self.sensitive_test.index)

        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type="sklearn.linear_model.LinearRegression",
            classifier=False,
            data=fairness_data,
        )

        scores = model._compute_group_fairness_scores(y_true, y_pred)

        self.assertIsInstance(scores, dict)
        # Check for regression metrics per group
        for metric in ["mse", "rmse", "mae"]:
            self.assertTrue(
                any(
                    f"A_{metric}" in key or f"B_{metric}" in key
                    for key in scores.keys()
                ),
            )

    def test_group_fairness_scores_naming_convention(self):
        """Test that group fairness scores follow naming convention."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)

        # Check naming convention: {group_name}_{metric}
        for key in scores.keys():
            self.assertTrue("_" in key, f"Key {key} should contain group_metric format")
            parts = key.split("_")
            self.assertGreaterEqual(
                len(parts),
                2,
                f"Key {key} should have group and metric",
            )

    def test_train_passes_sensitive_features_when_supported(self):
        class SensitiveFitEstimator:
            def __init__(self):
                self.received_sensitive = None

            def fit(self, X, y, sensitive_features=None):
                self.received_sensitive = sensitive_features
                return self

            def predict(self, X, sensitive_features=None):
                return pd.Series([0] * len(X))

        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_train_ = self.sensitive_test
        fairness_data.sensitive_test_ = self.sensitive_test
        fairness_data.sensitive_all_ = self.sensitive_test

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        model._model = SensitiveFitEstimator()

        model._train(self.X_test, self.y_test)

        self.assertIsNotNone(model._model.received_sensitive)
        self.assertEqual(len(model._model.received_sensitive), len(self.y_test))

    def test_predict_passes_sensitive_features_when_supported(self):
        class SensitivePredictEstimator:
            def fit(self, X, y):
                return self

            def predict(self, X, sensitive_features=None):
                if sensitive_features is None:
                    raise AssertionError("sensitive_features was not provided")
                return pd.Series([0] * len(X))

        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_test_ = self.sensitive_test
        fairness_data.sensitive_train_ = None
        fairness_data.sensitive_all_ = None

        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        model._model = SensitivePredictEstimator()

        y_pred = model._predict(self.X_test)

        self.assertEqual(len(y_pred), len(self.X_test))


class TestFairnessDefenseConfigApplyDefense(unittest.TestCase):
    """Tests for FairnessDefenseConfig.apply_defense with fairlearn estimators."""

    def setUp(self):

        self.FairnessDefenseConfig = FairnessDefenseConfig
        self.model_type = "sklearn.linear_model.LogisticRegression"

        self.X_train = pd.DataFrame(
            {"f1": [0, 1, 2, 3, 4, 5], "f2": [1, 2, 3, 4, 5, 6]},
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 1])
        self.sensitive_train = pd.Series(["A", "B", "A", "B", "A", "B"])

        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.sensitive_train_ = self.sensitive_train
        fairness_data.sensitive_test_ = None
        fairness_data.sensitive_all_ = None
        self.fairness_data = fairness_data

    def _make_fitted_defense(self, defense_name, defense_params=None):
        cfg = self.FairnessDefenseConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"max_iter": 200},
            defense_name=defense_name,
            defense_params=defense_params or {},
            data=self.fairness_data,
        )
        cfg._train(self.X_train, self.y_train)
        return cfg

    def test_apply_defense_reductions_exponentiated_gradient_string_constraint(self):
        """ExponentiatedGradient with constraint given as a module-path string."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {"constraints": "fairlearn.reductions.DemographicParity", "eps": 0.1},
        )
        result = cfg.apply_defense(None)
        from fairlearn.reductions import ExponentiatedGradient

        self.assertIsInstance(result, ExponentiatedGradient)

    def test_apply_defense_reductions_exponentiated_gradient_dict_constraint(self):
        """ExponentiatedGradient with constraint given as a _target_ dict."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": {"_target_": "fairlearn.reductions.EqualizedOdds"},
                "eps": 0.1,
            },
        )
        result = cfg.apply_defense(None)
        from fairlearn.reductions import ExponentiatedGradient

        self.assertIsInstance(result, ExponentiatedGradient)

    def test_apply_defense_reductions_requires_constraints(self):
        """ExponentiatedGradient without a constraints key must raise ValueError."""
        cfg = self._make_fitted_defense("fairlearn.reductions.ExponentiatedGradient")
        with self.assertRaises(
            ValueError,
            msg="constraints are required for reductions",
        ):
            cfg.apply_defense(None)

    def test_apply_defense_postprocessing_threshold_optimizer(self):
        """ThresholdOptimizer wraps the base estimator correctly."""
        cfg = self._make_fitted_defense(
            "fairlearn.postprocessing.ThresholdOptimizer",
            {"constraints": "demographic_parity"},
        )
        result = cfg.apply_defense(None)
        from fairlearn.postprocessing import ThresholdOptimizer

        self.assertIsInstance(result, ThresholdOptimizer)

    def test_apply_defense_postprocessing_no_constraints(self):
        """ThresholdOptimizer with no constraints key uses default."""
        cfg = self._make_fitted_defense("fairlearn.postprocessing.ThresholdOptimizer")
        result = cfg.apply_defense(None)
        from fairlearn.postprocessing import ThresholdOptimizer

        self.assertIsInstance(result, ThresholdOptimizer)

    def test_apply_defense_unsupported_fairlearn_submodule_raises(self):
        """Unsupported fairlearn submodule (e.g., fairlearn.metrics.*) must raise NotImplementedError."""
        cfg = self._make_fitted_defense("fairlearn.metrics.MetricFrame")
        with self.assertRaises((NotImplementedError, ImportError)):
            cfg.apply_defense(None)

    def test_apply_defense_defense_application_time_set(self):
        """defense_application_time is recorded after a successful fairlearn defense."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {"constraints": "fairlearn.reductions.DemographicParity", "eps": 0.1},
        )
        cfg.apply_defense(None)
        self.assertIsNotNone(cfg.defense_application_time)


if __name__ == "__main__":
    unittest.main()
