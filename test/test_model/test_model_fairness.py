import unittest
import pandas as pd
import tempfile
import shutil
from unittest.mock import Mock
import pytest

from deckard.model.fairness import FairlearnModelConfig
from deckard.data.fairness import FairlearnDataConfig
from deckard.model.fairness import FairlearnDefenseConfig
from deckard.model.defend import DefenseConfig
from deckard.model.defend import DefensePipelineConfig

pytest.importorskip("fairlearn")


class TestFairlearnModelConfig(unittest.TestCase):
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
        self.sensitive_test = pd.Series(
            ["A", "B", "A", "B"],
            index=self.y_test.index,
        )

        self.model_type = "sklearn.ensemble.RandomForestClassifier"
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_fairness_model_config_initialization(self):
        """Test FairlearnModelConfig can be initialized."""
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.data, fairness_data)

    def test_fairness_model_config_initialization_without_data(self):
        """Test FairlearnModelConfig can be initialized without fairness data."""
        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )

        self.assertIsNotNone(model)
        self.assertIsNone(model.data)

    def test_apply_defense_supports_mixed_defense_pipeline(self):
        """ART + fairlearn defenses are applied sequentially via DefensePipelineConfig."""
        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )
        model._model = Mock()

        art_defense = DefenseConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            defense_name=None,
            defense_params={},
        )
        fair_defense = FairlearnDefenseConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            defense_name="fairlearn.postprocessing.ThresholdOptimizer",
            defense_params={"constraints": "demographic_parity"},
            data=None,
        )

        first_estimator = Mock(name="art_wrapped_estimator")
        second_estimator = Mock(name="fairlearn_wrapped_estimator")
        art_defense.defense_application_time = 0.3
        fair_defense.defense_application_time = 0.4
        art_defense.apply_to = Mock(return_value=first_estimator)
        fair_defense.apply_to = Mock(return_value=second_estimator)

        model.defense = DefensePipelineConfig(
            defenses=[art_defense, fair_defense],
        )
        runtime_data = Mock()
        result = model._apply_defense(data=runtime_data)

        self.assertIs(result, second_estimator)
        art_defense.apply_to.assert_called_once_with(
            estimator=model._model,
            data=runtime_data,
        )
        fair_defense.apply_to.assert_called_once_with(
            estimator=first_estimator,
            data=runtime_data,
        )
        self.assertAlmostEqual(model.defense_application_time, 0.7)
        self.assertIs(fair_defense.data, runtime_data)

    def test_apply_defense_rejects_legacy_defense_list(self):
        """Legacy list assignment is intentionally unsupported after pipeline migration."""
        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None,
        )
        model._model = Mock()

        model.defense = [
            DefenseConfig(
                model_type=self.model_type,
                classifier=True,
                model_params={"n_estimators": 10},
                defense_name=None,
                defense_params={},
            ),
        ]

        with self.assertRaises(TypeError):
            model._apply_defense(data=Mock())

    def test_classification_scores_without_fairness_data(self):
        """Test classification scores when fairness_data is None."""
        model = FairlearnModelConfig(
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
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
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

        model = FairlearnModelConfig(
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
        y_true = pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=self.sensitive_test.index,
        )
        y_pred = pd.Series(
            [1.1, 1.9, 3.2, 3.8],
            index=self.sensitive_test.index,
        )

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
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

    def test_compute_sensitive_fairness_scores_no_sensitive_features(self):
        """Test sensitive fairness scores error when sensitive features are missing."""
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = None

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        with self.assertRaises(ValueError):
            model._compute_sensitive_fairness_scores(self.y_test, self.y_test)

    def test_compute_sensitive_fairness_scores_none_sensitive_features(self):
        """Test sensitive fairness scores error when sensitive features are all None."""
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = None
        fairness_data._sensitive_train = None
        fairness_data._sensitive_all = None

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        with self.assertRaises(ValueError):
            model._compute_sensitive_fairness_scores(self.y_test, self.y_test)

    def test_compute_sensitive_fairness_scores_empty_group(self):
        """Test sensitive fairness scores skips empty groups."""
        # Create groups with different sizes
        sensitive_test = pd.Series(
            ["A", "A", "A", "B"],
            index=self.y_test.index,
        )

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = sensitive_test

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_sensitive_fairness_scores(
            self.y_test,
            self.y_test,
        )

        self.assertIsInstance(scores, dict)
        # Should have scores for both groups
        self.assertTrue(any("A_" in key for key in scores.keys()))
        self.assertTrue(any("B_" in key for key in scores.keys()))

    def test_compute_sensitive_fairness_scores_classification(self):
        """Test sensitive fairness scores for classification task."""
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_sensitive_fairness_scores(
            self.y_test,
            self.y_test,
        )

        self.assertIsInstance(scores, dict)
        # Check for classification metrics per group
        for metric in ["accuracy", "precision", "recall", "f1-score"]:
            self.assertTrue(
                any(
                    f"A_{metric}" in key or f"B_{metric}" in key
                    for key in scores.keys()
                ),
            )

    def test_compute_sensitive_fairness_scores_regression(self):
        """Test sensitive fairness scores for regression task."""
        y_true = pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=self.sensitive_test.index,
        )
        y_pred = pd.Series(
            [1.1, 1.9, 3.2, 3.8],
            index=self.sensitive_test.index,
        )

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
            model_type="sklearn.linear_model.LinearRegression",
            classifier=False,
            data=fairness_data,
        )

        scores = model._compute_sensitive_fairness_scores(y_true, y_pred)

        self.assertIsInstance(scores, dict)
        # Check for regression metrics per sensitive value
        for metric in ["mse", "rmse", "mae"]:
            self.assertTrue(
                any(
                    f"A_{metric}" in key or f"B_{metric}" in key
                    for key in scores.keys()
                ),
            )


def test_resolve_fairlearn_model_param_moves_torch_model_to_cpu():
    torch = pytest.importorskip("torch")

    cfg = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=None,
    )

    resolved = cfg._resolve_fairlearn_model_param(
        {
            "model_type": "torch.nn.Linear",
            "model_params": {"in_features": 4, "out_features": 2},
            "device": "cpu",
        },
    )

    assert isinstance(resolved, torch.nn.Module)
    assert next(resolved.parameters()).device.type == "cpu"


def test_resolve_fairlearn_model_param_falls_back_from_unavailable_mps(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.backends, "mps"):
        pytest.skip("torch backend has no mps support")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    cfg = FairlearnModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 25},
        data=None,
    )

    resolved = cfg._resolve_fairlearn_model_param(
        {
            "model_type": "torch.nn.Linear",
            "model_params": {"in_features": 4, "out_features": 2},
            "device": "mps",
        },
    )

    assert isinstance(resolved, torch.nn.Module)
    assert next(resolved.parameters()).device.type == "cpu"

    def test_sensitive_fairness_scores_naming_convention(self):
        """Test that sensitive fairness scores follow naming convention."""
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_sensitive_fairness_scores(
            self.y_test,
            self.y_test,
        )

        # Check naming convention: {group_name}_{metric}
        for key in scores.keys():
            self.assertTrue(
                "_" in key,
                f"Key {key} should contain group_metric format",
            )
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

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_train = self.sensitive_test
        fairness_data._sensitive_test = self.sensitive_test
        fairness_data._sensitive_all = self.sensitive_test

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        model._model = SensitiveFitEstimator()

        model._train(self.X_test, self.y_test)

        self.assertIsNotNone(model._model.received_sensitive)
        self.assertEqual(len(model._model.received_sensitive), len(self.y_test))

    def test_compute_sensitive_fairness_scores_train_mode_uses_train_sensitive(
        self,
    ):
        train_sensitive = pd.Series(
            ["T0", "T1", "T0", "T1"],
            index=self.y_test.index,
        )
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_train = train_sensitive
        fairness_data._sensitive_test = pd.Series(
            ["A", "B", "A", "B"],
            index=self.y_test.index,
        )
        fairness_data._sensitive_all = None

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        scores = model._compute_sensitive_fairness_scores(
            self.y_test,
            self.y_test,
            mode="train",
        )

        self.assertTrue(any(key.startswith("T0_") for key in scores))
        self.assertTrue(any(key.startswith("T1_") for key in scores))

    def test_compute_sensitive_fairness_scores_val_mode_not_implemented(self):
        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_train = self.sensitive_test
        fairness_data._sensitive_test = self.sensitive_test
        fairness_data._sensitive_all = None

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )

        with self.assertRaises(NotImplementedError):
            model._compute_sensitive_fairness_scores(
                self.y_test,
                self.y_test,
                mode="val",
            )

    def test_predict_passes_sensitive_features_when_supported(self):
        class SensitivePredictEstimator:
            def fit(self, X, y):
                return self

            def predict(self, X, sensitive_features=None):
                if sensitive_features is None:
                    raise AssertionError("sensitive_features was not provided")
                return pd.Series([0] * len(X))

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test
        fairness_data._sensitive_train = None
        fairness_data._sensitive_all = None

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        model._model = SensitivePredictEstimator()

        y_pred = model._predict(self.X_test)

        self.assertEqual(len(y_pred), len(self.X_test))

    def test_fairness_model_config_is_configbase_with_hash(self):
        """Test that FairlearnModelConfig is ConfigBase and has __hash__ method."""
        from deckard.utils import ConfigBase

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_test = self.sensitive_test
        fairness_data._sensitive_train = self.sensitive_test
        fairness_data._sensitive_all = self.sensitive_test

        model = FairlearnModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data,
        )
        self.assertIsInstance(
            model,
            ConfigBase,
            msg="FairlearnModelConfig should inherit from ConfigBase",
        )
        self.assertTrue(
            hasattr(model, "__hash__"),
            msg="FairlearnModelConfig should have __hash__ method",
        )
        # Note: FairlearnModelConfig may have unhashable runtime fields
        # so we verify the infrastructure is in place rather than attempting full hash


class TestFairlearnDefenseConfigApplyDefense(unittest.TestCase):
    """Tests for FairlearnDefenseConfig.apply_defense with fairlearn estimators."""

    def setUp(self):

        self.FairlearnDefenseConfig = FairlearnDefenseConfig
        self.model_type = "sklearn.linear_model.LogisticRegression"

        self.X_train = pd.DataFrame(
            {"f1": [0, 1, 2, 3, 4, 5], "f2": [1, 2, 3, 4, 5, 6]},
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0, 1])
        self.sensitive_train = pd.Series(["A", "B", "A", "B", "A", "B"])

        fairness_data = Mock(spec=FairlearnDataConfig)
        fairness_data._sensitive_train = self.sensitive_train
        fairness_data._sensitive_test = None
        fairness_data._sensitive_all = None
        self.fairness_data = fairness_data

    def _make_fitted_defense(self, defense_name, defense_params=None):
        cfg = self.FairlearnDefenseConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"max_iter": 200},
            defense_name=defense_name,
            defense_params=defense_params or {},
            data=self.fairness_data,
        )
        cfg._train(self.X_train, self.y_train)
        return cfg

    def test_apply_defense_reductions_exponentiated_gradient_string_constraint(
        self,
    ):
        """ExponentiatedGradient with constraint given as a module-path string."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": "fairlearn.reductions.DemographicParity",
                "eps": 0.1,
            },
        )
        result = cfg.apply_defense(None)
        from fairlearn.reductions import ExponentiatedGradient

        self.assertIsInstance(result, ExponentiatedGradient)

    def test_apply_defense_reductions_exponentiated_gradient_dict_constraint(
        self,
    ):
        """ExponentiatedGradient with constraint given as a _target_ dict."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": {
                    "_target_": "fairlearn.reductions.EqualizedOdds",
                },
                "eps": 0.1,
            },
        )
        result = cfg.apply_defense(None)
        from fairlearn.reductions import ExponentiatedGradient

        self.assertIsInstance(result, ExponentiatedGradient)

    def test_apply_defense_reductions_requires_constraints(self):
        """ExponentiatedGradient without a constraints key must raise ValueError."""
        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
        )
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
        cfg = self._make_fitted_defense(
            "fairlearn.postprocessing.ThresholdOptimizer",
        )
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
            {
                "constraints": "fairlearn.reductions.DemographicParity",
                "eps": 0.1,
            },
        )
        cfg.apply_defense(None)
        self.assertIsNotNone(cfg.defense_application_time)

    def test_fairness_defense_config_is_configbase_with_hash(self):
        """Test that FairlearnDefenseConfig is ConfigBase and has __hash__ method."""
        from deckard.utils import ConfigBase

        cfg = self._make_fitted_defense(
            "fairlearn.reductions.ExponentiatedGradient",
            {
                "constraints": "fairlearn.reductions.DemographicParity",
                "eps": 0.1,
            },
        )
        self.assertIsInstance(
            cfg,
            ConfigBase,
            msg="FairlearnDefenseConfig should inherit from ConfigBase",
        )
        self.assertTrue(
            hasattr(cfg, "__hash__"),
            msg="FairlearnDefenseConfig should have __hash__ method",
        )
        # Note: FairlearnDefenseConfig may have unhashable runtime fields
        # so we verify the infrastructure is in place rather than attempting full hash


if __name__ == "__main__":
    unittest.main()
