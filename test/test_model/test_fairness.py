import unittest
import pandas as pd
import tempfile
import shutil
from deckard.model.fairness import FairnessModelConfig
from deckard.data.fairness import FairnessDataConfig

class TestFairnessModelConfig(unittest.TestCase):
    def setUp(self):
        # Create sample data with group information
        self.X_train = pd.DataFrame({
            "feature1": [0, 1, 2, 3, 4, 5],
            "feature2": [1, 2, 3, 4, 5, 6],
            "group": ["A", "B", "A", "B", "A", "B"]
        })
        self.y_train = pd.Series([0, 1, 0, 1, 0, 1])
        
        self.X_test = pd.DataFrame({
            "feature1": [6, 7, 8, 9],
            "feature2": [7, 8, 9, 10],
            "group": ["A", "B", "A", "B"]
        })
        self.y_test = pd.Series([1, 0, 1, 0])
        
        # Create groups Series for fairness evaluation
        self.groups = pd.Series(["A", "B", "A", "B"], index=self.y_test.index)
        
        self.model_type = "sklearn.ensemble.RandomForestClassifier"
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_fairness_model_config_initialization(self):
        """Test FairnessModelConfig can be initialized."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = self.groups
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.data, fairness_data)

    def test_fairness_model_config_initialization_without_data(self):
        """Test FairnessModelConfig can be initialized without fairness data."""
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None
        )
        
        self.assertIsNotNone(model)
        self.assertIsNone(model.data)

    def test_classification_scores_without_fairness_data(self):
        """Test classification scores when fairness_data is None."""
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=None
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
        fairness_data.groups = self.groups
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
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
            data=None
        )
        
        scores = model._regression_scores(y_true, y_pred)
        
        self.assertIsInstance(scores, dict)
        self.assertIn("mse", scores)
        # Should not contain group-specific scores
        self.assertNotIn("A_mse", scores)
        self.assertNotIn("B_mse", scores)

    def test_regression_scores_with_fairness_data(self):
        """Test regression scores includes group fairness metrics."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0], index=self.groups.index)
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8], index=self.groups.index)
        
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = self.groups
        
        model = FairnessModelConfig(
            model_type="sklearn.linear_model.LinearRegression",
            classifier=False,
            data=fairness_data
        )
        
        scores = model._regression_scores(y_true, y_pred)
        
        self.assertIsInstance(scores, dict)
        self.assertIn("mse", scores)
        # Should contain group-specific scores
        self.assertIn("A_mse", scores)
        self.assertIn("B_mse", scores)

    def test_compute_group_fairness_scores_no_groups(self):
        """Test group fairness scores with no groups attribute."""
        fairness_data = Mock(spec=FairnessDataConfig)
        del fairness_data.groups  # Remove groups attribute
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
        )
        
        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)
        
        self.assertEqual(scores, {})

    def test_compute_group_fairness_scores_none_groups(self):
        """Test group fairness scores when groups is None."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = None
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
        )
        
        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)
        
        self.assertEqual(scores, {})

    def test_compute_group_fairness_scores_empty_group(self):
        """Test group fairness scores skips empty groups."""
        # Create groups with different sizes
        groups = pd.Series(["A", "A", "A", "B"], index=self.y_test.index)
        
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = groups
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
        )
        
        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)
        
        self.assertIsInstance(scores, dict)
        # Should have scores for both groups
        self.assertTrue(any("A_" in key for key in scores.keys()))
        self.assertTrue(any("B_" in key for key in scores.keys()))

    def test_compute_group_fairness_scores_classification(self):
        """Test group fairness scores for classification task."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = self.groups
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
        )
        
        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)
        
        self.assertIsInstance(scores, dict)
        # Check for classification metrics per group
        for metric in ["accuracy", "precision", "recall", "f1-score"]:
            self.assertTrue(any(f"A_{metric}" in key or f"B_{metric}" in key for key in scores.keys()))

    def test_compute_group_fairness_scores_regression(self):
        """Test group fairness scores for regression task."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0], index=self.groups.index)
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8], index=self.groups.index)
        
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = self.groups
        
        model = FairnessModelConfig(
            model_type="sklearn.linear_model.LinearRegression",
            classifier=False,
            data=fairness_data
        )
        
        scores = model._compute_group_fairness_scores(y_true, y_pred)
        
        self.assertIsInstance(scores, dict)
        # Check for regression metrics per group
        for metric in ["mse", "rmse", "mae"]:
            self.assertTrue(any(f"A_{metric}" in key or f"B_{metric}" in key for key in scores.keys()))

    def test_group_fairness_scores_naming_convention(self):
        """Test that group fairness scores follow naming convention."""
        fairness_data = Mock(spec=FairnessDataConfig)
        fairness_data.groups = self.groups
        
        model = FairnessModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
            data=fairness_data
        )
        
        scores = model._compute_group_fairness_scores(self.y_test, self.y_test)
        
        # Check naming convention: {group_name}_{metric}
        for key in scores.keys():
            self.assertTrue("_" in key, f"Key {key} should contain group_metric format")
            parts = key.split("_")
            self.assertGreaterEqual(len(parts), 2, f"Key {key} should have group and metric")


if __name__ == "__main__":
    unittest.main()