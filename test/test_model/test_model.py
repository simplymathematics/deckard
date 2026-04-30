import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from deckard.model import ModelConfig
from deckard.data import DataConfig
from sklearn.ensemble import RandomForestClassifier


class TestModelConfig(unittest.TestCase):
    def setUp(self):
        # Simple binary classification data
        self.X_train = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, 4]})
        self.y_train = pd.Series([0, 1, 0, 1])
        self.X_test = pd.DataFrame({"a": [4, 5], "b": [5, 6]})
        self.y_test = pd.Series([1, 0])
        self.model_params = {"probability": True}
        self.model_type = "sklearn.ensemble.RandomForestClassifier"
        self.model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        self.tmpdir = tempfile.mkdtemp()
        self.model_file = os.path.join(self.tmpdir, "model.pkl")
        self.pred_file = os.path.join(self.tmpdir, "preds.pkl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_post_init(self):
        self.assertTrue(hasattr(self.model._model, "fit"))
        self.assertTrue(hasattr(self.model._model, "predict"))

    def test_train_and_predict(self):
        self.model._train(self.X_train, self.y_train)
        preds = self.model._predict(self.X_train)
        self.assertEqual(len(preds), len(self.y_train))

    def test_predict_proba(self):
        self.model._train(self.X_train, self.y_train)
        self.model.probability = True
        proba = self.model._predict_proba(self.X_train)
        self.assertEqual(len(proba), len(self.y_train))

    def test_classification_scores(self):
        scores = self.model._classification_scores(self.y_train, self.y_train)
        self.assertIn("accuracy", scores)
        self.assertIn("precision", scores)
        self.assertIn("recall", scores)
        self.assertIn("f1-score", scores)

    def test_classification_scores_handles_single_column_binary_outputs(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred_single_col = np.array([[0.1], [0.9], [0.2], [0.8]])
        scores = self.model._classification_scores(y_true, y_pred_single_col)
        self.assertIn("accuracy", scores)
        self.assertGreaterEqual(scores["accuracy"], 0.99)

    def test_regression_scores(self):
        # Use regression scores with float values
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 1.9, 3.2])
        scores = self.model._regression_scores(y_true, y_pred)
        self.assertIn("mse", scores)
        self.assertIn("rmse", scores)
        self.assertIn("mae", scores)

    def test_score(self):
        self.model._train(self.X_train, self.y_train)
        preds = self.model._predict(self.X_train)
        scores = self.model._score(self.y_train, preds)
        self.assertIsInstance(scores, dict)
        self.assertIn("accuracy", scores)

    def test_call_training_and_prediction(self):
        data = DataConfig()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        data()
        score_dict = model(data=data, model_file=self.model_file)
        scores = model.score_dict
        self.assertIsInstance(scores, dict)
        self.assertTrue("training_time" in scores and "prediction_time" in scores)
        self.assertTrue("accuracy" in scores)
        self.assertTrue("training_time" in scores)
        self.assertTrue("prediction_time" in scores)
        self.assertTrue(hasattr(model, "score_dict"))
        self.assertTrue(hasattr(model, "training_time"))
        self.assertTrue(hasattr(model, "training_score_time"))
        self.assertTrue(hasattr(model, "prediction_score_time"))
        self.assertTrue(hasattr(model, "prediction_time"))
        self.assertTrue(hasattr(model, "training_predictions"))
        self.assertTrue(hasattr(model, "predictions"))
        # Assert that the keys in score dict are also in the model.score_dit
        for key in score_dict:
            self.assertIn(key, scores)
        for key in scores:
            self.assertIn(key, score_dict)

    def test_call_saves_test_predictions_when_file_requested(self):
        data = DataConfig(dataset_name="make_classification", data_params={"n_samples": 40, "n_features": 4, "n_informative": 2, "n_redundant": 0})
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        test_pred_file = os.path.join(self.tmpdir, "test_predictions.pkl")
        model(data=data, model_file=self.model_file, test_predictions_file=test_pred_file)
        self.assertTrue(os.path.exists(test_pred_file))

    def test_call_saves_train_predictions_when_file_requested(self):
        data = DataConfig(dataset_name="make_classification", data_params={"n_samples": 40, "n_features": 4, "n_informative": 2, "n_redundant": 0})
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        train_pred_file = os.path.join(self.tmpdir, "train_predictions.pkl")
        model(data=data, model_file=self.model_file, train_predictions_file=train_pred_file)
        self.assertTrue(os.path.exists(train_pred_file))

    def test_call_saves_train_and_test_predictions_when_requested(self):
        data = DataConfig(dataset_name="make_classification", data_params={"n_samples": 40, "n_features": 4, "n_informative": 2, "n_redundant": 0})
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        train_pred_file = os.path.join(self.tmpdir, "train_predictions.pkl")
        test_pred_file = os.path.join(self.tmpdir, "test_predictions.pkl")
        model(
            data=data,
            model_file=self.model_file,
            train_predictions_file=train_pred_file,
            test_predictions_file=test_pred_file,
        )
        self.assertTrue(os.path.exists(train_pred_file))
        self.assertTrue(os.path.exists(test_pred_file))

    def test_call_saves_test_probabilities_when_file_requested(self):
        data = DataConfig(dataset_name="make_classification", data_params={"n_samples": 40, "n_features": 4, "n_informative": 2, "n_redundant": 0})
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        test_prob_file = os.path.join(self.tmpdir, "test_probabilities.pkl")
        model(data=data, model_file=self.model_file, test_probabilities_file=test_prob_file)
        self.assertTrue(os.path.exists(test_prob_file))

    def test_call_saves_train_probabilities_when_file_requested(self):
        data = DataConfig(dataset_name="make_classification", data_params={"n_samples": 40, "n_features": 4, "n_informative": 2, "n_redundant": 0})
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        train_prob_file = os.path.join(self.tmpdir, "train_probabilities.pkl")
        model(data=data, model_file=self.model_file, training_probabilities_file=train_prob_file)
        self.assertTrue(os.path.exists(train_prob_file))

    def test_load_predictions(self):
        preds = np.array([0, 1, 1, 0])
        pred_file = os.path.join(self.tmpdir, "preds.npy")
        np.save(pred_file, preds)
        # Patch load_data to use np.load for this test
        orig_load_data = self.model.load_data
        self.model.load_data = lambda fp: np.load(fp)
        loaded = self.model._load_predictions(pred_file)
        self.assertTrue(np.array_equal(loaded, preds))
        self.model.load_data = orig_load_data

    def test_load_or_train_model_trains_when_not_fitted_even_if_training_time_set(self):
        data = DataConfig(dataset_name="make_classification", data_params={"n_samples": 40, "n_features": 4, "n_informative": 2, "n_redundant": 0})
        data()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 5},
        )
        model.training_time = 1.23
        model._model = RandomForestClassifier(n_estimators=5)
        times = {}
        times = model._load_or_train_model(data, model_file=None, times=times)
        self.assertIn("training_time", times)
        self.assertIn("training_n", times)
        self.assertEqual(times["training_n"], len(data.y_train))

    def test_predict_falls_back_when_wrapped_output_matrix_is_invalid(self):
        class BasePredictor:
            def predict(self, X):
                return np.array([0, 1])

        class WrappedPredictor:
            def __init__(self):
                self.model = BasePredictor()

            def predict(self, X):
                return np.array([[1.0, 1.0], [1.0, 1.0]])

        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        model._model = WrappedPredictor()

        preds = model._predict(self.X_test)
        self.assertTrue(np.array_equal(preds, np.array([0, 1])))

    def test_decode_predictions_for_persistence_converts_2d_classifier_output(self):
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([[1.0, 1.0], [0.0, 2.0], [2.0, 0.0], [1.0, 1.0]])
        decoded = model._decode_predictions_for_persistence(y_pred, y_true=y_true)
        self.assertEqual(decoded.ndim, 1)
        self.assertEqual(len(decoded), len(y_true))


if __name__ == "__main__":
    unittest.main()
