import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import shutil
from deckard.model import ModelConfig
from deckard.data import DataConfig


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
        self.assertEqual(proba.shape[0], len(self.y_train))

    def test_classification_scores(self):
        scores = self.model._classification_scores(self.y_train, self.y_train)
        self.assertIn("accuracy", scores)
        self.assertIn("precision", scores)
        self.assertIn("recall", scores)
        self.assertIn("f1-score", scores)

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

    def test_save_and_load_model(self):
        self.model._train(self.X_train, self.y_train)
        self.model._save_model(self.model_file)
        self.assertTrue(Path(self.model_file).exists())
        loaded_model = ModelConfig(model_type=self.model_type, classifier=True)
        loaded_model._load_model(self.model_file)
        self.assertTrue(hasattr(loaded_model._model, "predict"))

    def test_call_training_and_prediction(self):
        data = DataConfig()
        model = ModelConfig(
            model_type=self.model_type,
            classifier=True,
            model_params={"n_estimators": 10},
        )
        data()
        score_dict = model(data=data, model_filepath=self.model_file)
        scores = model.score_dict
        self.assertIsInstance(scores, dict)
        self.assertTrue("training_time" in scores and "prediction_time" in scores)
        self.assertTrue("accuracy" in scores)

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


if __name__ == "__main__":
    unittest.main()
