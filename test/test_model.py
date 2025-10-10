import unittest
import pandas as pd
import numpy as np
import os
from deckard.model import ModelConfig, initialize_model_config, train_and_evaluate

class DummyArgs:
    data_filepath = None
    model_filepath = "dummy_model.pkl"
    model_config_file = None
    model_params = None
    probability = False

class Data:
    pass

def get_sample_data():
    X = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20)
    })
    y = pd.Series(np.random.randint(0, 2, size=20))
    data = Data()
    data._X_train = X.iloc[:15]
    data._y_train = y.iloc[:15]
    data._X_test = X.iloc[15:]
    data._y_test = y.iloc[15:]
    return data

class TestModelConfig(unittest.TestCase):

    def setUp(self):
        self.sample_data = get_sample_data()
        self.tmp_file = "temp_model.pkl"

    def tearDown(self):
        if os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)

    def test_modelconfig_init_and_train_predict(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        preds = config._predict(self.sample_data._X_test)
        self.assertEqual(len(preds), len(self.sample_data._y_test))
        scores = config._classification_scores(self.sample_data._y_test, preds)
        self.assertIn("accuracy", scores)

    def test_modelconfig_regression_scores(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 1.9, 3.2])
        config = ModelConfig(model_type="sklearn.linear_model.LinearRegression", classifier=False)
        scores = config._regression_scores(y_true, y_pred)
        self.assertIn("mse", scores)
        self.assertIn("rmse", scores)
        self.assertIn("mae", scores)

    def test_modelconfig_hash_changes(self):
        config1 = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config2 = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=False)
        self.assertNotEqual(hash(config1), hash(config2))

    def test_initialize_model_config_default(self):
        model = initialize_model_config()
        self.assertIsInstance(model, ModelConfig)

    def test_train_and_evaluate_classification(self):
        args = DummyArgs()
        args.model_filepath = self.tmp_file
        train_scores, test_scores, trained_model = train_and_evaluate(args, train=True, score=True, data=self.sample_data)
        self.assertTrue("train_accuracy" in train_scores or "train_f1-score" in train_scores)
        self.assertIsNotNone(trained_model)

    def test_model_save_and_load(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        config._save_model(self.tmp_file)
        config._model = None
        config._load_model(self.tmp_file)
        self.assertIsNotNone(config._model)


    def test_svc(self):
        config = ModelConfig(model_type="sklearn.svm.SVC", classifier=True, probability=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        preds = config._predict(self.sample_data._X_test)
        self.assertEqual(len(preds), len(self.sample_data._y_test))
        scores = config._classification_scores(self.sample_data._y_test, preds)
        self.assertIn("accuracy", scores)

    def test_regression_model(self):
        config = ModelConfig(model_type="sklearn.linear_model.LinearRegression", classifier=False)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        preds = config._predict(self.sample_data._X_test)
        self.assertEqual(len(preds), len(self.sample_data._y_test))
        scores = config._regression_scores(self.sample_data._y_test, preds)
        self.assertIn("mse", scores)
        self.assertIn("rmse", scores)
        self.assertIn("mae", scores)

    def test_modelconfig_train_with_filepath(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        config._save_model(self.tmp_file)
        self.assertTrue(os.path.exists(self.tmp_file))
        config_loaded = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config_loaded._load_model(self.tmp_file)
        self.assertIsNotNone(config_loaded._model)
        preds = config_loaded._predict(self.sample_data._X_test)
        self.assertEqual(len(preds), len(self.sample_data._y_test))

    # Additional tests

    def test_modelconfig_probability_flag(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True, probability=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        try:
            proba = config._predict_proba(self.sample_data._X_test)
            self.assertEqual(proba.shape[0], len(self.sample_data._y_test))
        except Exception as e:
            self.fail(f"Probability prediction failed: {e}")

    def test_modelconfig_invalid_model_type(self):
        with self.assertRaises(AssertionError):
            ModelConfig(model_type="xgboost.XGBClassifier", classifier=True)

    def test_modelconfig_save_without_training(self):
        config = ModelConfig()
        with self.assertRaises(ValueError):
            config._save_model(self.tmp_file)

    def test_modelconfig_load_invalid_file(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        with open(self.tmp_file, "wb") as f:
            f.write(b"not a model")
        with self.assertRaises(Exception):
            config._load_model(self.tmp_file)

    def test_modelconfig_score_train_flag(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        preds = config._predict(self.sample_data._X_train)
        scores = config._score(self.sample_data._y_train, preds, train=True)
        self.assertTrue(any(k.startswith("train_") for k in scores.keys()))

    def test_modelconfig_score_test_flag(self):
        config = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
        config._train(self.sample_data._X_train, self.sample_data._y_train)
        preds = config._predict(self.sample_data._X_test)
        scores = config._score(self.sample_data._y_test, preds, train=False)
        self.assertFalse(any(k.startswith("train_") for k in scores.keys()))

if __name__ == "__main__":
    unittest.main()
