import unittest
import tempfile
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
from deckard.data import DataConfig, DataPipelineConfig

class TestDataPipelineConfig(unittest.TestCase):
    def setUp(self):
        self.pipeline_config_dict = {
            "imputer": {"name": "sklearn.impute.SimpleImputer", "strategy": "mean"},
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
        }
        self.X_train = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0],
                "feature2": [np.nan, 1.0, 2.0, 3.0],
            }
        )
        self.y_train = pd.Series([0, 1, 0, 1])
        self.X_test = pd.DataFrame(
            {
                "feature1": [5.0, 6.0],
                "feature2": [4.0, np.nan],
            }
        )
        self.y_test = pd.Series([1, 0])

    def test_pipeline_initialization(self):
        config = DataPipelineConfig(pipeline=self.pipeline_config_dict)
        self.assertIsInstance(config.pipeline, Pipeline)
        self.assertEqual(len(config.pipeline.steps), 2)
        self.assertEqual(config.pipeline.steps[0][0], "imputer")
        self.assertEqual(config.pipeline.steps[1][0], "scaler")

    def test_pipeline_fit_and_transform(self):
        config = DataPipelineConfig(pipeline=self.pipeline_config_dict)
        X_train_transformed, X_test_transformed, _, _ = config(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertEqual(X_train_transformed.shape, (4, 2))
        self.assertEqual(X_test_transformed.shape, (2, 2))
        self.assertFalse(np.isnan(X_train_transformed).any())
        self.assertFalse(np.isnan(X_test_transformed).any())

    def test_pipeline_fit_time(self):
        config = DataPipelineConfig(pipeline=self.pipeline_config_dict)
        config(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(config.pipeline_fit_time)
        self.assertGreater(config.pipeline_fit_time, 0)

    def test_pipeline_transform_time(self):
        config = DataPipelineConfig(pipeline=self.pipeline_config_dict)
        config(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(config.pipeline_transform_time)
        self.assertGreater(config.pipeline_transform_time, 0)

    def test_invalid_pipeline_config(self):
        invalid_pipeline_config = {"step1": {"name": "InvalidModule.InvalidClass"}}
        with self.assertRaises(ModuleNotFoundError):
            DataPipelineConfig(pipeline=invalid_pipeline_config)

class TestDataConfig(unittest.TestCase):

    def basic_config(self):
        # Minimal config for DataConfig
        return DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 100,
                "n_features": 5,
                "n_informative": 1,
                "n_redundant": 0,
                "random_state": 42,
                "n_clusters_per_class": 1,
            },
            test_size=0.2,
            random_state=42,
            stratify=True,
            classifier=True,
        )

    def test_make_classification_data_loading_and_sampling(self):
        cfg = self.basic_config()
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        total = len(X_train) + len(X_test)
        self.assertEqual(total, 100)

    def test_make_regression_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="make_regression",
            data_params={
                "n_samples": 50,
                "n_features": 4,
                "n_informative": 2,
                "random_state": 1,
            },
            test_size=0.3,
            random_state=1,
            stratify=None,
            classifier=False,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), 50)

    def test_diabetes_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="diabetes",
            data_params={},
            test_size=0.25,
            random_state=0,
            stratify=None,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), len(cfg._X))

    def test_digits_data_loading_and_sampling(self):
        cfg = DataConfig(
            dataset_name="digits",
            data_params={},
            test_size=0.1,
            random_state=123,
            stratify=True,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), len(cfg._X))

    def test_hash_method_is_consistent(self):
        cfg = self.basic_config()
        h1 = hash(cfg)
        h2 = hash(cfg)
        self.assertEqual(h1, h2)

    def test_sample_raises_value_error_if_data_not_loaded(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 6,
                "n_informative": 4,
                "random_state": 7,
                "n_redundant": 0,
                "n_repeated": 0,
            },
        )
        cfg._X = None
        cfg._y = None
        with self.assertRaises(ValueError):
            cfg._sample()

    def test_load_data_raises_not_implemented_for_unknown_dataset(self):
        cfg = DataConfig(dataset_name="unknown_dataset", data_params={})
        with self.assertRaises(NotImplementedError):
            cfg._load_data()

    def test_load_data_raises_value_error_for_csv_without_target(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            csv_path = Path(tmpdirname) / "test.csv"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
            cfg = DataConfig(dataset_name=str(csv_path), data_params={})
            with self.assertRaises(ValueError):
                cfg._load_data()

    def test_call_returns_expected_shapes_for_make_classification(self):
        cfg = DataConfig(
            dataset_name="make_classification",
            data_params={
                "n_samples": 60,
                "n_features": 6,
                "n_informative": 4,
                "random_state": 7,
                "n_redundant": 0,
            },
            test_size=0.5,
            random_state=7,
            stratify=True,
        )
        cfg()
        X_train = cfg.X_train
        y_train = cfg.y_train
        X_test = cfg.X_test
        y_test = cfg.y_test
        self.assertEqual(X_train.shape[0], 30)
        self.assertEqual(X_test.shape[0], 30)
        self.assertEqual(X_train.shape[1], 6)
        self.assertEqual(X_test.shape[1], 6)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), 60)

    def test_save_self(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            score_path = Path(tmpdirname) / "scores.json"
            results = cfg(
                data_file=str(data_path),
                score_file=str(score_path),
            )
            self.assertTrue(data_path.exists())
            self.assertTrue(score_path.exists())
            self.assertIn("data_load_time", results)
            self.assertIn("data_sample_time", results)

    def test_load_self(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg(data_file=str(data_path))
            self.assertTrue(cfg._X is not None)

    def test_save_score_dict(self):
        cfg = self.basic_config()
        cfg()
        cfg.score_dict = {"mutual_info": 0.95, "chisquare": 0.9}
        with tempfile.TemporaryDirectory() as tmpdirname:
            score_path = Path(tmpdirname) / "scores.json"
            # save scores
            cfg.save_scores(cfg.score_dict, score_path)
            loaded_scores = cfg.load_scores(score_path)
            cfg(score_file=str(score_path))
            self.assertTrue(score_path.exists())
            self.assertIn("mutual_info", loaded_scores)
            self.assertIn("chisquare", loaded_scores)
            self.assertAlmostEqual(loaded_scores["mutual_info"], 0.95)
            self.assertAlmostEqual(loaded_scores["chisquare"], 0.9)

    def test_save_data_file(self):
        import tempfile

        cfg = self.basic_config()
        cfg()
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg(data_file=str(data_path))
            self.assertTrue(data_path.exists())
            # Load the data back and verify
            cfg = cfg.load(filepath=str(data_path))
            self.assertIsNotNone(cfg._X)
            self.assertIsNotNone(cfg._y)


if __name__ == "__main__":
    unittest.main()
