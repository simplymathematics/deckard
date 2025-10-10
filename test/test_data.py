import unittest
import pandas as pd
from pathlib import Path
from deckard.data import DataConfig
class TestDataConfig(unittest.TestCase):

    def basic_config(self):
        # Minimal config for DataConfig
        return DataConfig(
            dataset_name="make_classification",
            data_params={"n_samples": 100, "n_features": 5, "n_informative": 1, "n_redundant": 0, "random_state": 42, "n_clusters_per_class": 1 },
            test_size=0.2,
            random_state=42,
            stratify=True
        )

    def test_make_classification_data_loading_and_sampling(self):
        cfg = self.basic_config()
        X_train, y_train, X_test, y_test = cfg()
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
            data_params={"n_samples": 50, "n_features": 4, "n_informative": 2, "random_state": 1},
            test_size=0.3,
            random_state=1,
            stratify=None
        )
        X_train, y_train, X_test, y_test = cfg()
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
            stratify=None
        )
        X_train, y_train, X_test, y_test = cfg()
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
            stratify=True
        )
        X_train, y_train, X_test, y_test = cfg()
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
        cfg = DataConfig(dataset_name="make_classification", data_params={"n_samples": 60, "n_features": 6, "n_informative": 4, "random_state": 7, "n_redundant": 0, "n_repeated": 0})
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
            data_params={"n_samples": 60, "n_features": 6, "n_informative": 4, "random_state": 7, "n_redundant": 0},
            test_size=0.5,
            random_state=7,
            stratify=True
        )
        X_train, y_train, X_test, y_test = cfg()
        self.assertEqual(X_train.shape[1], 6)
        self.assertEqual(X_test.shape[1], 6)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), 60)

if __name__ == "__main__":
    unittest.main()