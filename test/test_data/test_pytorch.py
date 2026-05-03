import unittest
import tempfile
import shutil
from pathlib import Path
import pytest
from torch.utils.data import Dataset
from unittest.mock import patch

torch = pytest.importorskip("torch")
Tensor = pytest.importorskip("torch").Tensor
PytorchDataConfig = pytest.importorskip("deckard.data.pytorch").PytorchDataConfig


class TensorWithSensitiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        sensitive = int(y.item() % 2)
        return x, y, sensitive


class TensorWithDatasetSensitiveAttr(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self._sensitive = [int(i % 2) for i in range(len(y))]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TestPytorchDataConfig(unittest.TestCase):

    def setUp(self):
        X = torch.randn(300, 4)
        y = torch.randint(0, 2, (300,))
        self.config = PytorchDataConfig(
            dataset_name="torch.utils.data.TensorDataset",
            data_dir=self.temp_dir,
            test_size=100,
            train_size=100,
            random_state=42,
            data_params={"_args_": [X, y]},
        )

    @classmethod
    def setUpClass(cls):
        # Create temporary directory for data storage
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_initialization(self):
        self.assertEqual(self.config.dataset_name, "torch.utils.data.TensorDataset")
        self.assertEqual(self.config.data_dir, self.temp_dir)
        self.assertEqual(self.config.test_size, 100)
        self.assertEqual(self.config.train_size, 100)
        self.assertEqual(self.config.random_state, 42)
        self.assertTrue(self.config.stratify)

    def test_load_data(self):
        self.config._load_data()
        self.assertIsInstance(self.config._X, Tensor)
        self.assertIsInstance(self.config._y, Tensor)
        self.assertGreater(self.config.data_load_time, 0)

    def test_sample(self):
        self.config._load_data()
        self.config._sample()
        self.assertIsInstance(self.config.X_train, Tensor)
        self.assertIsInstance(self.config.y_train, Tensor)
        self.assertIsInstance(self.config.X_test, Tensor)
        self.assertIsInstance(self.config.y_test, Tensor)

    def test_call(self):
        scores = self.config(data_file=str(Path(self.temp_dir) / "data.pkl"))
        self.assertIn("data_load_time", scores)
        self.assertIn("data_sample_time", scores)
        self.assertGreater(scores["data_load_time"], 0)
        self.assertGreater(scores["data_sample_time"], 0)

    def test_invalid_dataset_name(self):
        self.config.dataset_name = "invalid_dataset"
        with self.assertRaises(Exception):
            self.config._load_data()

    def test_hash_method(self):
        h1 = hash(self.config)
        h2 = hash(self.config)
        self.assertEqual(h1, h2)

    def test_load_data_collects_sensitive_from_third_tuple_item(self):
        X = torch.randn(120, 4)
        y = torch.randint(0, 2, (120,))
        ds = TensorWithSensitiveDataset(X, y)
        cfg = PytorchDataConfig(
            dataset_name="dummy.dataset",
            data_dir=self.temp_dir,
            train_size=60,
            test_size=40,
            stratify=True,
            data_params={},
        )

        with patch("deckard.data.pytorch.load_class", return_value=ds):
            cfg._load_data()
        self.assertTrue(hasattr(cfg, "_sensitive"))
        self.assertEqual(len(cfg._sensitive), len(y))

        cfg._sample()
        self.assertEqual(len(cfg._sensitive_train), 60)
        self.assertEqual(len(cfg._sensitive_test), 40)

    def test_load_data_collects_sensitive_from_dataset_attribute(self):
        X = torch.randn(110, 4)
        y = torch.randint(0, 2, (110,))
        ds = TensorWithDatasetSensitiveAttr(X, y)
        cfg = PytorchDataConfig(
            dataset_name="dummy.dataset",
            data_dir=self.temp_dir,
            train_size=70,
            test_size=20,
            stratify=True,
            data_params={},
        )

        with patch("deckard.data.pytorch.load_class", return_value=ds):
            cfg._load_data()
        self.assertTrue(hasattr(cfg, "_sensitive"))
        self.assertEqual(len(cfg._sensitive), len(y))

        cfg._sample()
        self.assertEqual(len(cfg._sensitive_train), 70)
        self.assertEqual(len(cfg._sensitive_test), 20)

    def test_pytorch_data_hash_stable_after_call_and_runtime_mutation(self):
        original_hash = hash(self.config)
        self.config()
        self.config.score_dict["runtime"] = 1
        self.config.data_load_time = 999.0
        self.assertEqual(hash(self.config), original_hash)

    def test_pytorch_data_score_persistence_roundtrip(self):
        self.config._load_data()
        self.config._sample()
        scores = self.config._score()

        score_path = Path(self.temp_dir) / "pytorch_data_scores.json"
        self.config.save_scores(scores, str(score_path))
        loaded = self.config.load_scores(str(score_path))

        self.assertTrue(len(loaded) > 0)
        self.assertIn("class_counts", loaded)


if __name__ == "__main__":
    unittest.main()
    # Remove temporary directory after tests
