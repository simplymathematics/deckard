import unittest
import tempfile
import shutil
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
Tensor = pytest.importorskip("torch").Tensor
PytorchDataConfig = pytest.importorskip("deckard.layers.pytorch").PytorchDataConfig


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


if __name__ == "__main__":
    unittest.main()
    # Remove temporary directory after tests
