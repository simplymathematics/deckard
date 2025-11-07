import unittest
import tempfile
import shutil
from pathlib import Path
from deckard.data.pytorch import PytorchDataConfig
from torch import Tensor


class TestPytorchDataConfig(unittest.TestCase):

    def setUp(self):
        self.config = PytorchDataConfig(
            dataset_name="mnist",
            data_dir=self.temp_dir,
            test_size=0.2,
            train_size=0.7,
            random_state=42,
        )

    @classmethod
    def setUpClass(cls):
        # Create temporary directory for data storage
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_initialization(self):
        self.assertEqual(self.config.dataset_name, "mnist")
        self.assertEqual(self.config.data_dir, self.temp_dir)
        self.assertEqual(self.config.test_size, 0.2)
        self.assertEqual(self.config.train_size, 0.7)
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
        scores = self.config(data_file=str(Path(self.temp_dir) / "data.pt"))
        self.assertIn("data_load_time", scores)
        self.assertIn("data_sample_time", scores)
        self.assertGreater(scores["data_load_time"], 0)
        self.assertGreater(scores["data_sample_time"], 0)

    def test_invalid_dataset_name(self):
        self.config.dataset_name = "invalid_dataset"
        with self.assertRaises(NotImplementedError):
            self.config._load_data()

    def test_hash_method(self):
        h1 = hash(self.config)
        h2 = hash(self.config)
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
    # Remove temporary directory after tests
