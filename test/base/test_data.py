import shutil
import unittest
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path
from deckard.base.data import Data, config

names = ["mnist", "cifar10", "iris"]
# TODO other names


class testData(unittest.TestCase):
    def setUp(self, config=config):
        yaml.add_constructor("!Data:", Data)
        self.data_document = "!Data:\n" + config
        self.data = yaml.load(self.data_document, Loader=yaml.Loader)
        self.path = Path(__file__).parent / "data"
        self.filename = Path(self.path, "data.pkl")

    def test_hash(self):
        data1 = yaml.load(self.data_document, Loader=yaml.Loader)
        data2 = yaml.load(self.data_document, Loader=yaml.Loader)
        self.assertEqual(hash(str(data1)), hash(str(data2)))

    def test_eq(self):
        # Test that data yaml loads correctly
        data1 = yaml.load(self.data_document, Loader=yaml.Loader)
        data2 = yaml.load(self.data_document, Loader=yaml.Loader)
        self.assertEqual(data1, data2)

    def test_get_params(self):
        data = self.data
        params = data._asdict()["sample"]
        self.assertIsInstance(params["train_size"], (float, int))
        self.assertIsInstance(params["random_state"], int)
        self.assertIsInstance(params["shuffle"], bool)

    def test_load(self):
        data = self.data
        data = data.load(self.filename)
        self.assertIsInstance(data.X_train, np.ndarray)
        self.assertIsInstance(data.y_train, np.ndarray)
        self.assertIsInstance(data.X_test, np.ndarray)
        self.assertIsInstance(data.y_test, np.ndarray)
        self.assertEqual(data.X_train.shape[0], data.y_train.shape[0])

    def test_sample_data(self):
        document = deepcopy(self.data_document)
        document = document.replace("42", "43")
        data2 = yaml.load(document, Loader=yaml.Loader)
        self.assertNotEqual(self.data, data2)

    def tearDown(self):
        if Path(self.path).exists():
            shutil.rmtree(Path(self.path))
        if Path(self.path).is_dir():
            rmtree(self.path)
        if Path("model").is_dir():
            rmtree("model")
        if Path("data").is_dir():
            rmtree("data")
        if Path("reports").is_dir():
            rmtree("reports")
        del self.path
