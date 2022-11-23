import json
import os
import shutil
import tempfile
import unittest
import yaml
import numpy as np
from deckard.base import Data

names = ["mnist", "cifar10", "iris"]
# TODO other names


class testData(unittest.TestCase):
    def setUp(self):
        self.filename = "test_data.pkl"
        self.path = tempfile.mkdtemp()
        yaml.add_constructor("!Data:", Data)
        self.data_document = """
        !Data:
            params:
                shuffle : True
                random_state : 42
                train_size : 800
                stratify : True
                train_noise : 1
                time_series : True
                name: iris
        """
        # Test that data yaml loads correctly
        self.data = yaml.load(self.data_document, Loader=yaml.Loader)


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
        self.assertIsInstance(data.params["name"], str)
        self.assertIsInstance(data.params["train_size"], (float, int))
        self.assertIsInstance(data.params["random_state"], int)
        self.assertIsInstance(data.params["shuffle"], bool)
        self.assertIsInstance(data.params["time_series"], bool)

    def test_load(self):
        data = self.data
        data = data.load()
        self.assertIsInstance(data.X_train, np.ndarray)
        self.assertIsInstance(data.y_train, np.ndarray)
        self.assertIsInstance(data.X_test, np.ndarray)
        self.assertIsInstance(data.y_test, np.ndarray)
        self.assertEqual(data.X_train.shape[0], data.y_train.shape[0])


    def test_sample_data(self):
        document = """
        !Data:
            params:
                shuffle : True
                random_state : 220
                train_size : 800
                stratify : True
                train_noise : 1
                time_series : True
                name: iris
        """
        data2 = yaml.load(document, Loader=yaml.Loader)
        self.assertNotEqual(self.data, data2)


    def test_save_data(self):
        data = self.data
        data.save(filename=os.path.join(self.path, self.filename), data = self.data.load())
        self.assertTrue(os.path.exists(os.path.join(self.path, self.filename)))

    def tearDown(self):
        shutil.rmtree(self.path)
