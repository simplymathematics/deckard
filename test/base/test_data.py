import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from deckard.base import Data

datasets = ["mnist", "cifar10", "iris"]
# TODO other datasets


class testData(unittest.TestCase):
    def setUp(self):
        self.filename = "test_data.pkl"
        self.path = tempfile.mkdtemp()
        self.data = Data("mnist", train_size=100, random_state=0, shuffle=True)

    def test_init(self):
        """
        Validates data object.
        """
        for dataset in datasets:
            data = Data(dataset, train_size=100, random_state=0, shuffle=True)
            self.assertIsInstance(data, Data)
            self.assertIsInstance(data.params, dict)
            if isinstance(data.params, dict):
                self.assertIsInstance(data.params["dataset"], str)
                self.assertIsInstance(data.params["train_size"], (float, int))
                self.assertIsInstance(data.params["random_state"], int)
                self.assertIsInstance(data.params["shuffle"], bool)
                self.assertIsInstance(data.params["time_series"], bool)

    def test_hash(self):
        data1 = Data("mnist", train_size=100, random_state=0, shuffle=True)
        data2 = Data("mnist", train_size=100, random_state=0, shuffle=True)
        data3 = Data("cifar10", train_size=100, random_state=0, shuffle=True)
        self.assertEqual(data1.__hash__(), data2.__hash__())
        self.assertNotEqual(data3.__hash__(), data2.__hash__())

    def test_eq(self):
        data1 = Data("mnist", train_size=100, random_state=0, shuffle=True)
        data2 = Data("mnist", train_size=100, random_state=0, shuffle=True)
        data3 = Data("cifar10", train_size=100, random_state=0, shuffle=True)
        self.assertEqual(data1, data2)
        self.assertNotEqual(data1, data3)

    def test_get_params(self):
        data = Data("mnist", train_size=100, random_state=0, shuffle=True)
        self.assertIsInstance(data.params["dataset"], str)
        self.assertIsInstance(data.params["train_size"], (float, int))
        self.assertIsInstance(data.params["random_state"], int)
        self.assertIsInstance(data.params["shuffle"], bool)
        self.assertIsInstance(data.params["target"], bool)
        self.assertIsInstance(data.params["time_series"], bool)

    def test_call(self):
        data = Data("iris")
        data()
        self.assertIsInstance(data.X_train, np.ndarray)
        self.assertIsInstance(data.y_train, np.ndarray)
        self.assertIsInstance(data.X_test, np.ndarray)
        self.assertIsInstance(data.y_test, np.ndarray)
        self.assertEqual(data.X_train.shape[0], data.y_train.shape[0])

    def test_set_params(self):
        data = Data("mnist", train_size=100, random_state=0, shuffle=True)
        self.assertEqual(data.params["dataset"], "mnist")
        data.set_params(dataset="iris")
        self.assertEqual(data.dataset, "iris")
        self.assertRaises(TypeError, data.set_params, asw3="sas09d8fap0s98jf;a")
        self.assertRaises(ValueError, data.set_params, dataset=1)

    def test_sample_data(self):
        data = Data("mnist", random_state=220)
        data2 = Data("mnist", random_state=20)
        self.assertNotEqual(data, data2)

    def test_parse_data(self):
        data = "https://raw.githubusercontent.com/simplymathematics/datasets/master/titanic.csv"
        data = Data(data, target="Survived", shuffle=True, stratify=None)
        self.assertIsInstance(data, Data)
        data2 = Data(data, target="Ticket", shuffle=True, stratify=None)
        self.assertIsInstance(data2, Data)
        self.assertNotEqual(data, data2)

    def test_save_data(self):
        data = Data("iris")
        data.save(filename=self.filename, path=self.path)
        self.assertTrue(os.path.exists(os.path.join(self.path, self.filename)))

    def test_rpr_(self):
        data = Data("iris")
        with open(os.path.join(self.path, "test.json"), "w") as f:
            json.dump(str(data), f)
        data2 = Data(os.path.join(self.path, "test.json"))
        data = dict(data.params)
        data2 = dict(data2.params)
        # TODO: fix this
        del data["dataset"]
        del data2["dataset"]
        self.assertDictEqual(data, data2)

    def test_str(self):
        data = Data("iris", train_size=0.8)
        data()
        params = dict(data.params)
        data2 = Data(**params)
        data2()
        self.assertEqual(data, data2)
        self.assertDictEqual(dict(data.params), dict(data2.params))

    def tearDown(self):
        shutil.rmtree(self.path)
