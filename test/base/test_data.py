import os
import shutil
import tempfile
import unittest
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path
from deckard.base import Data
from deckard.base.hashable import my_hash

names = ["mnist", "cifar10", "iris"]
# TODO other names


class testData(unittest.TestCase):
    def setUp(self):
        self.filename = "test_data.pkl"
        self.path = "tmp"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        yaml.add_constructor("!Data:", Data)
        self.data_document = """
        !Data:
            name: classification
            sample:
                shuffle : True
                random_state : 42
                train_size : 800
                stratify : True
                time_series : True
            add_noise:
                train_noise : 1
            files:
                data_path : tmp
                data_filetype : pickle
            generate:
                n_samples: 1000
                n_features: 2
                n_informative: 2
                n_redundant : 0
                n_classes: 3
                n_clusters_per_class: 1
            sklearn_pipeline:
                steps:
                    sklearn.preprocessing.StandardScaler:
                        with_mean : true
                        with_std : true
                        X_train : true
                        X_test : true
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
        params = data._asdict()['sample']
        self.assertIsInstance(params["train_size"], (float, int))
        self.assertIsInstance(params["random_state"], int)
        self.assertIsInstance(params["shuffle"], bool)

    def test_load(self):
        data = self.data
        data = data.load()
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

    def test_save_data(self):
        data = self.data
        data.save(data=self.data.load())
        self.assertTrue(os.path.exists(os.path.join(self.path, my_hash(data._asdict()) + "." + "pickle")))

    def tearDown(self):
        shutil.rmtree(self.path)
