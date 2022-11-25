import os
import shutil
import tempfile
import unittest
from copy import deepcopy

import numpy as np
from deckard.base import Data, Model
from deckard.base.hashable import my_hash
from pathlib import Path
import yaml


yaml.add_constructor("!Data:", Data)
yaml.add_constructor("!Model:", Model)
class testModel(unittest.TestCase):
    def setUp(self):
        self.path ="model"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.model1 = """
        !Model:
            init:
                loss: "log_loss"
                name: sklearn.linear_model.SGDClassifier
            files:
                model_path : model
                model_filetype : pickle
            fit:
                epochs: 1000
                learning_rate: 1.0e-08
                log_interval: 10
            """
        self.model2 = """
        !Model:
            init:
                loss: "hinge"
                name: sklearn.linear_model.SGDClassifier
            files:
                model_path : model
                model_filetype : pickle
            fit:
                epochs: 1000
                learning_rate: 1.0e-08
                log_interval: 10
            """
        self.data1 = """
        !Data:
            sample:
                shuffle : True
                random_state : 42
                train_size : 800
                stratify : True
            add_noise:
                train_noise : 1
                time_series : True
            name: classification
            files:
                data_path : data
                data_filetype : pickle
            generate:
                n_samples: 1000
                n_features: 2
                n_informative: 2
                n_redundant : 0
                n_classes: 2
            sklearn_pipeline:
                - sklearn.preprocessing.StandardScaler
            transform:
                sklearn.preprocessing.StandardScaler:
                    with_mean : true
                    with_std : true
                    X_train : true
                    X_test : true
        """
        self.url = "https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1"
        
        self.loaded_model1 = yaml.load(self.model1, Loader=yaml.FullLoader)
        self.loaded_model2 = yaml.load(self.model2, Loader=yaml.FullLoader)
        self.loaded_model3 = yaml.load(self.model1, Loader=yaml.FullLoader)

    def test_model(self):
        doc = self.model1
        model1 = yaml.load(doc, Loader=yaml.FullLoader)
        assert model1._asdict() == self.loaded_model1._asdict()

    def test_hash(self):
        model1 = self.loaded_model1
        model2 = yaml.load(self.model1, Loader=yaml.FullLoader)
        model3 = self.loaded_model2
        model1 = model1._asdict()
        model2 = model2._asdict()
        model3 = model3._asdict()
        self.assertEqual(my_hash(model1), my_hash(model2))
        self.assertNotEqual(my_hash(model1), my_hash(model3))

    def test_eq(self):
        model1 = self.loaded_model1
        model2 = yaml.load(self.model1, Loader=yaml.FullLoader)
        model3 = self.loaded_model2
        self.assertEqual(model1, model2)
        self.assertNotEqual(model1, model3)


    def test_save_model(self):
        model1 = self.loaded_model1
        new = model1.load()
        filename = model1.save(new)
        self.assertTrue(Path(self.path, my_hash(model1._asdict()) + ".pickle").exists())

    def test_load(self):
        self.assertEqual(self.loaded_model1, self.loaded_model3)

    def test_load_with_defence(self):
        model1 = """
        !Model:
            init:
                loss: "log_loss"
                name: sklearn.linear_model.SGDClassifier
            files:
                model_path : model
                model_filetype : pickle
            fit:
                epochs: 1000
                learning_rate: 1.0e-08
                log_interval: 10
            art_pipeline:
                preprocessor_defence:
                    name: art.defences.preprocessor.FeatureSqueezing
                    params:
                        bit_depth: 4
        """
        model1 = yaml.load(model1, Loader=yaml.FullLoader)
        self.assertIn("preprocessor_defence", model1.art_pipeline)

    def tearDown(self):
        shutil.rmtree(self.path)
