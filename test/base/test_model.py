import shutil
import unittest
from pathlib import Path

import yaml

from deckard.base.data import Data
from deckard.base.hashable import my_hash
from deckard.base.model import Model

yaml.add_constructor("!Data:", Data)
yaml.add_constructor("!Model:", Model)


class testModel(unittest.TestCase):
    def setUp(self):
        self.path = "model"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.model1 = """
        !Model:
            init:
                loss: "log_loss"
                name: sklearn.linear_model.SGDClassifier
            fit:
                epochs: 1000
                learning_rate: 1.0e-08
                log_interval: 10
            sklearn_pipeline:
                feature_selection:
                    name: sklearn.preprocessing.StandardScaler
                    with_mean : true
                    with_std : true
            """
        self.model2 = """
        !Model:
            init:
                loss: "hinge"
                name: sklearn.linear_model.SGDClassifier
            fit:
                epochs: 1000
                learning_rate: 1.0e-08
                log_interval: 10
            sklearn_pipeline:
                feature_selection:
                    name: sklearn.preprocessing.StandardScaler
                    with_mean : true
                    with_std : true
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
            name: classification
            generate:
                n_samples: 1000
                n_features: 2
                n_informative: 2
                n_redundant : 0
                n_classes: 2
        """
        model_document = """
        !Model:
            init:
                name : "reports/model.pb"
            url : https://www.dropbox.com/s/hbvua7ynhvara12/cifar-10_ratio%3D0.h5?dl=1
            art_pipeline:
                preprocessor_defence : {name: art.defences.preprocessor.FeatureSqueezing, params: {bit_depth: 4, clip_values: [0, 1]}}
                postprocessor_defence : {name: art.defences.postprocessor.HighConfidence, params: {cutoff: 0.9}}
                transformer_defence : {name: art.defences.transformer.evasion.DefensiveDistillation, params: {batch_size: 128}}
                trainer_defence : {name: art.defences.trainer.AdversarialTrainerMadryPGD, params: {nb_epochs: 10}}
        """
        self.tf1 = yaml.load(model_document, Loader=yaml.Loader)
        self.url = "https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1"

        self.loaded_model1 = yaml.load(self.model1, Loader=yaml.FullLoader)
        self.loaded_model2 = yaml.load(self.model2, Loader=yaml.FullLoader)
        self.loaded_model3 = yaml.load(self.model1, Loader=yaml.FullLoader)
        self.filename = Path(self.path, "model.pickle")

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

    def test_tf_model(self):
        self.assertIsInstance(self.tf1, Model)

    def tearDown(self):
        shutil.rmtree(self.path)
