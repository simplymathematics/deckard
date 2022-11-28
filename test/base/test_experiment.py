import unittest
import warnings
import yaml
from collections.abc import Callable  # noqa F401
from copy import deepcopy
from argparse import Namespace
from pathlib import Path

import numpy as np
from art.attacks.evasion import BoundaryAttack  # noqa F401
from art.defences.postprocessor import HighConfidence  # noqa F401
from art.defences.preprocessor import FeatureSqueezing  # noqa F401
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # noqa F401

from sklearn.cluster import KMeans  # noqa F401
from sklearn.exceptions import UndefinedMetricWarning
from deckard.base import Data, Experiment, Model, Yellowbrick_Visualiser
from deckard.base.hashable import my_hash
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

yaml.add_constructor("!Data:", Data)
yaml.add_constructor("!Model:", Model)
yaml.add_constructor("!Experiment:", Experiment)
yaml.add_constructor("!Yellowbrick_Visualiser:", Yellowbrick_Visualiser)
class testExperiment(unittest.TestCase):
    def setUp(self):

        self.path = "reports"
        self.config = """
        !Experiment:
            model:
                init:
                    loss: "hinge"
                    name: sklearn.linear_model.SGDClassifier
                files:
                    model_path : reports
                    model_filetype : pickle
                # fit:
                #     epochs: 1000
                #     learning_rate: 1.0e-08
                #     log_interval: 10
            data:
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
                    data_path : reports
                    data_filetype : pickle
                generate:
                    n_samples: 1000
                    n_features: 2
                    n_informative: 2
                    n_redundant : 0
                    n_classes: 3
                    n_clusters_per_class: 1
                sklearn_pipeline:
                    sklearn.preprocessing.StandardScaler:
                            with_mean : true
                            with_std : true
                            X_train : true
                            X_test : true
            attack:
                init:
                    name: art.attacks.evasion.HopSkipJump
                    max_iter : 10
                    init_eval : 10
                    init_size : 10
                files:
                    adv_samples: adv_samples.json
                    adv_predictions : adv_predictions.json
                    adv_time_dict : adv_time_dict.json
                    attack_params : attack_params.json
            plots:
                balance: balance
                classification: classification
                confusion: confusion
                correlation: correlation
                radviz: radviz
                rank: rank
            scorers:
                accuracy:
                    name: sklearn.metrics.accuracy_score
                    normalize: true
                f1-macro:
                    average: macro
                    name: sklearn.metrics.f1_score
                f1-micro:
                    average: micro
                    name: sklearn.metrics.f1_score
                f1-weighted:
                    average: weighted
                    name: sklearn.metrics.f1_score
                precision:
                    average: weighted
                    name: sklearn.metrics.precision_score
                recall:
                    average: weighted
                    name: sklearn.metrics.recall_score
            files:
                ground_truth_file: ground_truth.json
                predictions_file: predictions.json
                time_dict_file: time_dict.json
                params_file: params.json
                score_dict_file: scores.json
                path: reports
            """
        self.file = "test_filename"
        self.here = Path(__file__).parent
        self.exp = yaml.load(self.config, Loader=yaml.FullLoader)

    def test_experiment(self):
        self.assertIsInstance(self.exp, Experiment)

    def test_hash(self):
        exp = yaml.load(self.config, Loader=yaml.FullLoader)
        self.assertEqual(my_hash(exp._asdict()), my_hash(self.exp._asdict()))
    

    def test_run(self):
        files = self.exp.files
        files = Namespace(**files)
        self.exp.run()
        path = Path(self.path)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
    

    # def test_save_data(self):
    #     data, _, _ , _, _ = self.exp.load()
    #     data = dict(vars(data.load()))
    #     path = self.exp.save_data(data)
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())
        
    # def test_save_params(self):
    #     _, _, _ , _, _ = self.exp.load()
    #     path = self.exp.save_params()
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())

    # def test_save_model(self):
    #     _, model, _ , _, _ = self.exp.load()
    #     model = model.load()
    #     path = self.exp.save_model(model)
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())

    # def test_save_predictions(self):
    #     data, model, _ , _, _ = self.exp.load()
    #     model = model.load()
    #     data = data.load()
    #     model.fit(data.X_train, data.y_train)
    #     predictions = model.predict(data.X_test)
    #     path = self.exp.save_predictions(predictions)
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())
    
    # def test_save_ground_truth(self):
    #     data, model, _ , _, _ = self.exp.load()
    #     model = model.load()
    #     data = data.load()
    #     model.fit(data.X_train, data.y_train)
    #     truth = model.predict(data.X_test)
    #     path = self.exp.save_ground_truth(truth)
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())
    
    # def test_save_time_dict(self):
    #     data, model, _ , _, _ = self.exp.load()
    #     time_dict = {"fit_time": 0, "pred_time": 0}
    #     path = self.exp.save_time_dict(time_dict) 
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())

    # def test_score(self):
    #     data, model, _ , _, _ = self.exp.load()
    #     model = model.load()
    #     data = data.load()
    #     model.fit(data.X_train, data.y_train)
    #     predictions = model.predict(data.X_test)
    #     score_dict = self.exp.score(predictions = predictions, ground_truth = data.y_test)
    #     path = self.exp.save_scores(score_dict)
    #     path = Path(self.path)
    #     self.assertTrue(path.exists())
        
    
    def tearDown(self) -> None:
        from shutil import rmtree
        if Path(self.here, "data").exists():
            rmtree(Path(self.here, "data"))
        if Path(self.here, "models").exists():
            rmtree(Path(self.here, "models"))
        if Path(self.here, self.path).exists():
            rmtree(Path(self.here, self.path))
        del self.path
        del self.file


if __name__ == "__main__":
    unittest.main()
