import tempfile
import unittest
import warnings
import yaml
from pathlib import Path

import numpy as np
from deckard.base import Attack
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class testAttackExperiment(unittest.TestCase):
    def setUp(self):

        self.path = "reports"
        self.filename = Path(self.path, "tmp.json")
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
        self.data, model, self.attack, _, _ = self.exp.load()
        self.model = model.load(art = True)
        self.data = self.data.load()
        self.data.y_train = LabelBinarizer().fit_transform(self.data.y_train)
        self.model.fit(self.data.X_train, self.data.y_train)

    def test_save_attack_predictions(self):
        preds = self.data.y_test
        path = self.attack.save_attack_predictions(preds, self.filename)
        self.assertTrue(path.exists())

    def test_save_attack_params(self):
        params = self.attack._asdict()
        path = self.attack.save_attack_params(self.filename)
        self.assertTrue(path.exists())

    def test_save_attack_time(self):
        time = {"time": 1}
        path = self.attack.save_attack_time(time, self.filename)
        self.assertTrue(path.exists())

    def test_load(self):
        self.assertTrue(isinstance(self.attack, Attack))
    
    def test_run_attack(self):
        preds, samples, time_dict = self.attack.run_attack(self.data, self.model, self.attack)
        self.assertIsInstance(preds, np.ndarray)
        self.assertIsInstance(samples, np.ndarray)
        self.assertIsInstance(time_dict, dict)

    def tearDown(self) -> None:
        from shutil import rmtree
        if Path(self.path).exists():
            rmtree(self.path)
        if Path("model").exists():
            rmtree("model")
        if Path("data").exists():
            rmtree("data")
        del self.path
        del self.file
