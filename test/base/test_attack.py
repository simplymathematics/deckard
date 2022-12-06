import pickle
import unittest
import warnings
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelBinarizer
from art.utils import load_dataset
from deckard.base import Attack, Experiment
from deckard.base.experiment import config as default_config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class testAttackExperiment(unittest.TestCase):
    def setUp(self, default_config = default_config):
        yaml.add_constructor("!Attack:", Attack)
        atk_config = "!Attack:" + default_config
        self.attack = yaml.load(atk_config, Loader=yaml.FullLoader)
        yaml.add_constructor("!Experiment:", Experiment)
        exp_config = "!Experiment:" + default_config
        self.exp = yaml.load(exp_config, Loader=yaml.FullLoader)
        self.path = "reports"
        data, model, _ = self.exp.load()
        self.data = data.load("reports/filename.pickle")
        self.model = model.load("reports/filename.pickle", art = True)
        self.data.y_train = LabelBinarizer().fit_transform(self.data.y_train)
        self.data.y_test = LabelBinarizer().fit_transform(self.data.y_test)
        self.model.fit(self.data.X_train, self.data.y_train)
        self.url = "https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1"
        

    def test_init(self):
        self.assertTrue(isinstance(self.attack, Attack))

    def test_load(self):
        atk, gen = self.attack.load(self.model)
        self.assertTrue(hasattr(atk, "generate"))
        self.assertIsInstance(gen, dict)

    def test_generate(self):
        atk, gen = self.attack.load(self.model)
        (
            adv_samples,
            attack_pred,
            time_dict,
        ) = self.attack.fit(self.data, self.model, atk, **gen)
        self.assertIsInstance(adv_samples, np.ndarray)
        self.assertIsInstance(time_dict, dict)
        self.assertIsInstance(attack_pred, np.ndarray)

    def test_save_attack_predictions(self):
        preds = self.data.y_test
        path = self.attack.save_attack_predictions(preds)
        path = Path(path)
        self.assertTrue(path.exists())

    def test_save_attack_params(self):
        path = self.attack.save_attack_params()
        path = Path(path)
        self.assertTrue(path.exists())

    def test_save_attack_time(self):
        time = {"time": 1}
        path = self.attack.save_attack_time(time)
        path = Path(path)
        self.assertTrue(path.exists())

    def test_run_attack(self):
        outs = self.attack.run_attack(self.data, self.model, self.attack)
        for _, filename in outs.items():
            self.assertTrue(Path(filename).exists())
    
    def test_whitebox_on_tf1(self):
        # disable eager execution
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        whitebox = {"attack" : {"init" : {"name" : "art.attacks.evasion.ZooAttack", "confidence" : 0.3, "max_iter" : 1}}}
        config = deepcopy(self.exp._asdict())
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test), min_, max_ = load_dataset('cifar10')
        data = Namespace(X_train = X_train[:10], y_train = y_train[:10], X_test = X_test[:1], y_test = y_test[:1])
        Path("reports").mkdir(parents=True, exist_ok=True)
        with open("reports/data.pickle", "wb") as f:
            pickle.dump(data, f)
        config['attack'] = whitebox['attack']
        tf1 = {"init" : {"name" : "art_models/model.pb", "library":"keras"}, "url" : "https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1"}
        config['attack'] = whitebox['attack']
        config['model'] = tf1        
        config['data'] = {"name" : "reports/data.pickle"}
        del config['plots']
        white_conf = "!Attack:\n" + str(yaml.dump(config))
        exp_conf = "!Experiment:\n" + str(yaml.dump(config))
        whitebox = yaml.load(white_conf, Loader=yaml.FullLoader)
        exp = yaml.load(exp_conf, Loader=yaml.FullLoader)
        outs = exp.run(art = True)
        self.assertIsInstance(outs, dict)

    def tearDown(self) -> None:
        from shutil import rmtree

        if Path(self.path).is_dir():
            rmtree(self.path)
        if Path("model").is_dir():
            rmtree("model")
        if Path("data").is_dir():
            rmtree("data")
        if Path("reports").is_dir():
            rmtree("reports")
        del self.path
