import unittest
import warnings
import yaml
from pathlib import Path

import numpy as np
from deckard.base import Attack, Experiment
from deckard.base.experiment import config
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelBinarizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class testAttackExperiment(unittest.TestCase):
    def setUp(self):
        yaml.add_constructor("!Attack:", Attack)
        atk_config = "!Attack:" + config
        self.attack = yaml.load(atk_config, Loader=yaml.FullLoader)
        yaml.add_constructor("!Experiment:", Experiment)
        exp_config = "!Experiment:" + config
        self.exp = yaml.load(exp_config, Loader=yaml.FullLoader)
        self.path = "reports"
        data, model, _ = self.exp.load()
        self.data = data.load("reports/filename.pickle")
        self.model = model.load("reports/filename.pickle")
        self.data.y_train = LabelBinarizer().fit_transform(self.data.y_train)
        self.data.y_test = LabelBinarizer().fit_transform(self.data.y_test)
        self.model.fit(self.data.X_train, self.data.y_train)

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
        for name, filename in outs.items():
            self.assertTrue(Path(filename).exists())

    def tearDown(self) -> None:
        from shutil import rmtree

        if Path(self.path).is_dir():
            rmtree(self.path)
        if Path("model").is_dir():
            rmtree("model")
        if Path("data").is_dir():
            rmtree("data")
        del self.path
