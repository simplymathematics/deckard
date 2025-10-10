import unittest
import tempfile
import os
import pickle
import pandas as pd
import numpy as np
import warnings
import sys

from deckard.data import DataConfig
from deckard.model import ModelConfig

from deckard.attack import AttackConfig, initialize_attack_config, attack_parser, supported_attacks


warnings.filterwarnings("ignore", category=UserWarning)
class DummyEstimator:
    def predict(self, X):
        n = len(X)
        return np.eye(2)[np.zeros(n, dtype=int)]

class DummyAttack:
    def fit(self, x, y=None, test_x=None, test_y=None):
        pass
    def infer(self, x, y, pred=None):
        # return something in the shape of y
        return np.zeros_like(y)
    
class DummyDataConfig:
    def __init__(self):
        self._X_train = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'sex'])
        self._y_train = pd.Series(np.random.randint(0, 2, 10))
        self._X_test = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'sex'])
        self._y_test = pd.Series(np.random.randint(0, 2, 10))

class TestAttackConfig(unittest.TestCase):
    def setUp(self):
        self.attack_size = 5
        self.config = AttackConfig(attack_size=self.attack_size, attack_name="art.attacks.evasion.HopSkipJump")
        self.data = DummyDataConfig()
        self.estimator = DummyEstimator()
        self.attack = DummyAttack()

    def test_pop_attribute(self):
        X = self.data._X_train.copy()
        X_np, target = self.config._pop_attribute(X, 'sex')
        self.assertEqual(X_np.shape[1], 2)
        self.assertEqual(len(target), len(X_np))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "attack_test.pkl")
            self.config._save(filepath)
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, "rb") as f:
                loaded = pickle.load(f)
            self.assertIsInstance(loaded, AttackConfig)

    def test_not_implemented_poison(self):
        with self.assertRaises(NotImplementedError):
            self.config._poison()
            
    def test_attack_size(self):
        data = DataConfig(
            dataset_name="digits",
            data_params={},
            test_size=0.1,
            random_state=123,
            stratify=True
        )
        data()
        model = ModelConfig(model_type="sklearn.linear_model.LogisticRegression", model_params={"max_iter": 100})
        model._train(data._X_train, data._y_train)
        model = model._model
        attack = AttackConfig(attack_name="art.attacks.evasion.HopSkipJump", attack_size=self.attack_size)
        attack(data, model)
        
        self.assertEqual(attack.attack_size, len(attack._attack))
    
    def test_not_implemented_extract(self):
        with self.assertRaises(NotImplementedError):
            self.config._extract()

    def test_initialize_attack_config_default(self):
        sys.argv = ['test']
        attack = initialize_attack_config()
        self.assertIsInstance(attack, AttackConfig)
        self.assertEqual(attack.attack_name, "art.attacks.evasion.HopSkipJump")

    def test_initialize_attack_config_predefined_cases(self):
        for case in supported_attacks:
            sys.argv = ['test', '--attack_config_file', case]
            attack = initialize_attack_config()
            self.assertIsInstance(attack, AttackConfig)
            self.assertIn("attack_name", attack.__dict__)
            self.assertTrue(attack.attack_name.startswith("art.attacks"))

    def test_initialize_attack_config_with_attack_params(self):
        sys.argv = ['test', '--attack_params', '++attack_params.init_eval=3', '++attack_name=art.attacks.evasion.HopSkipJump',]
        attack = initialize_attack_config()
        self.assertIsInstance(attack, AttackConfig)
        self.assertEqual(attack.attack_params.get("init_eval"), 3)

if __name__ == "__main__":
    unittest.main()