import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import tempfile
import unittest
from deckard.base import Data, Model, AttackExperiment
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from os import path, listdir


class testAttackExperiment(unittest.TestCase):
    def setUp(self):

        self.path = tempfile.mkdtemp()
        ART_DATA_PATH = self.path
        self.file = "test_filename"
        self.here = path.dirname(path.abspath(__file__))

    def test_save_attack_predictions(self):
        data = Data("iris", test_size=30)
        data()
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type="sklearn", path=self.path, art=True)
        model(art = True).fit(data.X_train, data.y_train)
        attack = {
            "name" : "art.attacks.evasion.BoundaryAttack",
            "params" : {
                "max_iter" : 10,
                "targeted" : False,
            }
        }
        experiment = AttackExperiment(data=data, model=model, attack=attack)
        experiment(path=self.path)
        file = experiment.save_attack_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(file))

    def test_save_attack_params(self):
        data = Data("iris", test_size=30)
        data()
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type="sklearn", path=self.path)
        model(art = True).fit(data.X_train, data.y_train)
        attack = {
            "name" : "art.attacks.evasion.BoundaryAttack",
            "params" : {
                "max_iter" : 10,
                "targeted" : False,
            }
        }
        experiment = AttackExperiment(data=data, model=model, attack=attack)
        files = experiment.save_params(path=self.path)
        for file in files:
            self.assertTrue(path.exists(path.join(self.path, file)))

    def test_attack_params(self):
        data = Data("iris", test_size=30)
        data()
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type="sklearn", path=self.path)
        model(art= True).fit(data.X_train, data.y_train)
        attack = {
            "name" : "art.attacks.evasion.BoundaryAttack",
            "params" : {
                "max_iter" : 10,
                "targeted" : False,
            }
        }
        experiment = AttackExperiment(data=data, model=model, attack=attack)
        

    def test_run_attack(self):
        data = Data("iris", test_size=30)
        data()
        model = DecisionTreeClassifier()
        estimator = ScikitlearnClassifier(DecisionTreeClassifier())
        model = Model(estimator, model_type="sklearn", path=self.path)
        estimator.fit(data.X_train, data.y_train)
        attack = {
            "name" : "art.attacks.evasion.BoundaryAttack",
            "params" : {
                "max_iter" : 10,
                "targeted" : False,
            }
        }
        experiment = AttackExperiment(data=data, model=model, attack=attack)
        experiment(path=self.path)
        self.assertIsInstance(experiment.attack, BoundaryAttack)
        self.assertIsInstance(experiment.adv, (list, np.ndarray))
        self.assertTrue("adv_fit_time" in str(experiment.time_dict))
        self.assertTrue("adv_pred_time" in str(experiment.time_dict))

    def test_set_attack(self):
        data = Data("iris", test_size=30)
        data()
        model = DecisionTreeClassifier()
        estimator = ScikitlearnClassifier(DecisionTreeClassifier())
        estimator.fit(data.X_train, data.y_train)
        model = Model(estimator, model_type="sklearn", path=self.path)
        attack = {
            "name" : "art.attacks.evasion.BoundaryAttack",
            "params" : {
                "max_iter" : 10,
                "targeted" : False,
            }
        }
        experiment = AttackExperiment(data=data, model=model, attack=attack)
        self.assertIsInstance(experiment.attack, object)
        self.assertIn("Attack", experiment.params)
        key = list(experiment.params["Attack"])[0]
        self.assertEqual(attack["name"], experiment.params["Attack"][key]['name'])
        self.assertEqual(attack["params"], experiment.params["Attack"][key]['params'])

    def test_run_attack_files(self):
        data = Data("iris", test_size=30)
        data()
        model = DecisionTreeClassifier()
        estimator = ScikitlearnClassifier(DecisionTreeClassifier())
        model = Model(estimator, model_type="sklearn", path=self.path)
        estimator.fit(data.X_train, data.y_train)
        attack = {
            "name" : "art.attacks.evasion.BoundaryAttack",
            "params" : {
                "max_iter" : 10,
                "targeted" : False,
            }
        }
        experiment = AttackExperiment(data=data, model=model, attack=attack)
        
        
        files = experiment(path=self.path)
        for file_ in files:
            bool_ = path.exists(path.join(self.path, file_))
            # Hacky work-around for ART weirdness. Patch submitted to ART.
            bool_2 = path.exists(path.join(self.path, file_.split(".")[0] + ".pickle"))
            bool_ = bool_ or bool_2
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
        del self.path
        del self.file
