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
        self.file = 'test_filename'
        self.here = path.dirname(path.abspath(__file__))

    def test_save_attack_predictions(self):
        data = Data('iris', test_size = 30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment = AttackExperiment(data = data, model = model, attack = attack)
        experiment._build_model()
        experiment(path = self.path)
        experiment(path = self.path)
        experiment.save_attack_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_attack(self):
        data = Data('iris', test_size = 30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment = AttackExperiment(data = data, model = model, attack = attack)
        experiment.save_attack_params(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_get_attack(self):
        data = Data('iris', test_size = 30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment = AttackExperiment(data = data, model = model, attack = attack)
        self.assertIsInstance(experiment.get_attack(), object)
        self.assertEqual(experiment.get_attack(), attack)  
        
    def test_run_attack(self):
        data = Data('iris', test_size = 30)
        model =  DecisionTreeClassifier()
        estimator = ScikitlearnClassifier( DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        estimator.fit(data.X_train, data.y_train)
        attack = BoundaryAttack(estimator, targeted = False, max_iter = 10, verbose = False)
        experiment = AttackExperiment(data = data, model = model, attack = attack)
        experiment.set_attack(attack)
        experiment(path = self.path)
        self.assertIsInstance(experiment.attack, BoundaryAttack)
        self.assertIsInstance(experiment.adv, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict['adv_fit_time'], (int, float))
        self.assertIsInstance(experiment.time_dict['adv_pred_time'], (int, float))
    
    def test_set_attack(self):
        data = Data('iris', test_size = 30)
        model =  DecisionTreeClassifier()
        estimator = ScikitlearnClassifier( DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        estimator.fit(data.X_train, data.y_train)
        attack = BoundaryAttack(estimator, targeted = False, max_iter = 10, verbose = False)
        experiment = AttackExperiment(data = data, model = model, attack = attack)
        old = hash(experiment)
        old_name = experiment.filename
        experiment.set_attack(attack)
        experiment(path = self.path)
        new = hash(experiment)
        new_name = experiment.filename
        self.assertIsInstance(experiment.attack, object)
        self.assertEqual(experiment.attack, attack)
        self.assertIn('Attack', experiment.params)
        self.assertIn('name', experiment.params['Attack'])
        self.assertIn('params', experiment.params['Attack'])
        self.assertNotEqual(old, new)
        self.assertNotEqual(old_name, new_name)
    
    def test_run_attack_files(self):
        data = Data('iris', test_size = 30)
        model =  DecisionTreeClassifier()
        estimator = ScikitlearnClassifier( DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        estimator.fit(data.X_train, data.y_train)
        attack = BoundaryAttack(estimator, targeted = False, max_iter = 10, verbose = False)
        experiment = AttackExperiment(data = data, model = model, attack = attack)
        experiment(path = self.path)
        these = ['attack_params.json', 'attack_examples.json', 'attack_time_dict.json']
        for file in these:
            bool_ = path.exists(path.join(self.path, file))
            self.assertTrue(bool_)
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
        del self.path
        del self.file