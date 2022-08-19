from importlib.resources import Resource
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
import unittest
import os, json
import tempfile
from deckard.base import Data, Experiment, Model
from deckard.base.utils import SUPPORTED_MODELS, return_score,  save_best_only, save_all, loggerCall, find_successes, remove_successes_from_queue
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from art.estimators.classification import PyTorchClassifier, KerasClassifier, TensorFlowClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnRandomForestClassifier
from logging import Logger
logger = loggerCall()
class testUtils(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_model'
        self.data = Data('iris')
        self.model = Model(RandomForestClassifier(), 'sklearn')
        self.model2 = Model(DecisionTreeClassifier(), 'sklearn')
        self.model3 = Model(SVC(), 'sklearn')

        self.experiment = Experiment(self.data, self.model)
        self.experiment2 = Experiment(self.data, self.model2)
        self.experiment3 = Experiment(self.data, self.model3)
        self.experiment.run(self.path)
        self.experiment.save_results(path = self.path)
        self.experiment.save_model(filename = 'model', path = self.path)
        self.experiment.save_data(filename = 'data.pkl', path = self.path)
        self.list = [self.experiment, self.experiment2]
    
    def test_return_score(self):
        result = return_score(scorer = 'acc', filename="scores.json", path=self.path)
        self.assertIsInstance(result, float)

    def test_save_best_only(self):
        save_best_only(path = self.path, scorer = 'ACC', exp_list=self.list, bigger_is_better=True)
        files = [x for x in os.listdir(self.path)]
        self.assertIn('model.pickle', files)
        self.assertIn('scores.json', files)
        self.assertIn('predictions.json', files)
        self.assertIn('data_params.json', files)
        self.assertIn('model_params.json', files)

    def test_save_all(self):
        old = [x[0] for x in os.walk(self.path)]
        save_all(path = self.path, scorer = 'ACC', exp_list=self.list, bigger_is_better=True, name = 'test')
        files = [x for x in os.listdir(self.path)]
        folders = [x[0] for x in os.walk(self.path)]
        self.assertTrue(len(folders)> len(old)) #+1 for the root folder
        self.assertIn('model.pickle', files)
        self.assertIn('scores.json', files)
        self.assertIn('predictions.json', files)
        self.assertIn('data_params.json', files)
        self.assertIn('model_params.json', files)
    
    def test_loggerCall(self):
        logger = loggerCall()
        self.assertIsInstance(logger, Logger)
    
    def test_find_successes(self):
        self.experiment = Experiment(self.data, self.model)
        self.experiment.run(self.path)
        self.experiment.save_params(path = self.path)
        self.experiment.save_results(path = self.path)
        self.experiment.save_model(filename = 'model.pickle', path = self.path)
        successes, failures = find_successes(self.path, 'model_params.json')
        self.assertIsInstance(successes, list)
        self.assertEqual(len(failures), 0)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.path)
        
