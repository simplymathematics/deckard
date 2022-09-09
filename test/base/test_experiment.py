import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import tempfile
import unittest
from deckard.base import Data, Model, Experiment
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import  KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from copy import deepcopy
from collections.abc import Callable
from art.attacks.evasion import BoundaryAttack
from art.defences.preprocessor import FeatureSqueezing
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.defences.postprocessor import HighConfidence
from os import path, listdir
class testExperiment(unittest.TestCase):
    def setUp(self):
        
        self.path = tempfile.mkdtemp()
        ART_DATA_PATH = self.path
        self.file = 'test_filename'
        self.here = path.dirname(path.abspath(__file__))

    def test_experiment(self):
        data = Data('iris', test_size = 30)
        self.assertIsInstance(data, Data)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        self.assertIsInstance(model, Model)
        experiment = Experiment(data = data, model = model)
        self.assertIsInstance(experiment, Experiment)
        self.assertIsInstance(experiment.data, Data)
        self.assertIsInstance(experiment.model, Model)
        self.assertIsInstance(experiment.params, dict)
        self.assertEqual(experiment.predictions, None)
        self.assertEqual(experiment.time_dict, None)
        self.assertEqual(experiment.model_type, 'sklearn')

    def test_hash(self):
        defence = FeatureSqueezing(bit_depth = 1, clip_values = (0, 1))
        defence2 = FeatureSqueezing(bit_depth = 1, clip_values = (0, 1))
        data = Data('iris', test_size = 30)
        model = Model(KNeighborsRegressor(5), model_type = 'sklearn', path = self.path)
        model2 = Model(KNeighborsRegressor(4), model_type = 'sklearn', path = self.path)
        model3 = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        model4 = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment2 = Experiment(data = data, model = model2)
        experiment3 = Experiment(data = data, model = model3)
        experiment4 = Experiment(data = data, model = model4)
        experiment5 = deepcopy(experiment)
        before = hash(experiment)
        self.assertEqual(hash(experiment), hash(experiment3))
        self.assertEqual(hash(experiment), hash(experiment5))
        self.assertNotEqual(hash(experiment), hash(experiment4))
        self.assertNotEqual(hash(experiment), hash(experiment2))
        experimentd1 = deepcopy(experiment)
        experimentd1.set_defence(defence)
        self.assertNotEqual(before, hash(experimentd1))
        experimentd2 = deepcopy(experiment)
        experimentd2.set_defence(defence2)
        self.assertNotEqual(hash(experiment), hash(experimentd2))
        self.assertEqual(hash(experimentd1), hash(experimentd2))

    def test_eq(self):
        data = Data('iris', test_size = 30)
        model = Model(KNeighborsRegressor(5), model_type = 'sklearn', path = self.path)
        model2 = Model(KNeighborsRegressor(4), model_type = 'sklearn', path = self.path)
        model3 = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        model4 = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment2 = Experiment(data = data, model = model2)
        experiment3 = Experiment(data = data, model = model3)
        experiment4 = Experiment(data = data, model = model4)
        experiment5 = deepcopy(experiment)
        self.assertEqual(experiment.data, experiment3.data)
        self.assertEqual(experiment.model, experiment3.model)
        self.assertEqual(experiment, experiment3)
        self.assertEqual(experiment, experiment5)
        self.assertNotEqual(experiment, experiment4)
        self.assertNotEqual(experiment, experiment2)

    def test_run(self):
        data = Data('iris', test_size = 30)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn('fit', experiment.time_dict)
        self.assertIn('predict', experiment.time_dict)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn('fit', experiment.time_dict)
        self.assertIn('predict', experiment.time_dict)

    def test_set_filename(self):
        data = Data('iris', test_size = 30)
        model = Model(KMeans(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.set_filename('test_filename')
        self.assertEqual(experiment.filename, 'test_filename')

    def test_save_cv_scores(self):
        from sklearn.model_selection import GridSearchCV
        data = Data('iris', test_size = 30)
        estimator = DecisionTreeClassifier()
        grid = GridSearchCV(estimator, {'max_depth': [1, 2, 3, 4, 5]}, cv=3, return_train_score=True)
        model = Model(grid, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        experiment.save_cv_scores(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_data(self):
        data = Data('iris', test_size = 30)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.save_data(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_params(self):
        data = Data('iris', test_size = 30)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.save_params(path=self.path)
        self.assertTrue(path.exists(path.join(self.path, 'model_params.json')))

    def test_save_model(self):
        import os
        data = Data('iris', test_size = 30)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.save_model(filename='model', path=self.path)
        self.assertTrue(path.exists(path.join(self.path, 'model.pickle')))

    def test_save_predictions(self):
        data = Data('iris', test_size = 30)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        experiment.save_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    ####################################################################################################################
    #                                                    DEFENSES                                                      #
    ####################################################################################################################
        
    def test_insert_sklearn_preprocessor(self):
        preprocessor = SimpleImputer
        preprocessor_params = {'strategy': 'mean'}
        preprocessor = SimpleImputer(**preprocessor_params)
        data = Data('iris', test_size = 30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        experiment.insert_sklearn_preprocessor(name = "Preprocessor", preprocessor = preprocessor, position = 0)
        experiment.run(path = self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)

    def test_set_defence(self):
        data = Data('iris', test_size = 30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        defence = FeatureSqueezing(bit_depth=1, clip_values = (0, 255), apply_fit = True)
        model2 = Model(estimator, model_type = 'sklearn', path = self.path, defence = defence)
        experiment2 = Experiment(data = data, model = model2, is_fitted = False)
        experiment.run(path = self.path)
        experiment.set_defence(path.join(self.here,'../data/defences/44237341343125383753414498103201859838/defence_params.json'))
        experiment.run(path = self.path)
        experiment2.run(path = self.path)
        pred1 = experiment.predictions
        pred2 = experiment2.predictions
        self.assertTrue(not np.array_equal(pred1, pred2))  
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
        del self.path
        del self.file

    
    

