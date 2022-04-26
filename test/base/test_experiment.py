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
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.defences.postprocessor import HighConfidence
from os import path, listdir
class testExperiment(unittest.TestCase):
    def setUp(self):
        
        self.path = tempfile.mkdtemp()
        ART_DATA_PATH = self.path
        self.file = 'test_filename'

    def test_experiment(self):

        data = Data('iris', train_size = .8)
        self.assertIsInstance(data, Data)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        self.assertIsInstance(model, Model)
        experiment = Experiment(data = data, model = model)
        self.assertIsInstance(experiment, Experiment)
        self.assertIsInstance(experiment.data, Data)
        self.assertIsInstance(experiment.model, Model)
        self.assertIsInstance(experiment.refit, str)
        self.assertIsInstance(experiment.params, dict)
        self.assertIsInstance(experiment.scorers, dict)
        self.assertEqual(experiment.predictions, None)
        self.assertEqual(experiment.scores, None)
        self.assertEqual(experiment.time_dict, None)
        self.assertEqual(experiment.model_type, 'sklearn')


    def test_hash(self):

        data = Data('iris', train_size = .8)
        model = Model(KNeighborsRegressor(5), model_type = 'sklearn', path = self.path)
        model2 = Model(KNeighborsRegressor(4), model_type = 'sklearn', path = self.path)
        model3 = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        model4 = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment2 = Experiment(data = data, model = model2)
        experiment3 = Experiment(data = data, model = model3)
        experiment4 = Experiment(data = data, model = model4)
        experiment5 = deepcopy(experiment)
        self.assertEqual(hash(experiment), hash(experiment3))
        self.assertEqual(hash(experiment), hash(experiment5))
        self.assertNotEqual(hash(experiment), hash(experiment4))
        self.assertNotEqual(hash(experiment), hash(experiment2))


    def test_eq(self):
        data = Data('iris', train_size = .8)
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

    def test_set_metric_scorer(self):
        data = Data('iris', train_size = .8)
        data2 = Data('iris', train_size = .8)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        model2 = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data2, model = model)
        experiment2 = Experiment(data = data, model = model2)
        experiment3 = Experiment(data = data, model = model2, scorers = {'accuracy': accuracy_score})
        self.assertIsInstance(experiment.scorers, dict)
        self.assertIsInstance(experiment2.scorers, dict)
        self.assertIsInstance(experiment3.scorers, dict)
        self.assertIn('MSE', experiment.scorers)
        self.assertIn('accuracy', experiment3.scorers)
        self.assertIn('F1', experiment2.scorers)
        self.assertIsInstance(experiment3.scorers['accuracy'], Callable)
        self.assertEqual(experiment3.scorers['accuracy'], accuracy_score)
        self.assertNotEqual(experiment3.scorers, experiment.scorers)
        self.assertNotEqual(experiment3.scorers, experiment2.scorers)


    

    def test_run(self):
        data = Data('iris', train_size = .8)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIsInstance(experiment.scores, dict)
        self.assertIn('fit_time', experiment.time_dict)
        self.assertIn('pred_time', experiment.time_dict)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn('fit_time', experiment.time_dict)
        self.assertIn('pred_time', experiment.time_dict)
        self.assertIsInstance(experiment.scores, dict)
    
    def test_run_attack(self):
        data = Data('iris', train_size = .8)
        model = DecisionTreeClassifier()
        model = Model(model, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        # experiment.run(self.path)
        attack = BoundaryAttack(experiment.model.model, targeted = False, max_iter = 10, verbose = False)
        experiment.set_attack(attack)
    
        experiment.run(path = self.path)
        experiment.run_attack(path = self.path)
        self.assertIsInstance(experiment.attack, BoundaryAttack)
        self.assertIsInstance(experiment.adv, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict['adv_fit_time'], (int, float))
        self.assertIsInstance(experiment.time_dict['adv_pred_time'], (int, float))
        self.assertIsInstance(experiment.adv_scores, dict)



    def test_set_filename(self):
        data = Data('iris', train_size = .8)
        model = Model(KMeans(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.set_filename('test_filename')
        self.assertEqual(experiment.filename, 'test_filename')

    def test_set_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(estimator, targeted = False, max_iter = 10, verbose = False)
        old = hash(experiment)
        old_name = experiment.filename
        experiment.set_attack(attack)
        new = hash(experiment)
        new_name = experiment.filename
        self.assertIsInstance(experiment.attack, object)
        self.assertEqual(experiment.attack, attack)
        self.assertIn('Attack', experiment.params)
        self.assertIn('name', experiment.params['Attack'])
        self.assertIn('params', experiment.params['Attack'])
        self.assertNotEqual(old, new)
        self.assertNotEqual(old_name, new_name)


    
    
    # def test_insert_sklearn_preprocessor(self):
    #     preprocessor = SimpleImputer
    #     preprocessor_params = {'strategy': 'mean'}
    #     preprocessor = SimpleImputer(**preprocessor_params)
    #     data = Data('iris', train_size = .8)
    #     estimator = DecisionTreeClassifier()
    #     model = Model(estimator)
    #     experiment = Experiment(data = data, model = model)
    #     experiment.insert_sklearn_preprocessor(name = "Preprocessor", preprocessor = preprocessor, position = 0)
    #     experiment.run(path = self.path)
    #     self.assertIsInstance(experiment.predictions, (list, np.ndarray))
    #     self.assertIsInstance(experiment.time_dict, dict)
    #     self.assertIsInstance(experiment.scores, dict)

    def test_get_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(estimator, targeted = False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        self.assertIsInstance(experiment.get_attack(), object)
        self.assertEqual(experiment.get_attack(), attack)

   
    def test_evaluate(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(self.path)
        self.assertIsInstance(experiment.scores, dict)
        self.assertIsInstance(list(experiment.scores.values())[0], (int, float))

    def test_evaluate_attack(self):
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        attack = BoundaryAttack(experiment.model.model, targeted=False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        experiment.run_attack(path = self.path)
        self.assertIsInstance(experiment.adv_scores, dict)
        self.assertIsInstance(list(experiment.adv_scores.values())[0], (int, float))
    

    def test_save_data(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.save_data(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_experiment_params(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.save_experiment_params(data_params_file = self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
        self.assertTrue(path.exists(path.join(self.path, 'model_params.json')))

    def test_save_model(self):
        import os
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.save_model(filename='model', path=self.path)
        self.assertTrue(path.exists(path.join(self.path, 'model.pickle')))

    def test_save_predictions(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        experiment.save_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_scores(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        experiment.save_scores(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_adv_predictions(self):
        data = Data('iris', train_size = .8)
        model = DecisionTreeClassifier()
        model = Model(model, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(experiment.model.model, targeted=False, max_iter=10, verbose = False)
        experiment.run(path = self.path)
        experiment.set_attack(attack)
        experiment.run_attack(path = self.path)
        experiment.save_adv_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_attack(self):
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(self.path)
        attack = BoundaryAttack(experiment.model.model, targeted=False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        experiment.save_attack_params(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))


    def test_save_cv_scores(self):
        from sklearn.model_selection import GridSearchCV
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        grid = GridSearchCV(estimator, {'max_depth': [1, 2, 3, 4, 5]}, cv=3, return_train_score=True)
        model = Model(grid, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        experiment.save_cv_scores(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_adv_scores(self):
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        attack = BoundaryAttack(experiment.model.model, targeted=False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        experiment.run_attack(self.path)
        experiment.save_adversarial_samples(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
        
    def test_save_results(self):
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        experiment.run(path = self.path)
        attack = BoundaryAttack(experiment.model.model, targeted=False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        experiment.run_attack(path = self.path)
        experiment.save_results(path=self.path)
        files = listdir(self.path)
        self.assertTrue(path.exists(self.path))
        # self.assertIn('model', files)
        self.assertIn('predictions.json', files)
        self.assertIn('scores.json', files)
        self.assertIn('attack_params.json', files)
        # self.assertIn('defence_params.json', files)
        self.assertIn('model_params.json', files)
        self.assertIn('data_params.json', files)
    
    def test_set_defence(self):
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        model = Model(estimator, model_type = 'sklearn', path = self.path)
        experiment = Experiment(data = data, model = model)
        defence = FeatureSqueezing(bit_depth=1, clip_values = (0, 255))
        model2 = Model(estimator, model_type = 'sklearn', path = self.path, defence = defence)
        experiment2 = Experiment(data = data, model = model2)
        experiment.set_defence('defence_params.json')
        experiment.run(path = self.path)
        scores1 = experiment.scores
        experiment2.run(path = self.path)
        scores2 = experiment2.scores
        for key, value in scores1.items():
            self.assertTrue(scores1[key] == scores2[key])
        
        experiment.set_defence('blank_defence_params.json')
        experiment.run(path = self.path)
        scores3 = experiment.scores
        for key, value in scores3.items():
            self.assertTrue(scores1[key] != scores3[key])

    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
        del self.path
        del self.file

    
    

