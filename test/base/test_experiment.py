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
        ART_DATA_PATH = path.join(self.path)
        self.file = 'test_filename'

    def test_experiment(self):

        data = Data('iris', train_size = .8)
        self.assertIsInstance(data, Data)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn')
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
        model = Model(KNeighborsRegressor(5), model_type = 'sklearn')
        model2 = Model(KNeighborsRegressor(4), model_type = 'sklearn')
        model3 = Model(KNeighborsRegressor(), model_type = 'sklearn')
        model4 = Model(DecisionTreeClassifier(), model_type = 'sklearn')
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
        model = Model(KNeighborsRegressor(5), model_type = 'sklearn')
        model2 = Model(KNeighborsRegressor(4), model_type = 'sklearn')
        model3 = Model(KNeighborsRegressor(), model_type = 'sklearn')
        model4 = Model(DecisionTreeClassifier(), model_type = 'sklearn')
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
        data2 = Data('diabetes')
        model = Model(KNeighborsRegressor(), model_type = 'sklearn')
        model2 = Model(DecisionTreeClassifier(), model_type = 'sklearn')
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


    def test_build_supervised_model(self):
        data = Data('iris', train_size = .8)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        y_pred, (fit_time, pred_time) = experiment._build_supervised_model()
        self.assertIsInstance(y_pred, (list, np.ndarray))
        self.assertIsInstance(fit_time, (int, float))
        self.assertIsInstance(pred_time, (int, float))
        self.assertEqual(len(y_pred), len(data.y_test))
        self.assertTrue(experiment.is_fitted)
    
    def test_build_unsupervised_model(self):
        data = Data('iris', train_size = .8)
        model = Model(KMeans(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        y_pred, (fit_pred_time) = experiment._build_unsupervised_model()
        self.assertIsInstance(y_pred, (list, np.ndarray))
        self.assertIsInstance(fit_pred_time, (int, float))
        self.assertEqual(len(y_pred), len(data.y_test))
        self.assertFalse(experiment.is_fitted)
        
    
    def test_build_time_series_model(self):
        #TODO:
        pass
    

    def test__is_supervised(self):
        data = Data('iris', train_size = .8)
        model1 = Model(KNeighborsClassifier(), model_type = 'sklearn')
        model2 = Model(KMeans(), model_type = 'sklearn')
        experiment = Experiment(model = model1, data=data)
        experiment2 = Experiment(model = model2, data=data)
        self.assertTrue(experiment._is_supervised())
        self.assertFalse(experiment2._is_supervised())


    def test_build_model(self):
        data = Data('iris', train_size = .8)
        #Experiment 1
        model1 = Model(KNeighborsClassifier(), model_type = 'sklearn')
        experiment = Experiment(model = model1, data=data)
        experiment._build_model()
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn('fit_time', experiment.time_dict)
        self.assertIn('pred_time', experiment.time_dict)
        # Experiment 2
        model2 = Model(KMeans(), model_type = 'sklearn')
        experiment2 = Experiment(model = model2, data=data)
        experiment2._build_model()
        self.assertIsInstance(experiment2.predictions, (list, np.ndarray))
        self.assertEqual(len(experiment2.predictions), len(data.y_test))
        self.assertIn('fit_pred_time', experiment2.time_dict)

    def test_run(self):
        data = Data('iris', train_size = .8)
        model = Model(KMeans(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.run()
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIsInstance(experiment.scores, dict)
        self.assertIn('fit_pred_time', experiment.time_dict)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.run()
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn('fit_time', experiment.time_dict)
        self.assertIn('pred_time', experiment.time_dict)
        self.assertIsInstance(experiment.scores, dict)
    
    def test_run_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(estimator, targeted = False, max_iter = 10, verbose = False)
        experiment.set_attack(attack)
    
        experiment.run()
        experiment.run_attack()
        self.assertIsInstance(experiment.attack, BoundaryAttack)
        self.assertIsInstance(experiment.adv, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict['adv_fit_time'], (int, float))
        self.assertIsInstance(experiment.time_dict['adv_pred_time'], (int, float))
        self.assertIsInstance(experiment.adv_scores, dict)



    def test_set_filename(self):
        data = Data('iris', train_size = .8)
        model = Model(KMeans(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.set_filename('test_filename')
        self.assertEqual(experiment.filename, 'test_filename')

    def test_set_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
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


    def test_set_defense(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        preprocessor = FeatureSqueezing(clip_values = (0, 255))
        old = hash(experiment)
        old_name = experiment.filename
        experiment.set_defense(preprocessor)
        new = hash(experiment)
        new_name = experiment.filename
        self.assertIsInstance(experiment.defense, object)
        self.assertEqual(experiment.defense, preprocessor)
        self.assertIn('Defense', experiment.params)
        self.assertIn('name', experiment.params['Defense'])
        self.assertIn('params', experiment.params['Defense'])
        self.assertNotEqual(old, new)
        self.assertNotEqual(old_name, new_name)

        experiment2 = Experiment(data = data, model = model)
        postprocessor = HighConfidence(.95)
        experiment2.set_defense(postprocessor)
        self.assertIsInstance(experiment2.defense, object)
        self.assertEqual(experiment2.defense, postprocessor)
        self.assertIn('Defense', experiment2.params)
        self.assertIn('name', experiment2.params['Defense'])
        self.assertIn('params', experiment2.params['Defense'])
    
    def test_insert_sklearn_preprocessor(self):
        preprocessor = SimpleImputer
        preprocessor_params = {'strategy': 'mean'}
        preprocessor = SimpleImputer(**preprocessor_params)
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        experiment.insert_sklearn_preprocessor(name = "Preprocessor", preprocessor = preprocessor, position = 0)

    def test_get_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(estimator, targeted = False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        self.assertIsInstance(experiment.get_attack(), object)
        self.assertEqual(experiment.get_attack(), attack)

    def test_get_defense(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        preprocessor = FeatureSqueezing(clip_values = (0, 255))
        experiment.set_defense(preprocessor)
        self.assertIsInstance(experiment.get_defense(), object)
        self.assertEqual(experiment.get_defense(), preprocessor)

    def test_evaluate(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment._build_model()
        experiment.evaluate()
        self.assertIsInstance(experiment.scores, dict)
        self.assertIsInstance(list(experiment.scores.values())[0], (int, float))

    def test_evaluate_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment._build_model()
        experiment.set_attack(attack)
        experiment._build_attack()
        experiment.evaluate_attack()
        self.assertIsInstance(experiment.adv_scores, dict)
        self.assertIsInstance(list(experiment.adv_scores.values())[0], (int, float))
    
    def test_evaluate_defense(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator)
        experiment = Experiment(data = data, model = model)
        experiment.run()
        preprocessor = FeatureSqueezing(clip_values = (0, 1))
        postprocessor = HighConfidence(.95)
        old = hash(experiment)
        experiment.set_defense(preprocessor)
        experiment2 = Experiment(data = data, model = model)
        experiment2.set_defense(postprocessor)
        experiment2.run()
        self.assertNotEqual(hash(experiment), hash(experiment2))
        self.assertNotEqual(old, hash(experiment))

    def test_save_data(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.save_data(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_experiment_params(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.save_experiment_params(data_params_file = self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
        self.assertTrue(path.exists(path.join(self.path, 'model_params.json')))

    def test_save_model(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.save_model(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_predictions(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.run()
        experiment.save_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_scores(self):
        data = Data('iris', train_size = .8)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.run()
        experiment.save_scores(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_adv_predictions(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment.run()
        experiment.set_attack(attack)
        experiment.run_attack()
        experiment.save_adv_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_attack(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        experiment.save_attack(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_defense(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        preprocessor = FeatureSqueezing(clip_values = (0, 255))
        experiment.set_defense(preprocessor)
        experiment.save_defense(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_cv_scores(self):
        from sklearn.model_selection import GridSearchCV
        data = Data('iris', train_size = .8)
        estimator = DecisionTreeClassifier()
        grid = GridSearchCV(estimator, {'max_depth': [1, 2, 3, 4, 5]}, cv=3, return_train_score=True)
        model = Model(grid, model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        experiment.run()
        experiment.save_cv_scores(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
    
    def test_save_adv_scores(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment.set_attack(attack)
        experiment.run()
        experiment.run_attack()
        experiment.save_adversarial_samples(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))
        
    def test_save_results(self):
        data = Data('iris', train_size = .8)
        estimator = ScikitlearnClassifier(model = DecisionTreeClassifier())
        model = Model(estimator, model_type = 'sklearn')
        experiment = Experiment(data = data, model = model)
        preprocessor = FeatureSqueezing(clip_values = (0, 255))
        experiment.set_defense(preprocessor)
        attack = BoundaryAttack(model.model, targeted=False, max_iter=10, verbose = False)
        experiment.run()
        experiment.set_attack(attack)
        experiment.run_attack()
        experiment.save_results(path=self.path)
        files = listdir(self.path)
        self.assertTrue(path.exists(self.path))
        # self.assertIn('model', files)
        self.assertIn('predictions.json', files)
        self.assertIn('adversarial_predictions.json', files)
        self.assertIn('scores.json', files)
        self.assertIn('adversarial_scores.json', files)
        self.assertIn('attack_params.json', files)
        self.assertIn('defense_params.json', files)
        self.assertIn('model_params.json', files)
        self.assertIn('data_params.json', files)
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
        del self.path
        del self.file

    
    

