import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import tempfile
import unittest
from deckard.base import Data, Model, Experiment, experiment
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

        self.path = path.abspath(tempfile.mkdtemp())
        ART_DATA_PATH = self.path
        self.file = "test_filename"
        self.here = path.dirname(path.abspath(__file__))

    def test_experiment(self):
        data = Data("iris", test_size=30)
        self.assertIsInstance(data, Data)
        model = Model(KNeighborsRegressor(), model_type="sklearn")
        model()
        self.assertIsInstance(model, Model)
        experiment = Experiment(data=data, model=model)
        self.assertIsInstance(experiment, Experiment)
        self.assertIsInstance(experiment.data, Data)
        self.assertIsInstance(experiment.model, Model)
        self.assertIsInstance(experiment.params, dict)
        self.assertEqual(
            experiment.model.model_type,
            "sklearn",
        )

    def test_hash(self):
        data = Data("iris", test_size=30)
        model = Model(KNeighborsRegressor(5), model_type="sklearn", path=self.path)
        model2 = Model(KNeighborsRegressor(4), model_type="sklearn", path=self.path)
        model3 = Model(KNeighborsRegressor(), model_type="sklearn", path=self.path)
        model4 = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment2 = Experiment(data=data, model=model2)
        experiment3 = Experiment(data=data, model=model3)
        experiment4 = Experiment(data=data, model=model4)
        experiment5 = deepcopy(experiment)
        self.assertEqual(experiment, experiment3)
        self.assertEqual(hash(experiment.data), hash(experiment3.data))
        self.assertEqual(hash(experiment.model), hash(experiment3.model))
        self.assertEqual(hash(experiment), hash(experiment3))
        self.assertEqual(hash(experiment), hash(experiment5))
        self.assertNotEqual(hash(experiment), hash(experiment4))
        self.assertNotEqual(hash(experiment), hash(experiment2))

    def test_eq(self):
        data = Data("iris", test_size=30)
        model = Model(
            KNeighborsRegressor(5),
            model_type="sklearn",
            path=self.path,
            classifier=False,
        )
        model2 = Model(
            KNeighborsRegressor(4),
            model_type="sklearn",
            path=self.path,
            classifier=False,
        )
        model3 = Model(
            KNeighborsRegressor(),
            model_type="sklearn",
            path=self.path,
            classifier=False,
        )
        model4 = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment2 = Experiment(data=data, model=model2)
        experiment3 = Experiment(data=data, model=model3)
        experiment4 = Experiment(data=data, model=model4)
        experiment5 = deepcopy(experiment)
        self.assertEqual(experiment.data, experiment3.data)
        self.assertEqual(experiment.model, experiment3.model)
        self.assertEqual(experiment, experiment3)
        self.assertEqual(experiment, experiment5)
        self.assertNotEqual(experiment, experiment4)
        self.assertNotEqual(experiment, experiment2)

    def test_run(self):
        data = Data("iris", test_size=30)
        model = Model(
            KNeighborsRegressor(),
            model_type="sklearn",
            path=self.path,
            classifier=False,
        )
        experiment = Experiment(data=data, model=model)
        experiment(path=self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn("fit_time", experiment.time_dict)
        self.assertIn("pred_time", experiment.time_dict)
        model = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment(path=self.path)
        self.assertIsInstance(experiment.predictions, (list, np.ndarray))
        self.assertIsInstance(experiment.time_dict, dict)
        self.assertIn("fit_time", experiment.time_dict)
        self.assertIn("pred_time", experiment.time_dict)

    def test_save_cv_scores(self):
        from sklearn.model_selection import GridSearchCV

        data = Data("iris", test_size=30)
        estimator = DecisionTreeClassifier()
        grid = GridSearchCV(
            estimator, {"max_depth": [1, 2, 3, 4, 5]}, cv=3, return_train_score=True
        )
        model = Model(grid, model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment(path=self.path)
        experiment.save_cv_scores(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_data(self):
        data = Data("iris", test_size=30)
        model = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment.save_data(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    def test_save_params(self):
        data = Data("iris", test_size=30)
        model = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment.save_params(path=self.path)
        self.assertTrue(path.exists(path.join(self.path, "model_params.json")))

    def test_save_model(self):
        data = Data("iris", test_size=30)
        model = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment.save_model(filename="model", path=self.path)
        self.assertTrue(path.exists(path.join(self.path, "model")))

    def test_save_predictions(self):
        data = Data("iris", test_size=30)
        model = Model(DecisionTreeClassifier(), model_type="sklearn", path=self.path)
        experiment = Experiment(data=data, model=model)
        experiment(path=self.path)
        experiment.save_predictions(filename=self.file, path=self.path)
        self.assertTrue(path.exists(path.join(self.path, self.file)))

    ####################################################################################################################
    #                                                    DEFENSES                                                      #
    ####################################################################################################################

    def test_insert_sklearn_preprocessor(self):
        config = {
            "name" : "sklearn.preprocessing.StandardScaler",
            "params" :{
                "with_mean" : True,
                "with_std" : True
            }
        }
        config2 = {
            "name" : "sklearn.impute.SimpleImputer",
            "params" :{
                "strategy" : "mean"
            }
        }
        data = Data("iris", test_size=30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator)
        model.insert_sklearn_preprocessor(
            name = "Preprocessor", preprocessor=config, position=0
        )
        model.insert_sklearn_preprocessor(
            name="Featurizer", preprocessor=config2, position=1
        )
        model()
        self.assertIn("Preprocessor", str(model.model.steps))
        self.assertIn(str("with_mean"), str(model.model.steps[0][1].get_params()))
        self.assertDictContainsSubset(config['params'], model.model.steps[0][1].get_params())
        self.assertIn("Featurizer", str(model.model.steps))
        self.assertIn(str("strategy"), str(model.model.steps[1][1].get_params()))
        self.assertDictContainsSubset(config2['params'], model.model.steps[1][1].get_params())
        self.assertIn(str(config['params']), str(model.params))
        self.assertIn(str(config2['params']), str(model.params))
        
    def test_insert_art_defence(self):
        config = {
             "name" : "art.defences.preprocessor.FeatureSqueezing",
             "params" : {
                 "clip_values" : (0, 1), 
                 "bit_depth" : 4
             }
        }
        data = Data("iris", test_size=30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator)
        model.insert_art_defence(config)
        self.assertIn("FeatureSqueezing", str(model.params['Defence']))
        self.assertIn(str("clip_values"), str(model.params['Defence']))
    
    def test_experiment_param_printing(self):
        config = {
             "name" : "art.defences.preprocessor.FeatureSqueezing",
             "params" : {
                 "clip_values" : (0, 1), 
                 "bit_depth" : 4
             }
        }
        data = Data("iris", test_size=30)
        estimator = DecisionTreeClassifier()
        model = Model(estimator)
        model.insert_art_defence(config)
        self.assertIn("FeatureSqueezing", str(model.params['Defence']))
        self.assertIn(str("clip_values"), str(model.params['Defence']))
        exp = Experiment(data, model)
        self.assertIn("Defence", str(exp))
        self.assertIn("Experiment", str(exp))
        self.assertIn("Model", str(exp))
        self.assertIn("Data", str(exp))
        

    def tearDown(self) -> None:
        from shutil import rmtree

        rmtree(self.path)
        del self.path
        del self.file


if __name__ == "__main__":
    unittest.main()
