import unittest, tempfile, shutil, os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from deckard.base import Model, Data
from copy import deepcopy
from art.defences.preprocessor import FeatureSqueezing
from art.defences.postprocessor import GaussianNoise
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.regression import ScikitlearnRegressor
import numpy as np

class testModel(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_model'
        self.url = "https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1"
    
    def test_model(self):
        estimators = [LogisticRegression(), DecisionTreeClassifier()]
        for estimator in estimators:
            model = Model(estimator, model_type = 'sklearn')
            self.assertIsInstance(model, Model)
            self.assertIsInstance(model.model_type, str)
            self.assertIsInstance(model.model, object)
        model = Model(model = self.file, path = self.path, url = self.url, model_type = 'tensorflow')
        self.assertIsInstance(model, Model)
        self.assertIsInstance(model.model_type, str)
        self.assertIsInstance(model.model, object)
    
    def test_hash(self):
        model1 = Model(LogisticRegression(), model_type = 'sklearn')
        model2 = Model(LogisticRegression(), model_type = 'sklearn')
        model3 = Model(DecisionTreeClassifier(), model_type = 'sklearn')
        model4 = deepcopy(model1)
        self.assertEqual(model1.__hash__(), model2.__hash__())
        self.assertNotEqual(model3.__hash__(), model2.__hash__())
        self.assertEqual(model1.__hash__(), model4.__hash__())
    
    def test_eq(self):
        model1 = Model(LogisticRegression('l2'), model_type = 'sklearn')
        model2 = Model(LogisticRegression(), model_type = 'sklearn')
        model3 = Model(LogisticRegression(penalty = 'l1'), model_type = 'sklearn')
        self.assertEqual(model1, model2)
        self.assertNotEqual(model1, model3)
    
    def test_get_params(self):
        model1 = Model(LogisticRegression(), model_type = 'sklearn')
        self.assertIsInstance(model1.model_type, str)
        self.assertIsInstance(model1, Model)
        self.assertIsInstance(model1.params, dict)

        model2 = Model(model = self.file, path = self.path, url = self.url, model_type = 'tensorflowv1')
        self.assertIsInstance(model2.model_type, str)
        self.assertIsInstance(model2, Model)
        self.assertIsInstance(model2.params, dict)
    
    def test_set_params(self):
        model1 = Model(LogisticRegression(), model_type = 'sklearn')
        self.assertEqual(model1.model_type, 'sklearn')
        model1.set_params({'penalty' : 'l1'})
        model1.set_params({'clip_values': (0, 1)})
        dictionary = model1.model.__dict__
        self.assertEqual(dictionary['_clip_values'][0], 0)
        self.assertEqual(dictionary['_clip_values'][1], 1)
        self.assertIn("penalty='l1'", str(dictionary))
        self.assertRaises(ValueError, model1.set_params, {'potato' : 'potato'})

    def test_save_model(self):
        model1 = Model(LogisticRegression(), model_type = 'sklearn', path = self.path, classifier=False)
        filename = model1.save_model(path = self.path, filename = self.file)
        self.assertTrue(os.path.exists(filename+".pickle"))

    def test_load(self):
        model = Model(LogisticRegression(), model_type = 'sklearn', path = self.path, classifier=False, art = False)
        file = model.save_model(filename = 'model', path = self.path)
        model2 = Model(path = self.path, model = file, model_type = 'sklearn', classifier=False, art = False)
        self.assertEqual(model, model2)
        
    def test_load_with_defence(self):
        defence = FeatureSqueezing(bit_depth = 4, clip_values = (0, 1))
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', defence = defence, path = self.path, classifier = True, art = True)
        file = model.save_model(filename = 'model', path = self.path)
        model2 = Model(path = self.path, model = file+".pickle", model_type = 'sklearn', classifier = True, art = True, defence = defence)
        self.assertEqual(model, model2)
        
    def test_set_defence_params(self):
        from art.defences.preprocessor.preprocessor import Preprocessor
        fsq = FeatureSqueezing(bit_depth = 4, clip_values = (0, 1))
        self.assertIsInstance(fsq, Preprocessor)
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', defence = fsq, path = self.path)
        self.assertIn('FeatureSqueezing', str(model.defence))
        self.assertEqual(model.params['Defence']['params']['bit_depth'], 4)
        self.assertEqual(model.params['Defence']['params']['clip_values'], (0, 1))
        self.assertEqual(model.params['Defence']['type'], 'preprocessor')

    def test_initialize_art_classifier(self):
        defence = FeatureSqueezing(bit_depth = 4, clip_values = (0, 1))
        model = Model(DecisionTreeClassifier(), model_type = 'sklearn', defence = defence, art = True)
        self.assertIsInstance(model.model, object)
        self.assertIsInstance(model.model_type, str)
        self.assertIsInstance(model.params, dict)
        self.assertIsInstance(model.defence, object)
        self.assertIsInstance(model.model, ScikitlearnClassifier)

    def test_initialize_art_regressor(self):
        defence = FeatureSqueezing(bit_depth = 4, clip_values = (0, 1))
        model = Model(LinearRegression(), model_type = 'sklearn', defence = defence, art = True, classifier=False)
        self.assertIsInstance(model.model, object)
        self.assertIsInstance(model.model_type, str)
        self.assertIsInstance(model.params, dict)
        self.assertIsInstance(model.defence, object)
        self.assertIsInstance(model.model, ScikitlearnRegressor)
        
    def test_is_supervised(self):
        data = Data('iris', train_size = .8)
        model1 = Model(KNeighborsClassifier(), model_type = 'sklearn')
        model2 = Model(KMeans(), model_type = 'sklearn')
        self.assertTrue(model1.is_supervised)
        self.assertFalse(model2.is_supervised)

    def test_fit(self):
        data = Data('iris', train_size = .8)
        model = Model(KNeighborsClassifier(), model_type = 'sklearn')
        model.fit(data.X_train, data.y_train)
        self.assertIsInstance(model.predict(data.X_test), np.ndarray)
    

    def test_predict(self):
        data = Data('iris', train_size = .8)
        model = Model(KNeighborsClassifier(), model_type = 'sklearn')
        model.fit(data.X_train, data.y_train)
        predictions = model.predict(data.X_test)
        self.assertIsInstance(predictions, (list, np.ndarray))
        defence = FeatureSqueezing(bit_depth = 4, clip_values = (0, 1))
        model = Model(LinearRegression(), model_type = 'sklearn', defence = defence, classifier = False)
        model.fit(data.X_train, data.y_train)
        predictions = model.predict(data.X_test)
        self.assertIsInstance(predictions, (list, np.ndarray))
    
    def test_str(self):
        data = Data('iris', train_size = .8)
        params = dict(data)
        data2 = Data(**params)
        self.assertEqual(data, data2)
        self.assertDictEqual(dict(data), dict(data2))
    
    def tearDown(self):
        shutil.rmtree(self.path)