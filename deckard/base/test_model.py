import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from deckard.base.model import Model
from copy import deepcopy
class Test_Model(unittest.TestCase):
    
    def test_Model(self):
        estimators = [LogisticRegression(), DecisionTreeClassifier()]
        for estimator in estimators:
            model = Model(estimator, model_type = 'sklearn')
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
        model1 = Model(LogisticRegression(), model_type = 'sklearn')
        model2 = Model(LogisticRegression(), model_type = 'sklearn')
        model3 = Model(LogisticRegression(penalty = 'l1'), model_type = 'sklearn')
        self.assertEqual(model1, model2)
        self.assertNotEqual(model1, model3)
    def test_get_params(self):
        model1 = Model(LogisticRegression(), model_type = 'sklearn')
        self.assertIsInstance(model1.model_type, str)
        self.assertIsInstance(model1, Model)
        params = model1.get_params()
        self.assertIsInstance(params, dict)
    def test_set_params(self):
        model1 = Model(LogisticRegression(), model_type = 'sklearn')
        self.assertEqual(model1.model_type, 'sklearn')
        model1.set_params({'model_type': 'sklearn'})
        self.assertEqual(model1.model_type, 'sklearn')
        model1.set_params({'foo': 'bar'})
        self.assertEqual(model1.params['foo'], 'bar')
