import logging
import unittest
import warnings
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

from deckard.base.utils import load_from_tup, factory

class testUtils(unittest.TestCase):
    def setUp(self):
        self.factory = {
            "module_class_string": "sklearn.linear_model.LogisticRegression",
            "super_cls": BaseEstimator,
            "penalty": "l2",
        }
        self.obj_gen = ("sklearn.linear_model.LogisticRegression", {"penalty": "l2"})
        
    def test_load_from_tuple(self):
        obj = load_from_tup(self.obj_gen)
        self.assertIsInstance(obj, BaseEstimator)
    
    def test_factory(self):
        obj = factory(**self.factory)
        self.assertIsInstance(obj, BaseEstimator)

