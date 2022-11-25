import logging
import unittest
import warnings
from sklearn.base import BaseEstimator
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

from deckard.base.utils import load_from_tup, factory, parse_config_for_libraries

class testUtils(unittest.TestCase):
    def setUp(self):
        self.path = "configs"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.factory = {
            "module_class_string": "sklearn.linear_model.LogisticRegression",
            "super_cls": BaseEstimator,
            "penalty": "l2",
        }
        self.obj_gen = ("sklearn.linear_model.LogisticRegression", {"penalty": "l2"})
        self.regex =  "params.yaml"
        self.file = Path(self.path) / self.regex
        self.output = "requirements.txt"
        assert self.file.exists()
        
    def test_load_from_tuple(self):
        obj = load_from_tup(self.obj_gen)
        self.assertIsInstance(obj, BaseEstimator)
    
    def test_factory(self):
        obj = factory(**self.factory)
        self.assertIsInstance(obj, BaseEstimator)
    
    def test_parse_config_for_libraries(self):
        (libraries , path)= parse_config_for_libraries(path=self.path, regex=self.regex, output=self.output)
        self.assertEqual(libraries, ["sklearn"])
        with open(path, "r") as f:
            for count, _ in enumerate(f):
                pass
        self.assertEqual(count + 1, len(libraries))

