import tempfile
import unittest
import warnings
import yaml
import json
from pathlib import Path

import numpy as np
from deckard.base.hashable import BaseHashable, my_hash, from_yaml
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class testBaseHashable(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.filetype = "yaml"
        self.config = """
        !BaseHashable
            files:
                path : tmp
                filetype : yaml
        """
        self.config2 = """
        !BaseHashable
            files:
                path : tmp
                filetype : yaml
        """
        self.config3 = """
        !BaseHashable
            files:
                path : tmp
                filetype : yml
        """
        self.config4 = """
            files:
                path : tmp
                filetype : yaml
        """
        yaml.add_constructor("!BaseHashable", BaseHashable)
        self.hashable = yaml.load(self.config, Loader=yaml.FullLoader)
        self.hashable2 = yaml.load(self.config2, Loader=yaml.FullLoader)
        self.hashable3 = yaml.load(self.config3, Loader=yaml.FullLoader)
        document = "!BaseHashable\n" + self.config4
        self.hashable4 = yaml.load(document, Loader=yaml.FullLoader)

    def test_new(self):
        self.assertIsInstance(self.hashable, BaseHashable)
        
    def test_hash(self):
        self.assertEqual(my_hash(self.hashable), my_hash(self.hashable2))
        self.assertEqual(hash(self.hashable), hash(self.hashable2))
        self.assertNotEqual(my_hash(self.hashable), my_hash(self.hashable3))
        self.assertNotEqual(hash(self.hashable), hash(self.hashable3))
    
    def test_repr(self):
        self.assertIsInstance(str(self.hashable), str)
    
    def test_to_dict(self):
        self.assertIsInstance(self.hashable.to_dict(), dict)
        
    def test_to_yaml(self):
        string_ = self.hashable.to_yaml()
        with open(self.path + "/test.yaml", "w") as f:
            yaml.dump(string_, f)
        with open(self.path + "/test.yaml", "r") as f:
            self.assertEqual(yaml.load(f, Loader=yaml.FullLoader), string_)
    
    def test_from_yaml(self):
        string_ = self.hashable.to_yaml()
        with open(self.path + "/test.yaml", "w") as f:
            yaml.dump(string_, f)
        hashable = from_yaml(hashable = self.hashable, filename = self.path + "/test.yaml")
        self.assertEqual(hashable, self.hashable)
    
    def test_load(self):
        self.assertRaises(NotImplementedError, self.hashable.load)
    
    def save_yaml(self):
        filename = self.hashable.save_yaml()
        test_filename = Path(self.path) / my_hash(self) + "." + self.filetype
        self.assertEqual(filename, test_filename)
        
    def tearDown(self):
        import shutil
        if Path(self.path).exists():
            shutil.rmtree(self.path)
