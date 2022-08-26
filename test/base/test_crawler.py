import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import tempfile
import unittest
from deckard.base import Data, Model, Experiment
from deckard.base.crawler import Crawler, crawler_config
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
from os import path, remove
from pandas import DataFrame

class testCrawler(unittest.TestCase):
    def setUp(self):
        self.path = "../data/"
        self.file = "../data/tmp_results.csv"
        self.config = crawler_config

    def test_crawler(self):
        crawler = Crawler(config = self.config, path = self.path, output = self.file)
        self.assertIsInstance(crawler, Crawler)
        self.assertIs(crawler.data, None)
        self.assertIsInstance(crawler.config, dict)
        self.assertIsInstance(crawler.path, str)
        self.assertEqual(crawler.output, self.file)
    
    def test_crawl_folder(self):
        c1 = Crawler(config = self.config, path = self.path, output = self.file)
        d1 = c1()
        self.assertIsInstance(d1, DataFrame)
        print(c1.output)
        self.assertTrue(path.isfile(c1.output))
        remove(self.file)

    def tearDown(self):
        pass