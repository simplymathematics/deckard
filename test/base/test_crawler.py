import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import tempfile
import unittest
from deckard.base import Data, Model, Experiment, Crawler
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
from pandas import DataFrame

class testCrawler(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp(dir = "/tmp", prefix='deckard_', suffix = "")
        ART_DATA_PATH = self.path
        self.file = 'test_filename'
        data = Data('iris', test_size = 30)
        self.assertIsInstance(data, Data)
        model = Model(KNeighborsRegressor(), model_type = 'sklearn', path = path.join(self.path, 'regressor'))
        self.assertIsInstance(model, Model)
        experiment = Experiment(data = data, model = model)
        experiment.run(filename = 'scores.json', path = path.join(self.path, 'regressor'))
        model2 = Model(KNeighborsClassifier(), model_type = 'sklearn', path = path.join(self.path, 'classifier'))
        experiment2 = Experiment(data = data, model = model2)
        experiment2.run(filename = 'scores.json', path = path.join(self.path, 'classifier'))
        

    def test_crawler(self):
        tmp1 = path.join(self.path, 'regressor')
        crawler = Crawler(config_file = 'config.yml', path = path.join(self.path, 'regressor'), output = self.file)
        result = crawler.crawl_folder()
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['scores'], dict)
        self.assertIsInstance(result['scores']['MAPE'], float)
        self.assertIsInstance(result['scores']['MSE'], float)
        self.assertIsInstance(result['scores']['MAE'], float)
        self.assertIsInstance(result['scores']['R2'], float)

        crawler2 = Crawler(config_file = 'config.yml', path = path.join(self.path, 'classifier'), output = self.file)
        result2 = crawler2.crawl_folder()
        self.assertIsInstance(result2, dict)
        self.assertIsInstance(result2['scores'], dict)
        self.assertIsInstance(result2['scores']['F1'], float)
        self.assertIsInstance(result2['scores']['ACC'], float)
        self.assertIsInstance(result2['scores']['PREC'], float)
        self.assertIsInstance(result2['scores']['REC'], float)
        self.assertIsInstance(result2['scores']['AUC'], float)
        
    
    def test_crawl_tree(self):
        crawler3 = Crawler(config_file = 'config.yml', path = self.path, output = self.file)
        result3 = crawler3.crawl_tree()
        self.assertIsInstance(result3, dict)
        self.assertEqual(len(list(result3)), 2)
        for result_dict in result3.keys():
            tmp = result3[result_dict]
            self.assertIsInstance(tmp, dict)
            self.assertIsInstance(tmp['scores'], dict)
            self.assertIsInstance(tmp['data_params'], dict)
            self.assertIsInstance(tmp['model_params'], dict)
    
    def test_clean_data(self):
        crawler = Crawler(config_file = 'config.yml', path = self.path, output = self.file)
        result = crawler.crawl_tree()
        clean_result = crawler.clean_data(result)
        self.assertIsInstance(clean_result, DataFrame)
        self.assertEqual(len(clean_result), 2)
        self.assertEqual(len(clean_result.columns), 17)

    def test_save_date(self):
        crawler = Crawler(config_file = 'config.yml', path = self.path, output = self.file)
        crawler2 = Crawler(config_file = 'config.yml', path = self.path, output = self.file)
        result = crawler.crawl_tree()
        clean_result = crawler.clean_data(result)
        json_file = crawler.save_data(clean_result, filetype = 'json')
        self.assertTrue(path.exists(json_file))
        csv_file = crawler2.save_data(clean_result, filetype = 'csv')
        self.assertTrue(path.exists(csv_file))
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
        del self.path
        del self.file