import warnings
import unittest
from deckard.base.crawler import Crawler, crawler_config
from os import path, remove
from pandas import DataFrame
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class testCrawler(unittest.TestCase):
    def setUp(self):
        self.path = "../data/"
        self.file = "results.json"
        self.config = crawler_config

    def test_crawler(self):
        crawler = Crawler(config = self.config)
        self.assertIsInstance(crawler, Crawler)
        self.assertIs(crawler.data, None)
        self.assertIsInstance(crawler.config, dict)
        self.assertIsInstance(crawler.path, str)
        self.assertEqual(crawler.result_file, self.file)
    
    def test_crawl_folder(self):
        c1 = Crawler(config = self.config)
        d1 = c1()
        self.assertIsInstance(d1, dict)
        self.assertTrue(path.isfile(c1.result_file))
        self.assertTrue(path.isfile(c1.status_file))
        remove(c1.result_file)
        remove(c1.status_file)

    def tearDown(self):
        pass