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