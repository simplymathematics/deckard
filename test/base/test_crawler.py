import warnings, os, unittest
from deckard.base.crawler import Crawler
from pandas import DataFrame
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

crawler_config = {
    "filenames": [
        "data_params",
        "defence_params",
        "experiment_params",
        "model_params",
        "attack_params",
        "predictions",
        "adversarial_predictions",
        "adversarial_scores",
        "scores",
        "time_dict",
    ],
    "filetype": "json",
    "results": "results.json",
    "status": "status.json",
    "scores_files": "scores.json",
    "adversarial_scores_file": "adversarial_scores.json",
    "schema": ["root", "path", "data", "directory", "layer", "defence_id", "attack_id"],
    "structured": [
        "defence_params",
        "attack_params",
        "adversarial_scores",
        "scores",
        "time_dict",
    ],
    "db": {},
    "root_folder": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data",
    ),
    "layers": ["control", "defences", "attacks"],
    "exclude": [],
}


class testCrawler(unittest.TestCase):
    def setUp(self):
        self.config = crawler_config

    def test_crawler(self):
        crawler = Crawler(config=self.config)
        self.assertIsInstance(crawler, Crawler)
        self.assertIs(crawler.data, None)
        self.assertIsInstance(crawler.config, dict)

    def test_crawl_folder(self):
        c1 = Crawler(config=self.config)
        df, sf = c1(self.config["root_folder"])
        result_file = os.path.join(self.config["root_folder"], self.config["results"])
        status_file = os.path.join(self.config["root_folder"], self.config["status"])
        self.assertTrue(os.path.isfile(result_file))
        self.assertTrue(os.path.isfile(status_file))
        self.assertTrue(os.stat(status_file).st_size >= 0)
        self.assertTrue(os.stat(result_file).st_size >= 0)
        self.assertIsInstance(df, DataFrame)
        self.assertIsInstance(sf, DataFrame)
        self.assertTrue(df.shape[0] >= 12 and df.shape[1] >= 11)
        self.assertTrue(sf.shape[0] >= 12 and sf.shape[1] >= 11)

    def tearDown(self):
        if os.path.isfile(self.config["status"]):
            os.remove(self.config["status"])
        if os.path.isfile(self.config["results"]):
            os.remove(self.config["results"])
