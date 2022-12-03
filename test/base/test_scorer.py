import logging
import os
import shutil
import unittest
import warnings
import yaml
import json
from pathlib import Path
import numpy as np
from pandas import DataFrame, Series
from deckard.base import Scorer
from deckard.base.experiment import config


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

yaml.add_constructor("!Scorer", Scorer)
class testScorer(unittest.TestCase):
    def setUp(self):
        
        self.path = "reports"
        self.config = config
        self.scorer = yaml.load("!Scorer" + str(config), Loader=yaml.FullLoader)
        self.scores_file = Path(self.path) / "scores.json"
        self.pred_file = Path(self.path) / "predictions.json"
        self.ground_file = Path(self.path) / "ground_truth.json"
        self.scores_file = Path(self.path) / "scores.json"
        score_dict = {"ACC": 0.5, "F1": 0.5}
        Path(self.path).mkdir(parents=True, exist_ok=True)
        with open(self.scores_file, "w") as f:
            json.dump(score_dict, f)
        assert self.scores_file.exists()
        preds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        with open(self.pred_file, "w") as f:
            json.dump(preds, f)
        assert self.pred_file.exists()
        with open(self.ground_file, "w") as f:
            json.dump(preds, f)
        assert self.ground_file.exists()
        with open(self.scores_file, "w") as f:
            json.dump(score_dict, f)
        assert self.scores_file.exists()
        

    def test_scorer(self):
        self.assertIsInstance(self.scorer, Scorer)

    def test_read_score_from_json(self):
        scorer = self.scorer
        score = scorer.read_score_from_json(score_file=self.scores_file, name="ACC")
        self.assertIsInstance(score, float)

    def test_read_data_from_json(self):
        scorer = self.scorer
        predictions = scorer.read_data_from_json(self.pred_file)
        self.assertTrue(isinstance(predictions, Series))

    def test_score(self):
        scorer = self.scorer
        predictions = scorer.read_data_from_json(self.pred_file)
        ground_truth = scorer.read_data_from_json(self.pred_file)
        score = scorer.score_from_memory(ground_truth, predictions)
        self.assertIsInstance(score, Series)

    def test_save_results(self):
        scorer = self.scorer
        scores = {"ACC": 0.5, "F1": 0.5}
        filename = self.scores_file
        path = self.path
        full_path = scorer.save(scores)
        self.assertTrue(Path(full_path).exists())
        



    def tearDown(self):
        if os.path.isfile("scores.json"):
            os.remove("scores.json")
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
