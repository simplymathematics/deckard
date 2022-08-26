import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest
from deckard.base import Scorer
from sklearn.metrics import accuracy_score
class testScorer(unittest.TestCase):
    def setUp(self):
        self.input_folder = '../data/'
        self.predictions_file = '../data/attacks/44237341343125383753414498103201859838/265329158026005788/adversarial_predictions.json'
        self.ground_truth_file = '../data/control/predictions.json'
        self.scores_file = '../data/control/scores.json'
        self.scorer_function = accuracy_score
        self.scorer_name = 'ACC'
    def test_scorer(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        self.assertEqual(scorer.name, self.scorer_name)
        self.assertEqual(scorer.score_function, self.scorer_function)
        self.assertEqual(scorer.smaller_is_better, False)
        self.assertIsInstance(scorer.best, float)
        self.assertIsInstance(scorer, Scorer)
    
    def test_read_score_from_json(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        score = scorer.read_score_from_json(self.scores_file)
        self.assertEqual(score, 1.0)
        self.assertNotEqual(score, scorer.best)
        scorer.evaluate_score_from_json(self.scores_file)
        self.assertEqual(score, scorer.best)

    def test_read_data_from_json(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        predictions = scorer.read_data_from_json(self.predictions_file)
        ground_truth = scorer.read_data_from_json(self.ground_truth_file)
        self.assertEqual(predictions.shape, ground_truth.shape)
        self.assertEqual(predictions.shape[0], ground_truth.shape[0])
        self.assertEqual(predictions.shape[1], ground_truth.shape[1])
    
    def test_score(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        predictions = scorer.read_data_from_json(self.predictions_file)
        ground_truth = scorer.read_data_from_json(self.ground_truth_file)
        score = scorer.score(ground_truth, predictions)
        self.assertEqual(score, .1)
    
    def test_update_best(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        before = scorer.best
        scorer.update_best(1)
        self.assertNotEqual(before, scorer.best)
        self.assertEqual(scorer.best, 1)
    
    def test_get_best(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        self.assertEqual(scorer.get_best(), -1e9)
        scorer.update_best(1)
        self.assertEqual(scorer.get_best(), 1)
    
    def test_get_name(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        self.assertEqual(scorer.get_name(), self.scorer_name)
    
    def test_evaluate_function(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        score = scorer.evaluate_function(self.ground_truth_file, self.predictions_file)
        self.assertEqual(score, 0.1)

    def test_evaluate_function_from_json(self):
        scorer = Scorer(name = self.scorer_name, score_function = self.scorer_function)
        score = scorer.evaluate_score_from_json(self.scores_file)
        self.assertEqual(score, 1.0)

    def tearDown(self):
        pass