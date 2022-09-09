import shutil
import warnings, logging, unittest, os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from deckard.base import Scorer, Data, Model, Experiment
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)
class testScorer(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"data")
        self.predictions_file = os.path.join(self.input_folder, 'attacks' , '44237341343125383753414498103201859838','265329158026005788', 'adversarial_predictions.json')
        self.ground_truth_file = os.path.join(self.input_folder, 'control', 'predictions.json')
        self.scores_file = os.path.join(self.input_folder, 'control' ,'scores.json')
        self.scorer = accuracy_score
        self.scorer_name = 'ACC'
        self.path = os.path.join(self.input_folder, 'tmp')
    
    def test_scorer(self):
        scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
        self.assertEqual(scorer.name[0], self.scorer_name)
        self.assertEqual(scorer.scorers[0], self.scorer)
        self.assertIsInstance(scorer, Scorer)
    
    def test_read_score_from_json(self):
        scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
        score = scorer.read_score_from_json(score_file = self.scores_file, name = 'ACC')
        self.assertEqual(score, 1.0)

    def test_read_data_from_json(self):
        scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
        predictions = scorer.read_data_from_json(self.predictions_file)
        ground_truth = scorer.read_data_from_json(self.ground_truth_file)
        self.assertEqual(predictions.shape, ground_truth.shape)
        self.assertEqual(predictions.shape[0], ground_truth.shape[0])
        self.assertEqual(predictions.shape[1], ground_truth.shape[1])
    
    def test_score(self):
        scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
        predictions = scorer.read_data_from_json(self.predictions_file)
        ground_truth = scorer.read_data_from_json(self.ground_truth_file)
        score = scorer.score(ground_truth, predictions)
        self.assertEqual(score['ACC'], 1)
    
    # def test_call(self):
    #     logger.debug(self.ground_truth_file, self.predictions_file)
    #     logger.debug(os.path.dirname(os.path.realpath(__file__)))
    #     scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
    #     score = scorer(self.ground_truth_file, self.predictions_file)
    #     self.assertEqual(score['ACC'], 0.1)


    # def test_call(self):
    #     data = Data('iris', test_size = 30)
    #     model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
    #     experiment = Experiment(data = data, model = model)
    #     experiment.run(self.path)
    #     scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
    #     scorer(self.ground_truth_file, self.predictions_file)
        


    # def test_save_scores(self):
    #     data = Data('iris', test_size = 30)
    #     model = Model(DecisionTreeClassifier(), model_type = 'sklearn', path = self.path)
    #     experiment = Experiment(data = data, model = model)
    #     experiment.run(path = self.path)
    #     scorer = Scorer(name = self.scorer_name, scorers = self.scorer)
    #     predictions = scorer.read_data_from_json(self.predictions_file)
    #     ground_truth = scorer.read_data_from_json(self.ground_truth_file)
    #     scorer.score(ground_truth, predictions)
    #     scorer.save_results(path = self.path)
  
    # def test_save_results(self):
    #     data = Data('iris', test_size = 30)
    #     estimator = DecisionTreeClassifier()
    #     model = Model(estimator, model_type = 'sklearn', path = self.path)
    #     experiment = Experiment(data = data, model = model)
    #     experiment.run(path = self.path)
    #     files = os.listdir(self.path)
    #     self.assertTrue(os.path.exists(self.path))
    #     scorer = Scorer()
    #     scorer(self.ground_truth_file, self.predictions_file, path = self.path)
    #     self.assertIn('scores.json', files)
        
    
    # def tearDown(self):
    #     if os.path.isfile('scores.json'):
    #         os.remove('scores.json')
    #     if os.path.isdir(self.path):
    #         shutil.rmtree(self.path)