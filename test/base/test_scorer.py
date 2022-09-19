import shutil
import warnings, logging, unittest, os
from deckard.base import Scorer, Data, Model, Experiment
from sklearn.tree import DecisionTreeClassifier
from tempfile import mkdtemp
from deckard.base.scorer import REGRESSOR_SCORERS, CLASSIFIER_SCORERS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics import (
    f1_score,
    roc_curve,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    make_scorer,
)


logger = logging.getLogger(__name__)


class testScorer(unittest.TestCase):
    def setUp(self):
        data = Data("iris")
        model = Model(DecisionTreeClassifier())
        experiment = Experiment(data, model)
        self.path = mkdtemp()
        self.model_name = "test_model"
        self.predictions_file = os.path.join(self.path, "predictions.json")
        self.ground_truth_file = os.path.join(self.path, "ground_truth.json")
        self.scores_file = os.path.join(self.path, "scores.json")
        self.config = {"ACC": accuracy_score}
        self.cl_config = CLASSIFIER_SCORERS
        self.re_config = REGRESSOR_SCORERS
        experiment(path=self.path, model_file=self.model_name)

    def test_scorer(self):
        scorer = Scorer(config=self.config)
        scorer(
            ground_truth_file=self.ground_truth_file,
            predictions_file=self.predictions_file,
            path=self.path,
        )
        self.assertEqual(scorer.names, self.config.keys())
        self.assertEqual(type(scorer.scorers), type(self.config.values()))
        self.assertIsInstance(scorer, Scorer)

    def test_read_score_from_json(self):
        scorer = Scorer(config=self.config)
        scorer(
            ground_truth_file=self.ground_truth_file,
            predictions_file=self.predictions_file,
            path=self.path,
        )
        score = scorer.read_score_from_json(score_file=self.scores_file, name="ACC")
        self.assertIsInstance(score, float)

    def test_read_data_from_json(self):
        scorer = Scorer(config=self.config)
        predictions = scorer.read_data_from_json(self.predictions_file)
        ground_truth = scorer.read_data_from_json(self.ground_truth_file)
        self.assertEqual(predictions.shape, ground_truth.shape)
        self.assertEqual(predictions.shape[0], ground_truth.shape[0])
        self.assertEqual(predictions.shape[1], ground_truth.shape[1])

    def test_score(self):
        scorer = Scorer(config=self.config)
        predictions = scorer.read_data_from_json(self.predictions_file)
        ground_truth = scorer.read_data_from_json(self.ground_truth_file)
        score = scorer.score(ground_truth, predictions)
        self.assertIsInstance(scorer, Scorer)
        self.assertIsInstance(score[0], float)

    def test_call(self):
        logger.debug(self.ground_truth_file, self.predictions_file)
        logger.debug(os.path.dirname(os.path.realpath(__file__)))
        scorer = Scorer(config=self.config)
        scorer = scorer(self.ground_truth_file, self.predictions_file, self.path)
        self.assertIsInstance(scorer, Scorer)

    def test_save_results(self):
        scorer = Scorer(config=self.config)
        scorer = scorer(self.ground_truth_file, self.predictions_file, self.path)
        files = os.listdir(self.path)
        self.assertIn("scores.json", files)

    def test_default_classifier_config(self):
        scorer = Scorer(config=self.cl_config)
        self.assertEqual(scorer.names, self.cl_config.keys())
        self.assertEqual(type(scorer.scorers), type(self.cl_config.values()))
        scorer(
            ground_truth_file=self.ground_truth_file,
            predictions_file=self.predictions_file,
            path=self.path,
        )

    def test_default_regressor_config(self):
        scorer = Scorer(config=self.re_config)
        self.assertEqual(scorer.names, self.re_config.keys())
        self.assertEqual(type(scorer.scorers), type(self.re_config.values()))
        scorer(
            ground_truth_file=self.ground_truth_file,
            predictions_file=self.predictions_file,
            path=self.path,
        )

    def tearDown(self):
        if os.path.isfile("scores.json"):
            os.remove("scores.json")
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
