from importlib.resources import Resource
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
import unittest, logging, tempfile
from deckard.base import Data, Experiment, Model
from deckard.base.utils import SUPPORTED_MODELS, find_successes, remove_successes_from_queue
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from art.estimators.classification import PyTorchClassifier, KerasClassifier, TensorFlowClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnRandomForestClassifier

logger = logging.getLogger(__name__)
class testUtils(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.file = 'test_model'
        self.data = Data('iris')
        self.model = Model(RandomForestClassifier(), 'sklearn')
        self.model2 = Model(DecisionTreeClassifier(), 'sklearn')
        self.model3 = Model(SVC(), 'sklearn')

        self.experiment = Experiment(self.data, self.model)
        self.experiment2 = Experiment(self.data, self.model2)
        self.experiment3 = Experiment(self.data, self.model3)
        self.experiment.run(self.path)
        self.experiment.save_model(filename = 'model', path = self.path)
        self.experiment.save_data(filename = 'data.pkl', path = self.path)
        self.list = [(self.experiment.name, self.experiment.params), (self.experiment2.name, self.experiment2.params)]
    
    def test_find_successes(self):
        self.experiment = Experiment(self.data, self.model)
        self.experiment.run(self.path)
        self.experiment.save_params(path = self.path)
        self.experiment.save_model(filename = 'model.pickle', path = self.path)
        successes, failures = find_successes(self.path, 'model_params.json')
        self.assertIsInstance(successes, list)
        self.assertEqual(len(failures), 0)
    
    def test_remove_successes_from_queue(self):
        self.experiment = Experiment(self.data, self.model)
        self.experiment.run(self.path)
        self.experiment.save_params(path = self.path)
        self.experiment.save_model(filename = 'model.pickle', path = self.path)
        successes, failures = find_successes(self.path, 'model_params.json')
        remove_successes_from_queue(successes, self.list)
        self.assertEqual(len(self.list), 2)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.path)
        
