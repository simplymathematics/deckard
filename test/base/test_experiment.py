import unittest
import warnings
import yaml
from collections.abc import Callable  # noqa F401
from pathlib import Path

from art.attacks.evasion import BoundaryAttack  # noqa F401
from art.defences.postprocessor import HighConfidence  # noqa F401
from art.defences.preprocessor import FeatureSqueezing  # noqa F401
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # noqa F401

from sklearn.cluster import KMeans  # noqa F401
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelBinarizer
from deckard.base import Data, Experiment, Model, Yellowbrick_Visualiser
from deckard.base.experiment import config as exp_config
from deckard.base.model import config as model_config
from deckard.base.data import config as data_config
from deckard.base.hashable import my_hash

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

yaml.add_constructor("!Data:", Data)
yaml.add_constructor("!Model:", Model)
yaml.add_constructor("!Experiment:", Experiment)
yaml.add_constructor("!Yellowbrick_Visualiser:", Yellowbrick_Visualiser)


class testExperiment(unittest.TestCase):
    def setUp(
        self,
        exp_config=exp_config,
        data_config=data_config,
        model_config=model_config,
    ) -> None:

        self.path = "tmp/reports/"
        self.config = exp_config
        self.file = "test_filename"
        self.here = Path(__file__).parent
        exp_config = "!Experiment:" + exp_config
        data_config = "!Data:" + data_config
        model_config = "!Model:" + model_config
        self.exp = yaml.load(exp_config, Loader=yaml.FullLoader)
        self.data = yaml.load(data_config, Loader=yaml.FullLoader)
        self.model = yaml.load(model_config, Loader=yaml.FullLoader)

    def test_data(self):
        self.assertIsInstance(self.data, Data)

    def test_model(self):
        self.assertIsInstance(self.model, Model)

    def test_experiment(self):
        self.assertIsInstance(self.exp, Experiment)

    def test_hash(self):
        exp_config = "!Experiment:" + self.config
        exp = yaml.load(exp_config, Loader=yaml.FullLoader)
        self.assertEqual(my_hash(exp._asdict()), my_hash(self.exp._asdict()))

    def test_run(self):
        outputs = self.exp.run()
        self.assertTrue(Path(outputs['scores']).exists())

    def test_save_data(self):
        (
            data,
            _,
            _,
        ) = self.exp.load()
        data_dict = data.load("reports/filename.pickle")
        path = data.save(data_dict, "reports/filename.pickle")
        self.assertTrue(path.exists())

    def test_save_params(self):
        (
            _,
            _,
            _,
        ) = self.exp.load()
        path = Path(self.exp.save_params())
        self.assertTrue(path.exists())

    def test_save_model(self):
        (
            _,
            model,
            _,
        ) = self.exp.load()
        model = model.load("reports/filename.pickle")
        path = self.exp.save_model(model)
        self.assertTrue(Path(path).exists())

    def test_save_predictions(self):
        (
            data,
            model,
            _,
        ) = self.exp.load()
        model = model.load("reports/filename.pickle")
        data = data.load("reports/filename.pickle")
        data.y_train = LabelBinarizer().fit_transform(data.y_train)
        data.y_test = LabelBinarizer().fit_transform(data.y_test)
        model.fit(data.X_train, data.y_train)
        predictions = model.predict(data.X_test)
        path = self.exp.save_predictions(predictions)
        path = Path(path)
        self.assertTrue(path.exists())

    def test_save_ground_truth(self):
        (
            data,
            model,
            _,
        ) = self.exp.load()
        model = model.load("reports/filename.pickle")
        data = data.load("reports/filename.pickle")
        data.y_train = LabelBinarizer().fit_transform(data.y_train)
        data.y_test = LabelBinarizer().fit_transform(data.y_test)
        model.fit(data.X_train, data.y_train)
        truth = model.predict(data.X_test)
        path = self.exp.save_ground_truth(truth)
        path = Path(path)
        self.assertTrue(path.exists())

    def test_save_time_dict(self):
        (
            data,
            model,
            _,
        ) = self.exp.load()
        time_dict = {"fit_time": 0, "pred_time": 0}
        path = self.exp.save_time_dict(time_dict)
        path = Path(path)
        self.assertTrue(path.exists())

    def test_score(self):
        (
            data,
            model,
            _,
        ) = self.exp.load()
        model = model.load("reports/filename.pickle")
        data = data.load("reports/filename.pickle")
        data.y_train = LabelBinarizer().fit_transform(data.y_train)
        data.y_test = LabelBinarizer().fit_transform(data.y_test)
        model.fit(data.X_train, data.y_train)
        predictions = model.predict(data.X_test)
        score_dict = self.exp.score(predictions=predictions, ground_truth=data.y_test)
        path = self.exp.save_scores(score_dict)
        path = Path(path)
        self.assertTrue(path.exists())

    def tearDown(self) -> None:
        from shutil import rmtree

        if Path(self.here, "data").exists():
            rmtree(Path(self.here, "data"))
        if Path(self.here, "models").exists():
            rmtree(Path(self.here, "models"))
        if Path(self.here, self.path).exists():
            rmtree(Path(self.here, self.path))
        del self.path
        del self.file


if __name__ == "__main__":
    unittest.main()
