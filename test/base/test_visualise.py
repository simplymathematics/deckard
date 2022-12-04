import unittest
import warnings
import yaml
from collections.abc import Callable  # noqa F401
from pathlib import Path

import numpy as np
from art.attacks.evasion import BoundaryAttack  # noqa F401
from art.defences.postprocessor import HighConfidence  # noqa F401
from art.defences.preprocessor import FeatureSqueezing  # noqa F401
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # noqa F401
from yellowbrick.contrib.wrapper import classifier, regressor, clusterer

from sklearn.cluster import KMeans  # noqa F401
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelBinarizer
from deckard.base import Data, Experiment, Model, Yellowbrick_Visualiser
from deckard.base.experiment import config as exp_config
from deckard.base.model import config as model_config
from deckard.base.data import config as data_config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

yaml.add_constructor("!Data:", Data)
yaml.add_constructor("!Model:", Model)
yaml.add_constructor("!Experiment:", Experiment)
yaml.add_constructor("!Yellowbrick_Visualiser:", Yellowbrick_Visualiser)


class testVisualiser(unittest.TestCase):
    def setUp(
        self,
        exp_config=exp_config,
        data_config=data_config,
        model_config=model_config,
    ) -> None:

        self.path = "reports"
        self.config = exp_config
        data, model, _ = yaml.load(
            "!Experiment:\n" + str(self.config),
            Loader=yaml.FullLoader,
        ).load()
        self.data = data.load()
        self.model = model.load(art=False)
        self.data.y_train = (
            LabelBinarizer().fit(self.data.y_train).transform(self.data.y_train)
        )
        self.data.y_test = (
            LabelBinarizer().fit(self.data.y_train).transform(self.data.y_test)
        )
        self.here = Path(__file__).parent
        self.data_viz = {
            "rank1d": "rank1d",
            "rank2d": "rank2d",
            "parallel": "parallel",
            "radviz": "radviz",
            "manifold": "manifold",
            "balance": "correlation",
        }
        self.class_viz = {
            "confusion": "confusion",
            "classification": "classification",
            "roc_auc": "roc_auc",
        }
        self.reg_viz = {
            "error": "error",
            "residuals": "residuals",
            # "alphas" : "alphas", # Not supported for 3rd party libs
        }
        self.clustering_viz = {
            "silhouette": "silhouette",
            "elbow": {"name": "elbow", "k": 3},
            "intercluster": "intercluster",
        }
        self.selection_viz = {
            "validation": {
                "name": "validation",
                "param_name": "alpha",
                "param_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "learning": {
                "name": "learning",
                "train_sizes": [10, 50, 100, 200, 500, 640],
            },
            "cross_validation": {
                "name": "cross_validation",
                "param_name": "alpha",
                "param_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "feature_importances": "feature_importances",
            "recursive": "recursive",
            "feature_dropping": "dropping_curve",
        }
        self.tag = "!Yellowbrick_Visualiser:\n"
        self.exp = "!Experiment:\n"

    def test_visualise_data(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        params["plots"] = self.data_viz
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        paths = viz.visualise_data(self.data)
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    def test_visualise_classification(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        params["plots"] = self.class_viz
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        self.model.fit(self.data.X_train, self.data.y_train)
        paths = viz.visualise_classification(self.data, classifier(self.model.model))
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    def test_visualise_regression(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        params["plots"] = self.reg_viz
        params["model"]["init"] = {
            "name": "sklearn.linear_model.LinearRegression",
            "fit_intercept": True,
        }
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        exp = yaml.load(self.exp + str(params), Loader=yaml.FullLoader)
        data, model, _ = exp.load()
        data = data.load()
        model = model.load()
        model.fit(data.X_train, data.y_train)
        paths = viz.visualise_regression(data, regressor(model.model))
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    def test_visualise_clustering(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        params["plots"] = self.clustering_viz
        params["model"]["init"] = {"name": "sklearn.cluster.KMeans", "n_clusters": 3}
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        exp = yaml.load(self.exp + str(params), Loader=yaml.FullLoader)
        data, model, _ = exp.load()
        data = data.load()
        model = model.load(art=True)
        model = model.model.fit(data.X_train, data.y_train).steps[-1][1]
        paths = viz.visualise_clustering(data, clusterer(model))
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    def test_visualise_model_selection(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        params["plots"] = self.selection_viz
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        self.data.y_train = np.argmax(self.data.y_train, axis=1)
        self.data.y_test = np.argmax(self.data.y_test, axis=1)
        model = self.model.model.fit(self.data.X_train, self.data.y_train).steps[-1][1]
        paths = viz.visualise_model_selection(self.data, model)
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    def test_visualise(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        self.data.y_train = np.argmax(self.data.y_train, axis=1)
        self.data.y_test = np.argmax(self.data.y_test, axis=1)
        model = self.model.model.fit(self.data.X_train, self.data.y_train).steps[-1][1]
        paths = viz.visualise(self.data, classifier(model))
        for path in paths:
            for subpath in paths[path]:
                filename = Path(paths[path][subpath])
                self.assertTrue(Path(filename).exists())

    def tearDown(self) -> None:
        from shutil import rmtree

        if Path(self.here, "data").exists():
            rmtree(Path(self.here, "data"))
        if Path(self.here, "models").exists():
            rmtree(Path(self.here, "models"))
        if Path(self.here, self.path).exists():
            rmtree(Path(self.here, self.path))
        del self.path


if __name__ == "__main__":
    unittest.main()
