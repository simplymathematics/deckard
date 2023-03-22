import unittest
import warnings
import yaml
from collections.abc import Callable  # noqa F401
from pathlib import Path
from argparse import Namespace
import numpy as np
from art.attacks.evasion import BoundaryAttack  # noqa F401
from art.defences.postprocessor import HighConfidence  # noqa F401
from art.defences.preprocessor import FeatureSqueezing  # noqa F401
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # noqa F401
from art.utils import to_categorical
from yellowbrick.contrib.wrapper import classifier, regressor  # , clusterer

from sklearn.cluster import KMeans  # noqa F401
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression  # noqa F401
from deckard.base import Data, Experiment, Model, Yellowbrick_Visualiser
from deckard.base.experiment import config as exp_config

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
    ) -> None:

        self.path = "reports"
        self.config = exp_config
        data, model, _ = yaml.load(
            "!Experiment:\n" + str(self.config),
            Loader=yaml.FullLoader,
        ).load()
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.data = data.load("reports/filename.pickle")
        self.model = model.load("reports/model.pickle", art=False)
        self.data.y_train = to_categorical(self.data.y_train)
        self.data.y_test = to_categorical(self.data.y_test)
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
                "param_name": "penalty",
                "param_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "learning": {
                "name": "learning",
                "train_sizes": [10, 20, 30, 40, 50],
            },
            "cross_validation": {
                "name": "cross_validation",
                "param_name": "penalty",
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
        data = data.load("reports/filename.pickle")
        model = model.load("reports/model.pickle")
        y_train = 1 * np.random.rand(len(data.y_train))
        model.model.fit(self.data.X_train, y_train)
        paths = viz.visualise_regression(data, regressor(self.model.model))
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    # def test_visualise_clustering(self):
    #     import os
    #     os.environ['OMP_NUM_THREADS'] = "1"
    #     params = yaml.load(self.config, Loader=yaml.FullLoader)
    #     params["plots"] = self.clustering_viz
    #     params["model"]["init"] = {"name": "sklearn.cluster.KMeans", "n_clusters": 1}
    #     viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
    #     exp = yaml.load(self.exp + str(params), Loader=yaml.FullLoader)
    #     data, model, _ = exp.load()
    #     data = data.load(f"{self.path}/tmp_data.pickle")
    #     model = model.load(f"{self.path}/new_model.pickle")
    #     y_train = np.argmax(data.y_train, axis=1)
    #     model, _ = model.fit(data.X_train, y_train)
    #     paths = viz.visualise_clustering(data, clusterer(model.model))
    #     for path in paths:
    #         filename = Path(paths[path])
    #         self.assertTrue(Path(filename).exists())

    def test_visualise_model_selection(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        params["plots"] = self.selection_viz
        params["model"]["init"] = {"name": "sklearn.linear_model.LogisticRegression"}
        params["model"]["fit"] = {}
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        model = LogisticRegression()
        X, y = load_iris(return_X_y=True)
        model.fit(X, y)
        data = Namespace(X_train=X, y_train=y, X_test=X, y_test=y)
        paths = viz.visualise_model_selection(data, model)
        for path in paths:
            filename = Path(paths[path])
            self.assertTrue(Path(filename).exists())

    def test_visualise(self):
        params = yaml.load(self.config, Loader=yaml.FullLoader)
        viz = yaml.load(self.tag + str(params), Loader=yaml.FullLoader)
        exp = yaml.load(self.exp + str(params), Loader=yaml.FullLoader)
        data, model, _ = exp.load()
        data = data.load("reports/filename.pickle")
        model = model.load("reports/model.pickle")
        model.model.fit(data.X_train, data.y_train)
        paths = viz.visualise(self.data, classifier(model.model))
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
