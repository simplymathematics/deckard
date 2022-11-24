from yellowbrick.classifier import (
    classification_report,
    confusion_matrix,
    roc_auc,
)
from yellowbrick.contrib.wrapper import classifier

# from yellowbrick.exceptions import ModelError
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yellowbrick.features import (
    RadViz,
    Rank1D,
    Rank2D,
    PCA,
    Manifold,
    ParallelCoordinates,
)
from yellowbrick.target import ClassBalance, FeatureCorrelation
from yellowbrick.regressor import prediction_error, residuals_plot
from yellowbrick.regressor.alphas import alphas
from yellowbrick.features import (
    PCA,
    Manifold,
    ParallelCoordinates,
    RadViz,
    Rank1D,
    Rank2D,
)
from yellowbrick.target import ClassBalance, FeatureCorrelation
from yellowbrick.cluster import (
    kelbow_visualizer,
    silhouette_visualizer,
    intercluster_distance,
)
from yellowbrick.model_selection import (
    learning_curve,
    validation_curve,
    cross_validation,
    feature_importances,
    rfecv,
    dropping_curve,
)
from yellowbrick.classifier import (
    classification_report,
    confusion_matrix,
    roc_auc,
)
from yellowbrick.contrib.wrapper import classifier, regressor, clusterer, wrap
from sklearn.model_selection import StratifiedKFold
from argparse import Namespace
import collections
from hashable import BaseHashable, my_hash
from copy import deepcopy
import yaml
from utils import factory
from data import Data
from model import Model
import logging


classification_visualisers = {
    "confusion": confusion_matrix,
    "classification": classification_report,
    "roc_auc": roc_auc,
}

regression_visualisers = {
    "error": prediction_error,
    "residuals": residuals_plot,
    "alphas": alphas,
}

clustering_visualisers = {
    "silhouette": silhouette_visualizer,
    "elbow": kelbow_visualizer,
    "intercluster": intercluster_distance,
}
# elbow requires k
model_selection_visualisers = {
    "validation": validation_curve,
    "learning": learning_curve,
    "cross_validation": cross_validation,
    "feature_importances": feature_importances,
    "recursive": rfecv,
    "dropping_curve": dropping_curve,
}

data_visualisers = {
    "rank1d": Rank1D,
    "rank2d": Rank2D,
    "parallel": ParallelCoordinates,
    "radviz": RadViz,
    "manifold": Manifold,
    "balance": ClassBalance,
    "correlation": FeatureCorrelation,
}

supported_visualisers = [data_visualisers.keys()]
supported_visualisers.extend(classification_visualisers.keys())
supported_visualisers.extend(model_selection_visualisers.keys())
supported_visualisers.extend(clustering_visualisers.keys())
supported_visualisers.extend(regression_visualisers.keys())
supported_visualisers.extend(classification_visualisers.keys())

supported_visualisers_dict = {
    "data": data_visualisers,
    "model": model_selection_visualisers,
    "classification": classification_visualisers,
    "clustering": clustering_visualisers,
    "regression": regression_visualisers,
}

logger = logging.getLogger(__name__)


class Yellowbrick_Visualiser(
    collections.namedtuple(
        typename="YellowBrick_Visualiser",
        field_names="data, model, plots, files, scorers",
        defaults=({}),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        """Generates a new Data object from a YAML node"""
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, object), (data, model).
        """
        params = deepcopy(self._asdict())
        if params["data"] is not {}:
            yaml.add_constructor("!Data", Data)
            data_document = """!Data\n""" + str(dict(params["data"]))
            data = yaml.load(data_document, Loader=yaml.Loader)

        else:
            raise ValueError("Data not specified in config file")
        if params["model"] is not {}:
            yaml.add_constructor("!Model", Model)
            model_document = """!Model\n""" + str(dict(params["model"]))
            model = yaml.load(model_document, Loader=yaml.Loader)

        else:
            model = {}
        if params["plots"] is not {}:
            yaml.add_constructor("!Yellowbrick_Visualiser", Yellowbrick_Visualiser)
            plots_document = """!Yellowbrick_Visualiser\n""" + str(dict(params))
            vis = yaml.load(plots_document, Loader=yaml.Loader)
        else:
            vis = None
        params.pop("data", None)
        params.pop("model", None)
        params.pop("plots", None)
        files = params.pop("files", None)
        return (data, model, files, vis)

    def visualise_data(self, data: Namespace) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param data: dict of data to be used for visualisation
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        plots = dict(self.plots)
        files = dict(self.files)
        path = files["path"]
        paths = []
        y_train = data.y_train
        X_train = data.X_train
        # Seeing if classes are specified
        classes = plots.pop("classes", None)
        # If not, set to the unique values of y_train
        classes = set(y_train) if classes is None else classes
        # Seeing if features is specified
        features = plots.pop("features", None)
        # If not, set to the range of X_train.shape[1]
        features = list(range(X_train.shape[1])) if features is None else features
        paths = []
        if len(plots.keys()) > 0:
            assert (
                "path" in files
            ), "Path to save plots is not specified in the files entry in params.yaml"
            Path(files["path"]).mkdir(parents=True, exist_ok=True)
            if "radviz" in plots:
                visualiser = RadViz(classes=classes)
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(files["path"], plots["radviz"]))
                paths.append(Path(files["path"], plots["radviz"]))
                plots.pop("radviz")
                plt.gcf().clear()
            if "rank1d" in plots:
                visualiser = Rank1D(algorithm="shapiro")
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(files["path"], plots["rank1d"]))
                paths.append(Path(files["path"], plots["rank1d"]))
                plots.pop("rank1d")
                plt.gcf().clear()
            if "rank2d" in plots:
                visualiser = Rank2D(algorithm="pearson")
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(files["path"], plots["rank2d"]))
                paths.append(Path(files["path"], plots["rank2d"]))
                plots.pop("rank2d")
                plt.gcf().clear()
            if "balance" in plots:
                visualiser = ClassBalance(labels=classes)
                visualiser.fit(y_train)
                visualiser.show(Path(files["path"], plots["balance"]))
                paths.append(Path(files["path"], plots["balance"]))
                plots.pop("balance")
                plt.gcf().clear()
            if "correlation" in plots:
                visualiser = FeatureCorrelation(labels=features)
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(files["path"], plots["correlation"]))
                paths.append(Path(files["path"], plots["correlation"]))
                plots.pop("correlation")
                plt.gcf().clear()
            if "pca" in plots:
                visualiser = PCA()
                visualiser.fit_transform(X_train, y_train)
                visualiser.show(Path(files["path"], plots["pca"]))
                paths.append(Path(files["path"], plots["pca"]))
                plots.pop("pca")
                plt.gcf().clear()
            if "manifold" in plots:
                visualiser = Manifold(manifold="tsne")
                visualiser.fit_transform(X_train, y_train)
                visualiser.show(Path(files["path"], plots["manifold"]))
                paths.append(Path(files["path"], plots["manifold"]))
            if "parallel" in plots:
                visualiser = ParallelCoordinates(classes=classes, features=features)
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(files["path"], plots["parallel"]))
                paths.append(Path(files["path"], plots["parallel"]))
                plots.pop("parallel")
                plt.gcf().clear()
        return paths

    def visualise_classification(self, data: Namespace, model: object) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        paths = []
        plots = dict(self.plots)
        files = dict(self.files)
        path = files["path"]
        yb_model = classifier(model)
        for name in classification_visualisers.keys():
            if name in plots.keys():
                visualiser = classification_visualisers[name]
                if len(set(data.y_train)) > 2:
                    viz = visualiser(
                        yb_model,
                        X_train=data.X_train,
                        y_train=data.y_train,
                        classes=[int(y) for y in np.unique(data.y_train)],
                    )
                elif len(set(data.y_train)) == 2:
                    try:
                        viz = visualiser(
                            yb_model,
                            X_train=data.X_train,
                            y_train=data.y_train,
                            binary=True,
                        )
                    except TypeError as e:
                        logger.warning(
                            f"Failed due to error {e}. Trying without binary",
                        )
                        viz = visualiser(
                            yb_model,
                            X_train=data.X_train,
                            y_train=data.y_train,
                        )
                else:
                    viz = visualiser(
                        yb_model,
                        X_train=data.X_train,
                        y_train=data.y_train,
                        classes=[0],
                    )
                viz.show(outpath=Path(path, plots[name]))
                assert Path(
                    path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise_regression(self, data: Namespace, model: object) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        paths = []
        plots = dict(self.plots)
        files = dict(self.files)
        path = files["path"]
        yb_model = regressor(model)
        for name in regression_visualisers.keys():
            if name in plots.keys():
                visualiser = regression_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                try:
                    viz = visualiser(
                        yb_model,
                        X_train=data.X_train,
                        y_train=data.y_train,
                        X_test=data.X_test,
                        y_test=data.y_test,
                        **params,
                    )
                except TypeError as e:
                    logger.warning(
                        f"Visualiser {name} failed with error {e}. Trying without test data.",
                    )
                    viz = visualiser(
                        yb_model, X_train=data.X_train, y_train=data.y_train, **params
                    )
                viz.show(outpath=Path(path, plots[name]))
                assert Path(
                    path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise_clustering(self, data: Namespace, model: object) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        plots = dict(self.plots)
        files = files = dict(self.files)
        path = files["path"]
        paths = []
        yb_model = clusterer(model)
        for name in clustering_visualisers.keys():
            if name in plots.keys():
                visualiser = regression_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                if name == "elbow":
                    assert (
                        "k" in params
                    ), f"Elbow visualiser requires k parameter, specify by making k a parameter of the dictionary named {name} in the config file."
                viz = visualiser(yb_model, X_train=data.X_train, **params)
                viz.show(outpath=Path(path, plots[name]))
                assert Path(
                    path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise_model_selection(self, data: Namespace, model: object) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :return: list of paths to the generated plots
        """
        plots = dict(self.plots)
        files = dict(self.files)
        paths = []
        scorer = list(self.scorers.keys())[0] if self.scorers is not {} else None
        cv = plots.pop("cv", None)
        if scorer is None:
            if all([isinstance(item, int) for item in list(set(data.y_train))]):
                scorer = "f1_weighted"
            else:
                scorer = "mse"
        if cv is None:
            cv = {"name": "sklearn.model_selection.StratifiedKFold", "n_splits": 5}
        assert (
            "name" in cv
        ), f"Cross validation method must be specified. Your config is {cv}."
        cv = factory(
            cv.pop("name"),
        )
        for name in model_selection_visualisers.keys():
            if name in plots.keys():
                visualiser = model_selection_visualiser[key]
                params = plots[name] if isinstance(plots[name], dict) else {}
                if "cross" or "recursive" or "validation" in name:
                    if "validation" in name:
                        assert (
                            "param_name" in params
                        ), "Validation curve visualiser requires param_name parameter."
                        assert (
                            "params_range" in params
                        ), "Validation curve visualiser requires params_range parameter."
                    viz = visualiser(
                        yb_model,
                        X=data.X_train,
                        y=data.y_train,
                        cv=cv,
                        scoring=scorer,
                        **params,
                    )
                elif "dropping" or "feature_importances" in name:
                    viz = visualiser(
                        yb_model,
                        X=X_train,
                        y=y_train,
                    )
                elif "learning" in name:
                    viz = visualiser(
                        yb_model, X=X_train, y=y_train, scoring=scorer, **params
                    )
                else:
                    raise ValueError("Unknown model selection visualiser.")
                viz.show(outpath=Path(path, plots[name]))
                assert Path(
                    path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path, str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise(self) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        data, model, files, vis = self.load()
        data = data.load()
        model = model.load()
        paths = []
        data_plots = self.visualise_data(data)
        model_plots = self.visualise_model_selection(data, model)
        cla_plots = self.visualise_classification(data, model)
        reg_plots = self.visualise_regression(data, model)
        clu_plots = self.visualise_clustering(data, model)
        paths.extend(data_plots)
        paths.extend(model_plots)
        paths.extend(reg_plots)
        paths.extend(cla_plots)
        paths.extend(clu_plots)
        for path in paths:
            assert (
                Path(path).is_file() or Path(str(path) + ".png").is_file()
            ), f"File {path} does not exist."
        return {
            "data": data_plots,
            "model": model_plots,
            "classification": cla_plots,
            "regression": reg_plots,
            "clustering": clu_plots,
        }

    def render(
        self,
        plot_dict,
        template="template.html",
        templating_string="{{data_plots}}",
        output_html="index.html",
    ) -> Path:
        """
        Renders a list of paths to plots into a HTML file.
        :param plot_dict: dict of paths to plots
        :param template: path to template file
        :param templating_string: string to be replaced by the plots
        :return: path to the generated HTML file
        """
        new_plot_dict = {}
        template_file = Path(template)
        with template_file.open("r") as f:
            template = f.read()
        for key in plot_dict():
            new_key = f"<h2> {key.capitalize()} Plots </h2>"
            assert isinstance(
                plot_dict[key], list,
            ), f"Plot dictionary must be a list of paths to plots. Your config is {plot_dict}."
            for plot_file in plot_dict[key]:
                assert Path(
                    plot_file,
                ).exists(), f"Unable to render. {plot_file} does not exist."
                new_value = f"<img src {plot_file} alt {key} />"
                new_plot_dict[new_key] = new_value
        template = template.replace(templating_string, str(new_plot_dict["data"]))
        assert (
            "path" in self.files
        ), f"Path to save the HTML file must be specified. Your config is {self.files}."
        output_file = Path(self.files["path"], my_hash(self._asdict()), output_html)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            f.write(template)
        return template_file.resolve()
