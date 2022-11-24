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
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import AlphaSelection
from yellowbrick.cluster import (
    KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
)
from yellowbrick.model_selection import (
    LearningCurve,
    ValidationCurve,
    CVScores,
    FeatureImportances,
    RFECV,
    DroppingCurve,
)
from yellowbrick.classifier import (
    ConfusionMatrix,
    ClassificationReport,
    ROCAUC,
)
from yellowbrick.contrib.wrapper import classifier, regressor, clusterer
from argparse import Namespace
import collections
from .hashable import BaseHashable, my_hash
from copy import deepcopy
import yaml
from .utils import factory
from data import Data
from model import Model
from experiment import Experiment
import logging
from sklearn.base import is_regressor, is_classifier

classification_visualisers = {
    "confusion": ConfusionMatrix,
    "classification": ClassificationReport,
    "roc_auc": ROCAUC,
}

regression_visualisers = {
    "error": PredictionError,
    "residuals": ResidualsPlot,
    "alphas": AlphaSelection,
}

clustering_visualisers = {
    "silhouette": SilhouetteVisualizer,
    "elbow": KElbowVisualizer,
    "intercluster": InterclusterDistance,
}
# elbow requires k
model_selection_visualisers = {
    "validation": ValidationCurve,
    "learning": LearningCurve,
    "cross_validation": CVScores,
    "feature_importances": FeatureImportances,
    "recursive": RFECV,
    "dropping_curve": DroppingCurve,
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
        field_names="data, model, plots, files, scorers, attack,",
        defaults=({},),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        """Generates a new Data object from a YAML node"""
        return super().__new__(cls, **loader.construct_mapping(node))


    def visualise_data(self, data: Namespace, path: str, prefix = None, samples:np.ndarray = None) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param data: dict of data to be used for visualisation
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        plots = deepcopy(self.plots)
        files = deepcopy(self.files)
        paths = []
        if samples is not None:
            data.X_train = samples
            data.y_train = data.y_train[:len(data.X_train)]
        y_train = data.y_train
        X_train = data.X_train
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        # Seeing if classes are specified
        classes = plots.pop("classes", None)
        # If not, set to the unique values of y_train
        classes = np.unique(y_train) if classes is None else classes
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
                plots['radviz'] = prefix + plots['radviz'] if prefix else plots['radviz']
                visualiser = RadViz(classes=classes)
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(path, plots["radviz"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["radviz"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("radviz")
                plt.gcf().clear()
            if "rank1d" in plots:
                plots['rank1d'] = prefix + plots['rank1d'] if prefix else plots['rank1d']
                visualiser = Rank1D(algorithm="shapiro")
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(path, plots["rank1d"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["rank1d"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("rank1d")
                plt.gcf().clear()
            if "rank2d" in plots:
                plots['rank2d'] = prefix + plots['rank2d'] if prefix else plots['rank2d']
                visualiser = Rank2D(algorithm="pearson")
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(path, plots["rank2d"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["rank2d"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("rank2d")
                plt.gcf().clear()
            if "balance" in plots:
                plots['balance'] = prefix + plots['balance'] if prefix else plots['balance']
                visualiser = ClassBalance(labels=classes)
                visualiser.fit(y_train)
                visualiser.show(Path(path, plots["balance"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["balance"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("balance")
                plt.gcf().clear()
            if "correlation" in plots:
                plots['correlation'] = prefix + plots['correlation'] if prefix else plots['correlation']
                visualiser = FeatureCorrelation(labels=features)
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(path, plots["correlation"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["correlation"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("correlation")
                plt.gcf().clear()
            if "pca" in plots:
                plots['pca'] = prefix + plots['pca'] if prefix else plots['pca']
                visualiser = PCA()
                visualiser.fit_transform(X_train, y_train)
                visualiser.show(Path(path, plots["pca"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["pca"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("pca")
                plt.gcf().clear()
            if "manifold" in plots:
                plots['manifold'] = prefix + plots['manifold'] if prefix else plots['manifold']
                visualiser = Manifold(manifold="tsne")
                visualiser.fit_transform(X_train, y_train)
                visualiser.show(Path(path, plots["manifold"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["manifold"]+ str(plots.pop("filetype", ".png"))))
            if "parallel" in plots:
                plots['parallel'] = prefix + plots['parallel'] if prefix else plots['parallel']
                visualiser = ParallelCoordinates(classes=classes, features=features)
                visualiser.fit(X_train, y_train)
                visualiser.show(Path(path, plots["parallel"]+ str(plots.pop("filetype", ".png"))))
                paths.append(Path(path, plots["parallel"]+ str(plots.pop("filetype", ".png"))))
                plots.pop("parallel")
                plt.gcf().clear()
        return paths

    def visualise_classification(
        self,
        data: Namespace,
        model: object,
        path: str,
        preds : np.ndarray = None,
        prefix: str = None,
    ) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        paths = []
        plots = deepcopy(self.plots)
        if "art" in str(type(model)):
            model = model.model
        yb_model = classifier(model)
        if preds is not None:
            data.y_test = preds
        for name in classification_visualisers:
            if name in plots:
                visualiser = classification_visualisers[name]
                plots[name] = prefix + plots[name] if prefix is not None else plots[name]
                if len(data.y_train.shape) > 1:
                    data.y_train = np.argmax(data.y_train, axis=1)
                if len(data.y_test.shape) > 1:
                    data.y_test = np.argmax(data.y_test, axis=1)
                if len(set(data.y_train)) > 2:
                    viz = visualiser(
                        yb_model,
                        classes=[int(y) for y in np.unique(data.y_train)],
                    )
                elif len(set(data.y_train)) == 2:
                    try:
                        viz = visualiser(
                            yb_model,
                            classes = [int(y) for y in np.unique(data.y_train)],
                            binary=True,
                        )
                    except TypeError as e:
                        logger.warning(
                            f"Failed due to error {e}. Trying without binary",
                        )
                        viz = visualiser(
                            yb_model,
                            classes = [int(y) for y in np.unique(data.y_train)],
                        )
                else:
                    viz = visualiser(
                        yb_model,
                        classes=[0],
                    )
                viz.fit(data.X_train, data.y_train)
                viz.score(data.X_test, data.y_test)
                _ = viz.show(outpath=Path(path, plots[name]+ str(plots.pop("filetype", ".png"))))
                assert Path(
                    path,
                    str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path,
                            str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise_regression(self, data: Namespace, model: object, path: str, preds:np.ndarray = None, prefix:str=None) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        paths = []
        plots = deepcopy(self.plots)
        if preds is not None:
            data.y_test = preds
        yb_model = regressor(model)
        for name in regression_visualisers.keys():
            if name in plots.keys():
                visualiser = regression_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                plots[name] = prefix + plots[name] if prefix is not None else plots[name]
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
                viz.fit(data.X_train, data.y_train)
                viz.score(data.X_test, data.y_test)
                _ = viz.show(outpath=Path(path, plots[name]+ str(plots.pop("filetype", ".png"))))
                assert Path(
                    path,
                    str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path,
                            str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise_clustering(self, data: Namespace, model: object, path: str, preds:np.ndarray = None, prefix:str = None) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        plots = deepcopy(self.plots)
        paths = []
        if preds is not None:
            data.y_test = preds
        yb_model = clusterer(model)
        for name in clustering_visualisers.keys():
            if name in plots.keys():
                visualiser = regression_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                plots[name] = prefix + plots[name] if prefix is not None else plots[name]
                if name == "elbow":
                    assert (
                        "k" in params
                    ), f"Elbow visualiser requires k parameter, specify by making k a parameter of the dictionary named {name} in the config file."
                viz = visualiser(yb_model, **params)
                viz.fit(data.X_train, data.y_train)
                viz.score(data.X_test, data.y_test)
                _ = viz.show(outpath=Path(path, plots[name]+ str(plots.pop("filetype", ".png"))))
                viz.show(outpath=Path(path, plots[name]))
                assert Path(
                    path,
                    str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path,
                            str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise_model_selection(
        self,
        data: Namespace,
        model: object,
        path: str,
        preds:np.ndarray = None,
        prefix: str = None, 
    ) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :return: list of paths to the generated plots
        """
        plots = deepcopy(self.plots)
        paths = []
        scorer = list(self.scorers.keys())[0] if self.scorers is not {} else None
        cv = plots.pop("cv", None)
        if preds is not None:
            data.y_test = preds
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
        if is_regressor(model):
            yb_model = regressor(model)
        elif is_classifier(model):
            yb_model = classifier(model)
        else:
            yb_model = clusterer(model)
        for name in model_selection_visualisers.keys():
            if name in plots.keys():
                plots[name] = prefix + plots[name] if prefix is not None else plots[name]
                visualiser = model_selection_visualisers[name]
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
                        cv=cv,
                        scoring=scorer,
                        **params,
                    )
                elif "dropping" or "feature_importances" in name:
                    viz = visualiser(
                        yb_model,
                    )
                elif "learning" in name:
                    viz = visualiser(
                        yb_model,
                        scoring=scorer,
                        **params,
                    )
                else:
                    raise ValueError("Unknown model selection visualiser.")
                viz.fit(data.X_train, data.y_train)
                viz.score(data.X_test, data.y_test)
                _ = viz.show(outpath=Path(path, plots[name]+ str(plots.pop("filetype", ".png"))))
                assert Path(
                    path,
                    str(plots[name]) + str(plots.pop("filetype", ".png")),
                ).is_file(), f"File {name} does not exist."
                paths.append(
                    str(
                        Path(
                            path,
                            str(plots[name]) + str(plots.pop("filetype", ".png")),
                        ),
                    ),
                )
                plt.gcf().clear()
        return paths

    def visualise(self, data, model, path: str, samples:np.ndarray = None, preds:np.ndarray = None, prefix:str = None) -> list:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        paths = []
        path.mkdir(parents=True, exist_ok=True)
        data_plots = self.visualise_data(data, path, samples = samples, prefix = prefix)
        model_plots = self.visualise_model_selection(data, model, path, preds, prefix)
        cla_plots = self.visualise_classification(data, model, path, preds, prefix)
        reg_plots = self.visualise_regression(data, model, path, preds, prefix)
        clu_plots = self.visualise_clustering(data, model, path, preds, prefix)
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

    # def render(
    #     self,
    #     plot_dict,
    #     template="template.html",
    #     templating_string=r"$plot_divs$",
    #     output_html="index.html",
    # ) -> Path:
    #     """
    #     Renders a list of paths to plots into a HTML file.
    #     :param plot_dict: dict of paths to plots
    #     :param template: path to template file
    #     :param templating_string: string to be replaced by the plots
    #     :return: path to the generated HTML file
    #     """
    #     new_plot_dict = {}
    #     template_file = Path(template)
    #     path = Path(output_html).parent
    #     path.mkdir(parents=True, exist_ok=True)
    #     with template_file.open("r") as f:
    #         template = f.read()
    #     for key in plot_dict.keys():
    #         new_key = f"<h2> {key.capitalize()} Plots </h2>"
    #         assert isinstance(
    #             plot_dict[key],
    #             list,
    #         ), f"Plot dictionary must be a list of paths to plots. Your config is {plot_dict}."
    #         for plot_file in plot_dict[key]:
    #             if not Path(plot_file).is_file():
    #                 plot_file = Path(str(plot_file) + ".png")
    #             assert Path(plot_file).is_file(), f"File {plot_file} does not exist."
    #             new_value = f"<img src {plot_file} alt {key} />"
    #             new_plot_dict[new_key] = new_value
    #     template = template.replace(templating_string, str(new_plot_dict))
    #     print(template)
    #     input("Press any key to continue...")
    #     assert (
    #         "path" in self.files
    #     ), f"Path to save the HTML file must be specified. Your config is {self.files}."
    #     with output_html.open("w") as f:
    #         f.write(template)
    #     return template_file.resolve()
