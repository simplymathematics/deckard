from dataclasses import dataclass, field
from hydra.utils import instantiate
from typing import Any, Union
import logging
from typing import Callable, List
from copy import deepcopy
from sklearn.base import is_regressor, is_classifier
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

from yellowbrick.contrib.prepredict import (
    PrePredict,
    CLASSIFIER,
    REGRESSOR,
    CLUSTERER,
    #    DensityEstimator,
)
from yellowbrick.target import ClassBalance, FeatureCorrelation
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import AlphaSelection
from yellowbrick.cluster import (
    KElbowVisualizer,
    SilhouetteVisualizer,
    InterclusterDistance,
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
from ..data import Data
from ..model import Model
from ..attack import Attack
from ..scorer import ScorerDict
from ..files import FileConfig


@dataclass
class PlotsConfig:
    balance: Union[str, None] = None
    classification: Union[str, None] = None
    confusion: Union[str, None] = None
    correlation: Union[str, None] = None
    radviz: Union[str, None] = None
    rank: Union[str, None] = None
    roc_auc: Union[str, None] = None
    error: Union[str, None] = None
    residuals: Union[str, None] = None
    alphas: Union[str, None] = None
    silhouette: Union[str, None] = None
    elbow: Union[str, None] = None
    intercluster: Union[str, None] = None
    validation: Union[str, None] = None
    learning: Union[str, None] = None
    cross_validation: Union[str, None] = None
    feature_importances: Union[str, None] = None
    recursive: Union[str, None] = None
    dropping_curve: Union[str, None] = None
    rank1d: Union[str, None] = None
    rank2d: Union[str, None] = None
    parallel: Union[str, None] = None
    manifold: Union[str, None] = None
    filetype: str = "pdf"


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


@dataclass
class Yellowbrick_Visualiser:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    attack: Attack = field(default_factory=Attack)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    files: FileConfig = field(default_factory=FileConfig)
    scorers: ScorerDict = field(default_factory=ScorerDict)

    def visualise_data(
        self,
        data: list,
        classes: list = None,
        features: list = None,
    ) -> List[Path]:
        """
        Visualise classification results according to the configuration file.
        :param data: dict of data to be used for visualisation
        :param self.files.path: path to save the plots
        :param self.plots: dict of plots to be generated
        :return: list of paths to the generated plots
        """
        plots = deepcopy(self.plots)
        files = deepcopy(self.files)
        path = Path(files["reports"])
        path.mkdir(parents=True, exist_ok=True)
        paths = []
        y_train = data[2]
        X_train = data[0]
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        # Seeing if classes are specified
        classes = plots.pop("classes", classes)
        # If not, set to the unique values of y_train
        classes = np.unique(y_train) if classes is None else classes
        # Seeing if features is specified
        features = plots.pop("features", features)
        # If not, set to the range of X_train.shape[1]
        features = list(range(X_train.shape[1])) if features is None else features
        paths = {}
        if len(plots.keys()) > 0:
            assert (
                "path" in files
            ), "Path to save plots is not specified in the files entry in params.yaml"
            Path(files["reports"]).mkdir(parents=True, exist_ok=True)
            if "radviz" in plots:
                visualiser = RadViz(classes=classes)
                visualiser.fit(X_train, y_train)
                visualiser.show(
                    Path(path, plots["radviz"] + str(plots.pop("filetype", ".png"))),
                )
                paths["radviz"] = str(
                    Path(
                        path,
                        plots["radviz"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "rank1d" in plots:
                visualiser = Rank1D(algorithm="shapiro")
                visualiser.fit(X_train, y_train)
                visualiser.show(
                    Path(path, plots["rank1d"] + str(plots.pop("filetype", ".png"))),
                )
                paths["rank1d"] = str(
                    Path(
                        path,
                        plots["rank1d"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "rank2d" in plots:
                visualiser = Rank2D(algorithm="pearson")
                visualiser.fit(X_train, y_train)
                visualiser.show(
                    Path(path, plots["rank2d"] + str(plots.pop("filetype", ".png"))),
                )
                paths["rank2d"] = str(
                    Path(
                        path,
                        plots["rank2d"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "balance" in plots:
                visualiser = ClassBalance(labels=classes)
                visualiser.fit(y_train)
                visualiser.show(
                    Path(path, plots["balance"] + str(plots.pop("filetype", ".png"))),
                )
                paths["balance"] = str(
                    Path(
                        path,
                        plots["balance"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "correlation" in plots:
                visualiser = FeatureCorrelation(labels=features)
                visualiser.fit(X_train, y_train)
                visualiser.show(
                    Path(
                        path,
                        plots["correlation"] + str(plots.pop("filetype", ".png")),
                    ),
                )
                paths["correlation"] = str(
                    Path(
                        path,
                        plots["correlation"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "pca" in plots:
                visualiser = PCA()
                visualiser.fit_transform(X_train, y_train)
                visualiser.show(
                    Path(path, plots["pca"] + str(plots.pop("filetype", ".png"))),
                )
                paths["pca"] = str(
                    Path(
                        path,
                        plots["pca"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "manifold" in plots:
                visualiser = Manifold(manifold="tsne")
                visualiser.fit_transform(X_train, y_train)
                visualiser.show(
                    Path(path, plots["manifold"] + str(plots.pop("filetype", ".png"))),
                )
                paths["manifold"] = str(
                    Path(
                        path,
                        plots["manifold"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
            if "parallel" in plots:
                visualiser = ParallelCoordinates(classes=classes, features=features)
                visualiser.fit(X_train, y_train)
                visualiser.show(
                    Path(path, plots["parallel"] + str(plots.pop("filetype", ".png"))),
                )
                paths["parallel"] = str(
                    Path(
                        path,
                        plots["parallel"] + str(plots.pop("filetype", ".png")),
                    ).as_posix(),
                )
                plt.gcf().clear()
        return paths

    def visualise_classification(
        self,
        data: list,
        model: Callable,
        classes: list = None,
        predictions_file=None,
    ) -> List[Path]:
        """
        Visualise classification results according to the configuration file.
        :param self.plots: dict of plots to be generated
        :param self.data: dict of data to be used for visualisation
        :param self.model: model object
        :param self.files.path: path to save the plots
        :param data: list of data to be used for visualisation
        :param model: model object to be used for visualisation
        :param predictions_file: path to predictions file
        :return: list of paths to the generated plots
        """
        plots = deepcopy(self.plots)
        files = deepcopy(self.files)
        path = Path(files["reports"])
        path.mkdir(parents=True, exist_ok=True)
        paths = {}
        # Seeing if classes are specified
        classes = plots.pop("classes", classes)
        # If not, set to the unique values of y_train
        classes = np.unique(data[2]) if classes is None else classes

        for name in classification_visualisers:
            if name in plots:
                if predictions_file is not None and Path(predictions_file).exists():
                    visualiser = PrePredict(predictions_file, CLASSIFIER)
                else:
                    visualiser = classification_visualisers[name]
                if len(data[2].shape) > 1:
                    data[2] = np.argmax(data[2], axis=1)
                if len(data[3].shape) > 1:
                    data[3] = np.argmax(data[3], axis=1)
                if len(set(np.unique(data[2]))) > 2:
                    viz = visualiser(
                        model,
                        classes=[int(y) for y in np.unique(data[2])],
                    )
                elif len(set(np.unique(data[2]))) == 2:
                    try:
                        viz = visualiser(
                            model,
                            classes=[int(y) for y in np.unique(data[2])],
                            binary=True,
                        )
                    except TypeError as e:
                        logger.warning(
                            f"Failed due to error {e}. Trying without binary",
                        )
                        viz = visualiser(
                            model,
                            classes=[int(y) for y in np.unique(data[2])],
                        )
                else:
                    viz = visualiser(model, classes=[0])
                viz.fit(X=data[0], y=data[2])
                viz.score(data[1], data[3])
                filename = Path(path, plots[name] + str(plots.pop("filetype", ".png")))
                _ = viz.show(outpath=filename)
                assert filename.exists(), f"File {filename} does not exist"
                paths[name] = str(filename.as_posix())
                plt.gcf().clear()
        return paths

    def visualise_regression(
        self,
        data: list,
        model: object,
        predictions_file=None,
    ) -> List[Path]:
        """
        Visualise classification results according to the configuration file.

        :param data: dict of data to be used for visualisation
        :param model: model object
        :param self.files.path: path to save the plots
        :param self.plots: dict of plots to be generated
        :param predictions_file: path to predictions file
        :return: list of paths to the generated plots
        """
        files = self.files()
        path = Path(files["reports"])
        path.mkdir(parents=True, exist_ok=True)
        paths = {}
        plots = deepcopy(self.plots)
        for name in regression_visualisers.keys():
            if name in plots.keys():
                if predictions_file is not None and Path(predictions_file).exists():
                    visualiser = PrePredict(predictions_file, REGRESSOR)
                else:
                    visualiser = regression_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                name = (
                    plots[name] if isinstance(plots[name], str) else params.pop("name")
                )
                try:
                    viz = visualiser(
                        model,
                        X_train=data[0],
                        y_train=data[2],
                        X_test=data[1],
                        y_test=data[3],
                        **params,
                    )
                except TypeError as e:
                    logger.warning(
                        f"Visualiser {name} failed with error {e}. Trying without test data.",
                    )
                    viz = visualiser(model, X_train=data[0], y_train=data[2], **params)
                viz.fit(data[0], data[2])
                viz.score(data[1], data[3])
                filename = Path(path, plots[name] + str(plots.pop("filetype", ".png")))
                _ = viz.show(outpath=filename)
                assert filename.is_file(), f"File {name} does not exist."
                paths[name] = str(filename.as_posix())
                plt.gcf().clear()
        return paths

    def visualise_clustering(
        self,
        data: list,
        model: object,
        predictions_file=None,
    ) -> List[Path]:
        """
        Visualise classification results according to the configuration file.
        :param data: dict of data to be used for visualisation
        :param model: model object
        :param self.plots: dict of plots to be generated
        :param self.files.path: path to save the plots
        :param predictions_file: path to predictions file
        :return: list of paths to the generated plots
        """
        files = self.files()
        path = Path(files["reports"])
        path.mkdir(parents=True, exist_ok=True)
        paths = {}
        plots = deepcopy(self.plots)
        for name in clustering_visualisers.keys():
            if name in plots.keys():
                if predictions_file is not None and Path(predictions_file).exists():
                    visualiser = PrePredict(predictions_file, CLUSTERER)
                else:
                    visualiser = clustering_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                name = (
                    plots[name] if isinstance(plots[name], str) else params.pop("name")
                )
                if name == "elbow":
                    assert (
                        "k" in params
                    ), f"Elbow visualiser requires k parameter, specify by making k a parameter of the dictionary named {name} in the config file."
                viz = visualiser(model, **params)
                viz.fit(data[0], data[2])
                # viz.score(data[1], data[3])
                filename = Path(path, name + str(plots.pop("filetype", ".png")))
                _ = viz.show(outpath=filename)
                viz.show(outpath=Path(path, name + str(plots.pop("filetype", ".png"))))
                paths[name] = str(filename.as_posix())
                plt.gcf().clear()
        return paths

    def visualise_model_selection(self, data: list, model: object) -> List[Path]:
        """
        Visualise classification results according to the configuration file.
        :param data: list of data to be used for visualisation
        :param model: model object
        :param self.plots: dict of plots to be generated
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        files = self.files()
        path = Path(files["reports"])
        path.mkdir(parents=True, exist_ok=True)
        paths = {}
        plots = deepcopy(self.plots)
        scorer = list(self.scorers.keys())[0] if self.scorers is not {} else None
        cv = plots.pop("cv", None)
        if scorer is None:
            if all([isinstance(item, int) for item in list(set(data[2]))]):
                scorer = "f1_weighted"
            else:
                scorer = "mse"
        if cv is None:
            cv = {"name": "sklearn.model_selection.StratifiedKFold", "n_splits": 5}
        assert (
            "name" in cv
        ), f"Cross validation method must be specified. Your config is {cv}."
        config = {"_target_": cv.pop("name"), "kwargs": cv}
        cv = instantiate(config)
        for name in model_selection_visualisers.keys():
            if name in plots.keys():
                visualiser = model_selection_visualisers[name]
                params = plots[name] if isinstance(plots[name], dict) else {}
                name = (
                    plots[name] if isinstance(plots[name], str) else params.pop("name")
                )
                if "cross" or "recursive" or "validation" in name:
                    if "validation" in name:
                        assert (
                            "param_name" in params
                        ), f"Validation curve visualiser requires param_name parameter. Parameter keys are {params.keys()}"
                        assert (
                            "param_range" in params
                        ), f"Validation curve visualiser requires params_range parameter. Parameter keys are {params.keys()}."
                    viz = visualiser(model, cv=cv, scoring=scorer, **params)
                elif "dropping" or "feature_importances" in name:
                    viz = visualiser(model)
                elif "learning" in name:
                    assert (
                        "train_sizes" in params
                    ), "Learning curve visualiser requires train_sizes parameter."
                    viz = visualiser(model, scoring=scorer, **params)
                else:
                    raise ValueError("Unknown model selection visualiser.")
                viz.fit(data[0], data[2])
                viz.score(data[1], data[3])
                filename = Path(path, str(name) + str(plots.pop("filetype", ".png")))
                _ = viz.show(outpath=str(filename))
                assert filename.is_file(), f"File {name} does not exist."
                paths[name] = str(filename.as_posix())
                f"File {name} does not exist."
                plt.gcf().clear()
        return paths

    def visualise(self, data, model, art=False) -> dict:
        """
        Visualise classification results according to the configuration file.
        :param data: list of data to be used for visualisation
        :param model: model object or predictions array.
        :param mtype: type of model, necessary if predictions are passed instead of an untrained model.
        :param self.plots: dict of plots to be generated
        :param self.files.path: path to save the plots
        :return: list of paths to the generated plots
        """
        if art is True and hasattr(model, "model"):
            model = model.model
        if is_regressor(model):
            model = regressor(model)
        elif is_classifier(model):
            model = classifier(model)
        else:
            model = clusterer(model)
        data_plots = self.visualise_data(data)
        model_plots = self.visualise_model_selection(data, model)
        cla_plots = self.visualise_classification(data, model)
        reg_plots = self.visualise_regression(data, model)
        clu_plots = self.visualise_clustering(data, model)
        plot_dict = {}
        if data_plots is not None:
            plot_dict["data"] = data_plots
        if model_plots is not None:
            plot_dict["model"] = model_plots
        if cla_plots is not None:
            plot_dict["classification"] = cla_plots
        if reg_plots is not None:
            plot_dict["regression"] = reg_plots
        if clu_plots is not None:
            plot_dict["clustering"] = clu_plots
        return plot_dict

    def __call__(self) -> Any:
        data = self.data()
        data, model = self.model.initialize(data)
        if str(type(model)).startwith("art."):
            art = True
        else:
            art = False
        plot_dict = self.visualise(data, model, art=art)
        return plot_dict
