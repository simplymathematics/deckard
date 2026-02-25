

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Union, Literal


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold, ShuffleSplit
# yellow brick imports
# Feature Visualizers
from yellowbrick.features.rankd import Rank1D, Rank2D
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.jointplot import JointPlotVisualizer
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.manifold import Manifold
# Target Visualizers Imports
from yellowbrick.target import BalancedBinningReference, ClassBalance, FeatureCorrelation
# Regressor Visualizers
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import ManualAlphaSelection
# Classifier Visualizers
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassificationReport, ClassPredictionError, DiscriminationThreshold
# Clustering Visualizers
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
# Model Selection Visualizers
from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores, FeatureImportances, RFECV, DroppingCurve


from ..experiment import ExperimentConfig
from ..file import FileConfig
from ..data import DataConfig
from ..model import ModelConfig
from ..attack import AttackConfig
from ..utils import ConfigBase

feature_viz_types = [
    "rank1d",
    "rank2d",
    "radviz",
    "pcoords",
    "jointplot",
    "pca",
    "manifold"
]
target_viz_types = [
    "class_balance",
    "balanced_binning_reference",
    "feature_correlation"
]
regressor_viz_types = [
    "prediction_error",
    "residuals_plot",
    "alpha_selection"
]
classifier_viz_types = [
    "roc_auc",
    "precision_recall_curve",
    "classfication_report",
    "class_prediction_error",
    "discrimination_threshold"
]
cluster_viz_types = [
    "k_elbow",
    "silhouette",
    "intercluster_distance"
]
model_selection_viz_types = [
    "validation_curve",
    "learning_curve",
    "cv_scores",
    "feature_importances",
    "rfecv",
    "dropping_curve"
]

all_viz_types = feature_viz_types + target_viz_types + regressor_viz_types + classifier_viz_types + cluster_viz_types + model_selection_viz_types

all_viz_objects = [
    Rank1D,
    Rank2D,
    RadViz,
    ParallelCoordinates,
    JointPlotVisualizer,
    PCADecomposition,
    Manifold,
    ClassBalance,
    BalancedBinningReference,
    FeatureCorrelation,
    PredictionError,
    ResidualsPlot,
    ManualAlphaSelection,
    ROCAUC,
    PrecisionRecallCurve,
    ClassificationReport,
    ClassPredictionError,
    DiscriminationThreshold,
    KElbowVisualizer,
    SilhouetteVisualizer,
    InterclusterDistance,
    ValidationCurve,
    LearningCurve,
    CVScores,
    FeatureImportances,
    RFECV,
    DroppingCurve
]
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass

class YellowbrickPlotConfig(ConfigBase):
    """
YellowbrickPlotConfig is a configuration class for generating and saving various types of 
Yellowbrick visualizations. It supports feature, target, regressor, classifier, cluster, 
and model selection visualizations. The class provides methods to load data, configure 
visualizers, and generate plots based on the specified plot type.
Attributes
------------
    plot_type (Literal): The type of Yellowbrick plot to generate. Supported types include 
        "rank1d", "rank2d", "radviz", "pcoords", "jointplot", "pca", "manifold", 
        "class_balance", "balanced_binning_reference", "feature_correlation", 
        "prediction_error", "residuals_plot", "alpha_selection", "classfication_report", 
        "roc_auc", "precision_recall_curve", "class_prediction_error", 
        "discrimination_threshold", "k_elbow", "silhouette", "intercluster_distance", 
        "validation_curve", "learning_curve", "cv_scores", "feature_importances", 
        "rfecv", "dropping_curve".
    features (Union[List[str], Literal["all"]]): The features to include in the plot. 
        Defaults to "all".
    classes (Union[List[str], Literal["all"]]): The classes to include in the plot. 
        Defaults to "all".
    title (str): The title of the plot. Defaults to "Yellowbrick Plot".
    save_path (str): The file path to save the generated plot. Defaults to "yellowbrick_plot.png".
    plot_params (dict): Additional parameters to pass to the Yellowbrick visualizer.
Methods
--------
    load_data(data, test=False, attack=None):
        Loads the dataset based on the experiment configuration. Supports loading training, 
        test, or attack data.
    visualize_features(data: DataConfig, ax=None):
        Generates and saves feature-based Yellowbrick visualizations such as "rank1d", 
        "rank2d", "radviz", etc.
    visualize_targets(data: DataConfig, ax=None):
        Generates and saves target-based Yellowbrick visualizations such as "class_balance", 
        "balanced_binning_reference", and "feature_correlation".
    visualize_regressors(data: DataConfig, model: ModelConfig, ax=None):
        Generates and saves regressor-based Yellowbrick visualizations such as 
        "prediction_error", "residuals_plot", and "alpha_selection".
    visualize_classifiers(data: DataConfig, model: ModelConfig, ax):
        Generates and saves classifier-based Yellowbrick visualizations such as 
        "classification_report", "roc_auc", "precision_recall_curve", etc.
    visualize_clusters(data: DataConfig, model: ModelConfig, ax):
        Generates and saves cluster-based Yellowbrick visualizations such as "k_elbow", 
        "silhouette", and "intercluster_distance".
    visualize_model_selection(data: DataConfig, model: ModelConfig, ax=None):
        Generates and saves model selection-based Yellowbrick visualizations such as 
        "validation_curve", "learning_curve", "cv_scores", etc.
    parse_cv():
        Parses the cross-validation configuration from the plot parameters.
    parse_range():
        Parses the parameter range for validation curves or other range-based visualizations.
    visualize(experiment: ExperimentConfig, ax=None):
        Main method to generate and save the Yellowbrick plot. Determines the appropriate 
        visualization method based on the plot type.
    show(visualizer):
        Displays or saves the generated Yellowbrick visualizer plot.
    __call__(experiment):
        Invokes the visualize method to generate the plot for the given experiment.
        
Example
--------
    >>> cfg = YellowbrickPlotConfig(plot_type="pca", features=["feature1", "feature2"], classes=["class1", "class2"], 
    ...                             title="PCA Plot", save_path="pca_plot.png", plot_params={"scale": True})
    >>> experiment = ExperimentConfig(...)  # Assume this is properly defined
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> cfg.visualize(experiment=experiment, ax=ax, fig=fig)
"""
    plot_type: Literal[
        f"{all_viz_types}"
    ] = "pca"
    features: Union[List[str], Literal["all"]] = "all"
    classes: Union[List[str], Literal["all"]] = "all"
    title: str = "Yellowbrick Plot"
    save_path: str = "yellowbrick_plot.png"
    plot_params: dict = field(default_factory=dict)


    
    
    def load_data(self, data, test = False, attack:Union[AttackConfig, None]=None):
        """Loads the dataset based on the experiment configuration."""
        # Assert that either test or attack is False or both are False
        assert not (test and attack), "Cannot load both test and attack data simultaneously"
        X, y = data.X_train, data.y_train
        X_test, y_test = data.X_test, data.y_test
        if self.classes == "all":
            classes = np.unique(y)
        else:
            classes = self.classes
        if self.features == "all":
            features = X.columns.tolist()
        else:
            features = self.features
        if attack is not None:
            X_attack, y_attack = attack.attack, attack.labels
            return X_attack, y_attack, classes, features
        if not test:
            return X, y, classes, features
        else:
            return X_test, y_test, classes, features
    
    
    def visualize_features(self, data:DataConfig, ax=None):
        """Generates and saves the Yellowbrick data plot.
        
        Args
        -----
        data (DataConfig): The configuration object containing the dataset to be visualized.
        ax (_type_, optional): The matplotlib axis to plot on. Defaults to None.    
        
        Raises
        ------
        ValueError: If the specified plot type is not supported for feature visualization.
        
        """
        X, y, classes, features = self.load_data(data)
        if self.plot_type == "rank1d":
            visualizer = Rank1D(features=features, classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.transform(X)
        elif self.plot_type == "rank2d":
            visualizer = Rank2D(features=features, classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.transform(X)
        elif self.plot_type == "radviz":
            visualizer = RadViz(classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.transform(X)
        elif self.plot_type == "pcoords":
            visualizer = ParallelCoordinates(classes=classes, **self.plot_params, features=features)
            visualizer.fit_transform(X, y)
        elif self.plot_type == "jointplot":
            assert "columns" in self.plot_params, "Columns must be specified for jointplot"
            visualizer = JointPlotVisualizer(**self.plot_params, ax=ax)
            visualizer.fit_transform(X, y)
        elif self.plot_type == "pca":
            visualizer = PCADecomposition(classes=classes, **self.plot_params, ax=ax)
            visualizer.fit_transform(X, y)
        elif self.plot_type == "manifold":
            visualizer = Manifold(classes=classes, **self.plot_params, features=features, verbose=True, ax=ax)
            visualizer.fit_transform(X, y, verbose=True)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick plot saved to {self.save_path}")
    
    def visualize_targets(self, data:DataConfig, ax=None):
        """
        Visualizes statistics involving the data's training or testing labels.

        Args
        ------
        data (DataConfig): _description_
        ax (_type_, optional): _description_. Defaults to None.

        Raises
        --------
        ValueError: If the specified plot type is not supported for target visualization.
        """
        X, y, classes, feature_indices = self.load_data(data)
        if self.plot_type == "class_balance":
            visualizer = ClassBalance(labels=classes, **self.plot_params, ax=ax)
            visualizer.fit(y)
        elif self.plot_type == "balanced_binning_reference":
            visualizer = BalancedBinningReference(**self.plot_params, ax=ax)
            visualizer.fit(y)
        elif self.plot_type == "feature_correlation":
            visualizer = FeatureCorrelation(features=feature_indices, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick target plot saved to {self.save_path}")
    
    def visualize_regressors(self, data:DataConfig, model:ModelConfig, ax=None):
        X, y, _, _ = self.load_data(data)
        X_test, y_test, _, _ = self.load_data(data, test=True)
        if self.plot_type == "prediction_error":
            visualizer = PredictionError(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "residuals_plot":
            visualizer = ResidualsPlot(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "alpha_selection":
            visualizer = ManualAlphaSelection(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick regressor plot saved to {self.save_path}")
    
    def visualize_classifiers(self, data:DataConfig, model:ModelConfig, ax):
        X, y, classes, _ = self.load_data(data)
        X_test, y_test, _, _ = self.load_data(data, test=True)
        if self.plot_type == "classfication_report":
            visualizer = ClassificationReport(model, classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "roc_auc":
            visualizer = ROCAUC(model, classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "precision_recall_curve":
            visualizer = PrecisionRecallCurve(model, classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "class_prediction_error":
            visualizer = ClassPredictionError(model, classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "discrimination_threshold":
            visualizer = DiscriminationThreshold(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick classifier plot saved to {self.save_path}")
    
    def visualize_clusters(self, data:DataConfig, model:ModelConfig, ax):
        X, _, _, _ = self.load_data(data)
        if self.plot_type == "k_elbow":
            visualizer = KElbowVisualizer(model, **self.plot_params, ax=ax)
            visualizer.fit(X)
        elif self.plot_type == "silhouette":
            visualizer = SilhouetteVisualizer(model, **self.plot_params, ax=ax)
            visualizer.fit(X)
        elif self.plot_type == "intercluster_distance":
            visualizer = InterclusterDistance(model, **self.plot_params, ax=ax)
            visualizer.fit(X)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick cluster plot saved to {self.save_path}")
    
    
    def visualize_model_selection(self, data:DataConfig, model:ModelConfig, ax=None):
        """
        Visualizes various model selection plots using Yellowbrick visualizers.

        Parameters:
            data (DataConfig): The configuration object containing the dataset to be used.
            model (ModelConfig): The configuration object for the model to be visualized.
            ax (matplotlib.axes.Axes, optional): The matplotlib axes on which to draw the plot. 
                If None, a new figure and axes will be created.

        Raises:
            ValueError: If the specified `plot_type` is not supported.

        Supported Plot Types:
            - "validation_curve": Plots the validation curve for a model over a range of hyperparameter values.
            - "learning_curve": Plots the learning curve showing training and validation scores over varying training sizes.
            - "cv_scores": Visualizes cross-validation scores for the model.
            - "feature_importances": Displays the feature importances as determined by the model.
            - "rfecv": Performs recursive feature elimination with cross-validation and visualizes the results.
            - "dropping_curve": Visualizes the effect of dropping features on model performance.

        Notes:
            - The `plot_type` attribute of the class determines which plot is generated.
            - The `plot_params` attribute is used to pass additional parameters to the Yellowbrick visualizers.
            - The visualizer is displayed and saved to the path specified by `self.save_path`.

        Example:
            >>> cfg = YellowbrickPlotConfig(plot_type="validation_curve", plot_params={"cv": 5, "param_range": [1, 100]})
            >>> data_config = DataConfig(...)  # Assume this is properly defined
            >>> model_config = ModelConfig(...)  # Assume this is properly defined
            >>> cfg.visualize_model_selection(data_config, model_config, ax=ax)
        """
        X, y, _, features = self.load_data(data)
        cv = self.parse_cv()
        self.plot_params["cv"] = cv
        if self.plot_type == "validation_curve":
            param_range = self.parse_range()
            self.plot_params["param_range"] = param_range
            visualizer = ValidationCurve(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        elif self.plot_type == "learning_curve":
            sizes = self.parse_range()
            self.plot_params["train_sizes"] = sizes
            visualizer = LearningCurve(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        elif self.plot_type == "cv_scores":
            visualizer = CVScores(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        elif self.plot_type == "feature_importances":
            visualizer = FeatureImportances(model, **self.plot_params, labels=features, ax=ax)
            visualizer.fit(X, y)
        elif self.plot_type == "rfecv":
            visualizer = RFECV(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        elif self.plot_type == "dropping_curve":
            visualizer = DroppingCurve(model, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick model selection plot saved to {self.save_path}")

    def parse_cv(self):
        assert "cv" in self.plot_params, "CV must be specified for model selection plots"
        cv = self.plot_params.pop("cv")
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv)
        else:
            assert "name" in cv, "CV configuration must have a 'name' key"
            name = cv.pop("name")
            if name == "kfold":
                cv = KFold(**cv)
            elif name == "timeseries":
                cv = TimeSeriesSplit(**cv)
            elif name == "stratifiedkfold":
                cv = StratifiedKFold(**cv)
            elif name == "shufflesplit":
                cv = ShuffleSplit(**cv)
            else:
                raise ValueError(f"Unsupported CV type: {name}")
        return cv
    
    def parse_range(self):
        assert "param_range" in self.plot_params, "Param_range must be specified for validation_curve"
        param_range = self.plot_params.pop("param_range")
        num = self.plot_params.pop("num", 10)
        assert len(param_range) == 2 or len(param_range) == 3, "Param_range must be a list of 2 or 3 values"
        if len(param_range) == 2:
            param_range = np.linspace(param_range[0], param_range[1], num=num)
        elif len(param_range) == 3:
            if param_range[2] == "log":
                param_range = np.logspace(np.log10(param_range[0]), np.log10(param_range[1]), num=num)
            elif param_range[2] == "linear":
                param_range = np.linspace(param_range[0], param_range[1], num=num)
            elif isinstance(param_range[2], (int, float)):
                steps = (param_range[1] - param_range[0]) // param_range[2]
                param_range = np.linspace(start= param_range[0], stop =param_range[1], num=steps, dtype=type(param_range[2]))
            else:
                raise ValueError("Distribution must be either 'log' or 'linear'")
        return param_range
    
    def visualize(self, experiment:ExperimentConfig, ax=None):
        """Main method to generate and save the Yellowbrick plot."""
        if not hasattr(experiment, "score_dict") or len(experiment.score_dict) == 0:
            experiment()
        if hasattr(experiment, "data"):
            data = experiment.data
        else:
            raise ValueError("Experiment must have a data attribute")
        if hasattr(experiment, "model"):
            model = experiment.model.get_model()
        else:
            model = None
        if hasattr(experiment, "attack"):
            attack = experiment.attack
        else:
            attack = None
        # Validate that either ax is provided or otherwise create a new figure
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        if self.plot_type in feature_viz_types:
            self.visualize_features(data=data, ax=ax)
        elif self.plot_type in target_viz_types:
            self.visualize_targets(data=data, ax=ax)
        elif self.plot_type in regressor_viz_types:
            self.visualize_regressors(data=data, model=model, ax=ax)
        elif self.plot_type in classifier_viz_types:
            self.visualize_classifiers(data=data, model=model, ax=ax)
        elif self.plot_type in cluster_viz_types:
            self.visualize_clusters(data=data, model=model, ax=ax)
        elif self.plot_type in model_selection_viz_types:
            self.visualize_model_selection(data=data, model=model, ax=ax)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        if attack is not None:
            raise NotImplementedError("Attack visualization not implemented yet")
    
    def show(self, visualizer):
        assert hasattr(visualizer, "show"), "Visualizer does not have a show method"
        assert isinstance(visualizer, tuple(all_viz_objects)), "Visualizer is not a recognized Yellowbrick visualizer"
        visualizer.show(outpath=self.save_path)

    def __call__(self, experiment):
        self.visualize(experiment=experiment)

@dataclass
class YellowBrickConfigList(ConfigBase):
    """Configuration for a list of Yellowbrick plots."""

    plots : List[Literal[f"{all_viz_types}"]] = field(default_factory=list)
    files: Union[FileConfig, None] = None
    
    def __len__(self):
        return len(self.plots)
    
    def __call__(self, experiment=Union[ExperimentConfig, List[ExperimentConfig]], axes = None):
        plot_length = len(self)
        if isinstance(experiment, list):
            assert plot_length == len(experiment), "Number of plots must match number of experiments"
        else:
            experiment = [experiment] * plot_length
        
        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=plot_length, figsize=(8 * plot_length, 6))
        for plot, exp in zip(self.plots, experiment):
            plot_cfg = YellowbrickPlotConfig(plot_type=plot, experiment=exp)
            ax = axes if plot_length == 1 else axes[self.plots.index(plot)]
            try:
                plot_cfg.visualize(experiment=exp, ax=ax)
            except Exception as e:
                print(f"Failed to generate plot {plot}: {e}")
        if self.files is not None:
            plt.savefig(self.files.get_file_path())
            logger.info(f"Yellowbrick plots saved to {self.files.get_file_path()}")
        plt.close(fig)
        return fig