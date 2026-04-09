

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Union, Literal
from hydra.core.hydra_config import HydraConfig

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
    """Configuration for Yellowbrick data plots."""

    plot_type: Literal[
        f"{all_viz_types}"
    ] = "pca"
    features: Union[List[str], Literal["all"]] = "all"
    classes: Union[List[str], Literal["all"]] = "all"
    title: str = "Yellowbrick Plot"
    save_path: str = "yellowbrick_plot.png"
    experiment: ExperimentConfig | None = field(default_factory=ExperimentConfig)
    plot_params: dict = field(default_factory=dict)
            

    def initialize_experiment(self):
        """Initializes the experiment configuration."""
        if hasattr(self.experiment.data, "X_train") and self.experiment.data.X_train is not None:
            return
        self.experiment()
    
    def load_data(self, test = False, attack=False):
        """Loads the dataset based on the experiment configuration."""
        # Assert that either test or attack is False or both are False
        assert not (test and attack), "Cannot load both test and attack data simultaneously"
        X, y = self.experiment.data.X_train, self.experiment.data.y_train
        X_test, y_test = self.experiment.data.X_test, self.experiment.data.y_test
        if self.classes == "all":
            classes = np.unique(y)
        else:
            classes = self.classes
        if self.features == "all":
            features = X.columns.tolist()
        else:
            features = self.features
        if attack:
            X_attack, y_attack = self.experiment.attack.attack, self.experiment.data.y_train[:self.experiment.attack.attack_size]
            return X_attack, y_attack, classes, features
        if not test:
            return X, y, classes, features
        else:
            return X_test, y_test, classes, features
    
    def load_model(self):
        """Loads the model based on the experiment configuration."""
        model = self.experiment.model
        return model.get_model()
    
    def visualize_features(self, ax=None):
        """Generates and saves the Yellowbrick data plot."""
        X, y, classes, features = self.load_data()
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
    
    def visualize_targets(self, ax=None):
        X, y, classes, feature_indices = self.load_data()
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
    
    def visualize_regressors(self, ax=None):
        X, y, _, _ = self.load_data()
        X_test, y_test, _, _ = self.load_data(test=True)
        model = self.load_model()
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
    
    def visualize_classifiers(self, ax):
        X, y, classes, _ = self.load_data()
        X_test, y_test, _, _ = self.load_data(test=True)
        model = self.load_model()
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
    
    def visualize_clusters(self, ax):
        X, _, _, _ = self.load_data()
        model = self.load_model()
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
    
    def visualize_model_selection(self, ax=None):
        X, y, _, features = self.load_data()
        model = self.load_model()
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
    
    def visualize(self, ax=None):
        """Main method to generate and save the Yellowbrick plot."""
        self.initialize_experiment()
        # Validate that either ax is provided or otherwise create a new figure
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        if self.plot_type in feature_viz_types:
            self.visualize_features(ax)
        elif self.plot_type in target_viz_types:
            self.visualize_targets(ax)
        elif self.plot_type in regressor_viz_types:
            self.visualize_regressors(ax)
        elif self.plot_type in classifier_viz_types:
            self.visualize_classifiers(ax)
        elif self.plot_type in cluster_viz_types:
            self.visualize_clusters(ax)
        elif self.plot_type in ["validation_curve", "learning_curve", "cv_scores", "feature_importances", "rfecv", "dropping_curve"]:
            self.visualize_model_selection(ax)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
    
    def show(self, visualizer):
        assert hasattr(visualizer, "show"), "Visualizer does not have a show method"
        assert isinstance(visualizer, tuple(all_viz_objects)), "Visualizer is not a recognized Yellowbrick visualizer"

        visualizer.show(outpath=self.save_path)
        
    def __len__(self):
        return 1
    
    def __call__(self):
        self.initialize_experiment()
        self.visualize()

@dataclass
class YellowBrickConfigList(ConfigBase):
    """Configuration for a list of Yellowbrick plots."""

    plots : List[Literal[f"{all_viz_types}"]] = field(default_factory=list)
    experiment: Union[ExperimentConfig, None] = None
    experiments: List[ExperimentConfig] = field(default_factory=list)
    file: Union[str, None] = None
    
    def __post_init__(self):
        if len(self.experiments) == 0:
            assert isinstance(self.experiment, ExperimentConfig), "Either a single experiment or a list of experiments must be provided"
            self.experiments = [self.experiment for _ in self.plots]
        else:
            assert len(self.plots) == len(self.experiments), "Number of plots must match number of experiments"
            assert self.experiment is None, "Either a single experiment or a list of experiments must be provided, not both"
        assert len(self.plots) == len(self.experiments), "Number of plots must match number of experiments"    
    
    def __iter__(self):
        for plot_type, experiment in zip(self.plots, self.experiments):
            plot_cfg = YellowbrickPlotConfig(
                plot_type=plot_type,
                features="all",
                classes="all",
                title=plot_type.replace("_", " ").title(),
                save_path=f"plots/"+plot_type+".png",
                experiment=experiment,
            )
            yield plot_cfg
    
    def __len__(self):
        return len(self.plots)
    
    
    
    def __call__(self):
        plot_length = len(self)
        fig, axes = plt.subplots(nrows=plot_length, ncols=1, figsize=(10, 8*plot_length))
        for plot_cfg in self:
            ax = axes[plot_cfg.plot_type] if plot_length > 1 else axes
            try:
                plot_cfg()
            except Exception as e:
                print(f"Failed to generate plot {plot_cfg.plot_type}")
        if self.file is not None:
            plt.savefig(self.file)
            logger.info(f"Yellowbrick plots saved to {self.file}")
        plt.close(fig)
        return fig