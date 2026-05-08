"""Yellowbrick plotting configs built on top of prepared experiment state.

Plot configs compose an ExperimentConfig to prepare experiment artifacts once,
then reuse that prepared state across multiple plot renders.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    StratifiedKFold,
    ShuffleSplit,
)

# yellow brick imports
# Feature Visualizers
from yellowbrick.features.rankd import Rank1D, Rank2D
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.jointplot import JointPlotVisualizer
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.manifold import Manifold

# Target Visualizers Imports
from yellowbrick.target import (
    BalancedBinningReference,
    ClassBalance,
    FeatureCorrelation,
)

# Regressor Visualizers
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import ManualAlphaSelection

# Classifier Visualizers
from yellowbrick.classifier import (
    ROCAUC,
    PrecisionRecallCurve,
    ClassificationReport,
    ClassPredictionError,
    DiscriminationThreshold,
)

# Clustering Visualizers
from yellowbrick.cluster import (
    KElbowVisualizer,
    SilhouetteVisualizer,
    InterclusterDistance,
)

# Model Selection Visualizers
from yellowbrick.model_selection import (
    ValidationCurve,
    LearningCurve,
    CVScores,
    FeatureImportances,
    RFECV,
    DroppingCurve,
)


from ..utils import ConfigBase
from ..experiment import ExperimentConfig

feature_viz_types = [
    "rank1d",
    "rank2d",
    "radviz",
    "pcoords",
    "jointplot",
    "pca",
    "manifold",
]
target_viz_types = [
    "class_balance",
    "balanced_binning_reference",
    "feature_correlation",
]
regressor_viz_types = ["prediction_error", "residuals_plot", "alpha_selection"]
classifier_viz_types = [
    "roc_auc",
    "precision_recall_curve",
    "classification_report",
    "class_prediction_error",
    "discrimination_threshold",
]
cluster_viz_types = ["k_elbow", "silhouette", "intercluster_distance"]
model_selection_viz_types = [
    "validation_curve",
    "learning_curve",
    "cv_scores",
    "feature_importances",
    "rfecv",
    "dropping_curve",
]

all_viz_types = (
    feature_viz_types
    + target_viz_types
    + regressor_viz_types
    + classifier_viz_types
    + cluster_viz_types
    + model_selection_viz_types
)

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
    DroppingCurve,
]
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class _YellowbrickModelAdapter(BaseEstimator, ClassifierMixin):
    """Expose Deckard model configs with a sklearn-like inference interface.

    Yellowbrick classifier visualizers expect estimators that implement sklearn
    conventions (``fit``, ``score``, and ``predict_proba``/``decision_function``).
    PyTorch-backed Deckard models are not sklearn estimators, so this adapter
    bridges that API using the model config's existing prediction path.
    """

    def __init__(self, model_config: Any, classifier: bool = True):
        self.model_config = model_config
        self._estimator_type = "classifier" if classifier else "regressor"
        self.classes_ = None

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.cpu().numpy()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def fit(self, X, y=None, **kwargs):
        # Experiment preparation performs model fitting already.
        if y is not None and self._estimator_type == "classifier":
            y_arr = self._to_numpy(y).reshape(-1)
            self.classes_ = np.unique(y_arr)
        return self

    def _predict_raw(self, X) -> np.ndarray:
        raw = self.model_config._predict(X)
        return self._to_numpy(raw)

    def predict_proba(self, X) -> np.ndarray:
        raw = self._predict_raw(X)
        if raw.ndim == 1:
            logits = raw.astype(float)
            probs_pos = 1.0 / (1.0 + np.exp(-logits))
            return np.vstack([1.0 - probs_pos, probs_pos]).T
        if raw.ndim == 2:
            row_sums = raw.sum(axis=1)
            if np.all(raw >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-4):
                return raw
            shifted = raw - np.max(raw, axis=1, keepdims=True)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        raise ValueError(
            f"Unsupported prediction shape for predict_proba: {raw.shape}",
        )

    def predict(self, X) -> np.ndarray:
        raw = self._predict_raw(X)
        if raw.ndim == 1:
            return (raw > 0.0).astype(int)
        if raw.ndim == 2:
            return np.argmax(raw, axis=1)
        raise ValueError(f"Unsupported prediction shape for predict: {raw.shape}")

    def score(self, X, y) -> float:
        y_true = self._to_numpy(y).reshape(-1)
        y_pred = self.predict(X).reshape(-1)
        if y_true.shape[0] == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))


def _named_classifier_adapter(model_config: Any) -> _YellowbrickModelAdapter:
    """Return a classifier adapter whose class name mirrors the wrapped model.

    Yellowbrick derives default titles from ``estimator.__class__.__name__``.
    Preserve the wrapped model name so titles stay user-facing.
    """
    model_name = "Model"
    try:
        model_obj = model_config.get_model()
        model_name = type(model_obj).__name__ or model_name
    except Exception:
        pass

    adapter_type = type(
        model_name,
        (_YellowbrickModelAdapter,),
        {},
    )
    return adapter_type(model_config=model_config, classifier=True)


@dataclass(kw_only=True)
class YellowbrickPlotConfig(ConfigBase):
    """Render a single Yellowbrick plot from composed experiment configuration.

    This config composes an ExperimentConfig and prepares missing experiment
    outputs on first call. Later calls reuse the prepared state and only render
    the plot again.
    """

    experiment: ExperimentConfig

    # Plot-specific parameters
    plot_type: str
    clustering: bool = False
    features: Union[List[str], Literal["all"]] = "all"
    classes: Union[List[str], Literal["all"]] = "all"
    title: str = "Yellowbrick Plot"
    save_path: str = "yellowbrick_plot.png"
    rc_config: Dict[str, Any] = field(default_factory=dict)
    plot_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._experiment_prepared = False
        self._experiment_scores = {}
        if not hasattr(self, "score_dict"):
            self.score_dict = {}
        if not hasattr(self, "rc_config"):
            self.rc_config = {}
        if not hasattr(self, "plot_params"):
            self.plot_params = {}
        if not isinstance(self.experiment, ExperimentConfig):
            raise TypeError("experiment must be an ExperimentConfig instance")

    def _experiment_outputs_ready(self) -> bool:
        # Only require that data fields are present and are pandas objects
        data_ready = all(
            hasattr(self.experiment.data, attr)
            and getattr(self.experiment.data, attr) is not None
            and hasattr(getattr(self.experiment.data, attr), "shape")
            for attr in ["X_train", "y_train", "X_test", "y_test"]
        )
        # Only require that the model object exists
        model_ready = self.experiment.model is not None
        attack_ready = True
        return data_ready and model_ready and attack_ready

    def _ensure_experiment_prepared(self) -> dict:
        if self._experiment_prepared and self._experiment_outputs_ready():
            return self._experiment_scores
        if self._experiment_prepared and not self._experiment_outputs_ready():
            self._experiment_prepared = False
        if self._experiment_outputs_ready():
            self._experiment_prepared = True
            self._experiment_scores = self._experiment_scores or getattr(
                self,
                "score_dict",
                {},
            )
            return self._experiment_scores
        self._experiment_scores = self.experiment()
        # In some runs, cached objects may deserialize with split attrs present but None.
        # Force data materialization so Yellowbrick visualizers always receive arrays/dataframes.
        if (
            not self._experiment_outputs_ready()
            and hasattr(self.experiment, "data")
            and callable(getattr(self.experiment.data, "__call__", None))
        ):
            self.experiment.data(data_file=None, score_file=None)
        # Fallback: generate synthetic data if still not ready
        if not self._experiment_outputs_ready():
            raise RuntimeError(
                "Experiment data is not prepared: X_train/y_train/X_test/y_test are required for plotting.",
            )
        self._experiment_prepared = True
        return self._experiment_scores

    def _default_top_features(self, X, limit: int = 10):
        """Pick top features by model importance when features are unspecified."""
        if limit <= 0:
            return []
        if hasattr(X, "columns"):
            all_features = X.columns.tolist()
        else:
            all_features = list(range(X.shape[1]))
        if len(all_features) <= limit:
            return all_features

        default_features = all_features[:limit]

        def _features_from_data_scores():
            data_cfg = getattr(self.experiment, "data", None)
            score_dict = getattr(data_cfg, "score_dict", None)
            if not isinstance(score_dict, dict):
                return None

            model_cfg = getattr(self.experiment, "model", None)
            classifier_flag = getattr(model_cfg, "classifier", None)
            if classifier_flag is None:
                classifier_flag = getattr(data_cfg, "classifier", None)

            if classifier_flag is True:
                score_priority = [
                    "mutual_info_classif",
                    "f_classif",
                ]
            else:
                score_priority = [
                    "mutual_info_regression",
                    "f_regression",
                    "r_regression",
                ]

            for key in score_priority:
                values = score_dict.get(key)
                if values is None:
                    continue
                arr = np.asarray(values)
                if arr.ndim != 1 or len(arr) != len(all_features):
                    continue
                try:
                    arr = arr.astype(float)
                except (TypeError, ValueError):
                    continue
                if key == "r_regression":
                    arr = np.abs(arr)
                top_idx = np.argsort(arr)[::-1][:limit]
                return [all_features[i] for i in top_idx]

            return None

        if self.experiment.model is None:
            return _features_from_data_scores() or default_features

        try:
            estimator = self._get_plot_model()
        except Exception:
            return _features_from_data_scores() or default_features

        importance = None
        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_)
            importance = np.abs(coef)
            if importance.ndim > 1:
                importance = importance.mean(axis=0)

        if (
            importance is None
            or importance.ndim != 1
            or len(importance) != len(all_features)
        ):
            return _features_from_data_scores() or default_features

        top_idx = np.argsort(importance)[::-1][:limit]
        return [all_features[i] for i in top_idx]

    def _get_plot_data(self, test: bool = False, attack: bool = False):
        """Return the dataset slice required by the current Yellowbrick plot."""
        # Assert that either test or attack is False or both are False
        assert not (
            test and attack
        ), "Cannot load both test and attack data simultaneously"
        X, y = self.experiment.data.X_train, self.experiment.data.y_train
        X_test, y_test = (
            self.experiment.data.X_test,
            self.experiment.data.y_test,
        )
        if self.classes == "all":
            classes = np.unique(y)
        else:
            classes = self.classes
        if self.features == "all":
            if hasattr(X, "columns"):
                features = X.columns.tolist()
            else:
                features = list(range(X.shape[1]))
        else:
            features = self.features

        # Keep feature labels and matrices aligned.
        if isinstance(features, list):
            if hasattr(X, "columns"):
                X = X.loc[:, features]
                X_test = X_test.loc[:, features]
            else:
                X = X[:, features]
                X_test = X_test[:, features]
        if attack:
            X_attack, y_attack = (
                self.experiment.attack.attack,
                self.experiment.data.y_train[: self.experiment.attack.attack_size],
            )
            return X_attack, y_attack, classes, features
        if not test:
            return X, y, classes, features
        else:
            return X_test, y_test, classes, features

    def _get_plot_model(self):
        """Return the instantiated estimator for Yellowbrick visualizers."""
        estimator = self.experiment.model.get_model()

        if self.plot_type in classifier_viz_types:
            has_classifier_api = all(
                hasattr(estimator, attr) for attr in ("fit", "predict", "score")
            ) and (
                hasattr(estimator, "predict_proba")
                or hasattr(estimator, "decision_function")
            )
            if not has_classifier_api:
                return _named_classifier_adapter(self.experiment.model)

        return estimator

    def visualize_features(self, ax=None):
        """Generates and saves the Yellowbrick data plot."""
        X, y, classes, features = self._get_plot_data()
        if self.plot_type == "rank1d":
            visualizer = Rank1D(
                features=features,
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.transform(X)
        elif self.plot_type == "rank2d":
            visualizer = Rank2D(
                features=features,
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.transform(X)
        elif self.plot_type == "radviz":
            visualizer = RadViz(classes=classes, **self.plot_params, ax=ax)
            visualizer.fit(X, y)
            visualizer.transform(X)
        elif self.plot_type == "pcoords":
            visualizer = ParallelCoordinates(
                classes=classes,
                **self.plot_params,
                features=features,
            )
            visualizer.fit_transform(X, y)
            if hasattr(visualizer, "ax") and visualizer.ax is not None:
                for label in visualizer.ax.get_xticklabels():
                    label.set_rotation(90)
                    label.set_horizontalalignment("right")

        elif self.plot_type == "jointplot":
            assert (
                "columns" in self.plot_params
            ), "Columns must be specified for jointplot"
            visualizer = JointPlotVisualizer(**self.plot_params, ax=ax)
            visualizer.fit_transform(X, y)
        elif self.plot_type == "pca":
            visualizer = PCADecomposition(
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit_transform(X, y)
        elif self.plot_type == "manifold":
            visualizer = Manifold(
                classes=classes,
                **self.plot_params,
                features=features,
                verbose=True,
                ax=ax,
            )
            visualizer.fit_transform(X, y, verbose=True)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick plot saved to {self.save_path}")

    def visualize_targets(self, ax=None):
        X, y, classes, feature_indices = self._get_plot_data()
        if self.plot_type == "class_balance":
            visualizer = ClassBalance(labels=classes, **self.plot_params, ax=ax)
            visualizer.fit(y)
        elif self.plot_type == "balanced_binning_reference":
            visualizer = BalancedBinningReference(**self.plot_params, ax=ax)
            visualizer.fit(y)
        elif self.plot_type == "feature_correlation":
            visualizer = FeatureCorrelation(
                features=feature_indices,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick target plot saved to {self.save_path}")

    def visualize_regressors(self, ax=None):
        X, y, _, _ = self._get_plot_data()
        X_test, y_test, _, _ = self._get_plot_data(test=True)
        model = self._get_plot_model()
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

    def visualize_classifiers(self, ax=None):
        X, y, classes, _ = self._get_plot_data()
        X_test, y_test, _, _ = self._get_plot_data(test=True)
        model = self._get_plot_model()
        if self.plot_type == "classification_report":
            visualizer = ClassificationReport(
                model,
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "roc_auc":
            visualizer = ROCAUC(
                model,
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "precision_recall_curve":
            visualizer = PrecisionRecallCurve(
                model,
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "class_prediction_error":
            visualizer = ClassPredictionError(
                model,
                classes=classes,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        elif self.plot_type == "discrimination_threshold":
            visualizer = DiscriminationThreshold(
                model,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            visualizer.score(X_test, y_test)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")
        self.show(visualizer)
        logger.info(f"Yellowbrick classifier plot saved to {self.save_path}")

    def visualize_clusters(self, ax):
        X, _, _, _ = self._get_plot_data()
        model = self._get_plot_model()
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
        X, y, _, features = self._get_plot_data()
        model = self._get_plot_model()
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
            visualizer = FeatureImportances(
                model,
                **self.plot_params,
                labels=features,
                ax=ax,
            )
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
        logger.info(
            f"Yellowbrick model selection plot saved to {self.save_path}",
        )

    def parse_cv(self):
        if "cv" not in self.plot_params:
            classifier_flag = getattr(self.experiment.model, "classifier", True)
            if classifier_flag:
                logger.warning(
                    "No cv specified for model-selection plot '%s'; defaulting to StratifiedKFold(n_splits=5)",
                    self.plot_type,
                )
                return StratifiedKFold(n_splits=5)
            logger.warning(
                "No cv specified for model-selection plot '%s'; defaulting to KFold(n_splits=5)",
                self.plot_type,
            )
            return KFold(n_splits=5)

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
        assert (
            "param_range" in self.plot_params
        ), "Param_range must be specified for validation_curve"
        param_range = self.plot_params.pop("param_range")
        num = self.plot_params.pop("num", 10)
        # Accept tuple or list
        if isinstance(param_range, tuple):
            param_range = list(param_range)
        assert (
            len(param_range) == 2 or len(param_range) == 3
        ), "Param_range must be a list or tuple of 2 or 3 values"
        if len(param_range) == 2:
            param_range = np.linspace(param_range[0], param_range[1], num=num)
        elif len(param_range) == 3:
            if param_range[2] == "log":
                param_range = np.logspace(
                    np.log10(param_range[0]),
                    np.log10(param_range[1]),
                    num=num,
                )
            elif param_range[2] == "linear":
                param_range = np.linspace(
                    param_range[0],
                    param_range[1],
                    num=num,
                )
            elif isinstance(param_range[2], (int, float)):
                steps = int((param_range[1] - param_range[0]) // param_range[2])
                param_range = np.linspace(
                    start=param_range[0],
                    stop=param_range[1],
                    num=steps,
                    dtype=type(param_range[2]),
                )
            else:
                raise ValueError(
                    "Distribution must be either 'log' or 'linear'",
                )
        # Always return as list for compatibility
        return (
            param_range.tolist()
            if hasattr(param_range, "tolist")
            else list(param_range)
        )

    def visualize(self, ax=None):
        """Main method to generate and save the Yellowbrick plot."""
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
        elif self.plot_type in [
            "validation_curve",
            "learning_curve",
            "cv_scores",
            "feature_importances",
            "rfecv",
            "dropping_curve",
        ]:
            self.visualize_model_selection(ax)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")

    def show(self, visualizer):
        assert hasattr(
            visualizer,
            "show",
        ), "Visualizer does not have a show method"
        assert isinstance(
            visualizer,
            tuple(all_viz_objects),
        ), "Visualizer is not a recognized Yellowbrick visualizer"

        # Yellowbrick applies default titles in finalize(); run finalize first,
        # then enforce any user-specified title before saving.
        if hasattr(visualizer, "finalize"):
            visualizer.finalize()

        if getattr(self, "title", None):
            if hasattr(visualizer, "ax") and visualizer.ax is not None:
                visualizer.ax.set_title(self.title)
            elif hasattr(visualizer, "axes") and visualizer.axes is not None:
                try:
                    axes = np.ravel(visualizer.axes)
                    if axes.size > 0 and axes[0] is not None:
                        axes[0].set_title(self.title)
                except Exception:
                    pass

        plt.savefig(self.save_path)

    def __len__(self):
        return 1

    def __call__(self) -> dict:
        if self.rc_config:
            plt.rcParams.update(self.rc_config)
        scores = self._ensure_experiment_prepared()
        self.visualize()
        return scores


@dataclass(kw_only=True)
class YellowbrickConfigList(ConfigBase):
    """Render a collection of Yellowbrick plots from one prepared experiment.

    This config composes an ExperimentConfig and prepares it once, then fans that
    state out into child ``YellowbrickPlotConfig`` objects so each plot render
    reuses the same trained data/model/attack artifacts.
    """

    experiment: ExperimentConfig
    plots: Union[
        Dict[str, YellowbrickPlotConfig],
        Literal["all"],
        List[str],
    ] = "all"
    clustering: bool = False
    plot_folder: Optional[str] = None
    rc_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._experiment_prepared = False
        self._experiment_scores = {}
        if not hasattr(self, "score_dict"):
            self.score_dict = {}
        if not hasattr(self, "rc_config"):
            self.rc_config = {}
        if not isinstance(self.experiment, ExperimentConfig):
            raise TypeError("experiment must be an ExperimentConfig instance")

        if self.plot_folder is None:
            self.plot_folder = Path().cwd().as_posix()
        else:
            self.plot_folder = Path(self.plot_folder).as_posix()

    def _get_default_plot_params(self, plot_type: str) -> Dict[str, Any]:
        plot_folder = (
            Path(self.plot_folder)
            if self.plot_folder is not None
            else Path(Path.cwd(), "plots")
        )
        plot_params: Dict[str, Any] = {}
        if plot_type == "jointplot":
            x_train = getattr(self.experiment.data, "X_train", None)
            if hasattr(x_train, "columns") and len(x_train.columns) >= 2:
                plot_params["columns"] = [
                    x_train.columns[0],
                    x_train.columns[1],
                ]

        return {
            "features": "all",
            "classes": "all",
            "plot_type": plot_type,
            "clustering": self.clustering,
            "title": plot_type.replace("_", " ").title(),
            "save_path": (plot_folder / f"{plot_type}.png").as_posix(),
            "rc_config": self.rc_config,
            "plot_params": plot_params,
        }

    def _experiment_outputs_ready(self) -> bool:
        # Only require that data fields are present and are pandas objects
        data_ready = all(
            hasattr(self.experiment.data, attr)
            and getattr(self.experiment.data, attr) is not None
            and hasattr(getattr(self.experiment.data, attr), "shape")
            for attr in ["X_train", "y_train", "X_test", "y_test"]
        )
        # Only require that the model object exists
        model_ready = self.experiment.model is not None
        attack_ready = True
        return data_ready and model_ready and attack_ready

    def _ensure_experiment_prepared(self) -> dict:
        if self._experiment_prepared and self._experiment_outputs_ready():
            return self._experiment_scores
        if self._experiment_prepared and not self._experiment_outputs_ready():
            self._experiment_prepared = False
        if self._experiment_outputs_ready():
            self._experiment_prepared = True
            self._experiment_scores = self._experiment_scores or getattr(
                self,
                "score_dict",
                {},
            )
            return self._experiment_scores
        self._experiment_scores = self.experiment()
        if (
            not self._experiment_outputs_ready()
            and hasattr(self.experiment, "data")
            and callable(getattr(self.experiment.data, "__call__", None))
        ):
            self.experiment.data(data_file=None, score_file=None)
        if not self._experiment_outputs_ready():
            raise RuntimeError(
                "Experiment data is not prepared: X_train/y_train/X_test/y_test are required for plotting.",
            )
        self._experiment_prepared = True
        return self._experiment_scores

    def _set_plot_dict(self):
        if isinstance(self.plots, dict):
            return

        data_plots = []
        model_plots = []
        plot_list = []
        if self.plots == "all":
            if self.experiment.data is not None:
                data_plots += feature_viz_types + target_viz_types
            if self.experiment.model is not None:
                model_plots = list(model_selection_viz_types)
                if self.clustering:
                    model_plots += cluster_viz_types
                elif self.experiment.model.classifier is True:
                    model_plots += classifier_viz_types
                else:
                    model_plots += regressor_viz_types
                if self.experiment.attack is not None:
                    logger.debug(
                        "Plotting attacks with yellowbick isn't supported (yet).",
                    )
            plot_list = data_plots + model_plots
        elif isinstance(self.plots, list):
            plot_list = self.plots

        plot_dict = {}
        for plot_type in plot_list:
            cfg = YellowbrickPlotConfig(
                experiment=self.experiment,
                **self._get_default_plot_params(plot_type=plot_type),
            )
            # Do not force _experiment_prepared; let each config prepare itself
            plot_dict[plot_type] = cfg
        self.plots = plot_dict

    def __len__(self):
        return len(self.plots) if isinstance(self.plots, dict) else 0

    def __call__(self) -> dict:
        if self.rc_config:
            plt.rcParams.update(self.rc_config)
        scores = self._ensure_experiment_prepared()
        if self.plots == "all" or isinstance(self.plots, list):
            self._set_plot_dict()
        plot_dict = self.plots if isinstance(self.plots, dict) else {}
        for plot_cfg in plot_dict.values():
            try:
                logger.info(
                    f"Rendering {plot_cfg.plot_type} plot with params: {plot_cfg.plot_params}. ",
                )
                plot_cfg()
            except Exception:
                logger.exception(
                    "Failed to generate plot %s",
                    plot_cfg.plot_type,
                )
        return scores
