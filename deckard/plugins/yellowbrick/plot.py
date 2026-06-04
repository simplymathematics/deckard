"""Yellowbrick plotting configs built on top of prepared experiment state.

Plot configs compose an ExperimentConfig to prepare experiment artifacts once,
then reuse that prepared state across multiple plot renders.
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Union, get_args

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)
from ...artifacts import ScoreDict

# Classifier Visualizers
from yellowbrick.classifier import (
    ROCAUC,
    ClassificationReport,
    ClassPredictionError,
    DiscriminationThreshold,
    PrecisionRecallCurve,
)

# Clustering Visualizers
from yellowbrick.cluster import (
    InterclusterDistance,
    KElbowVisualizer,
    SilhouetteVisualizer,
)
from yellowbrick.features.jointplot import JointPlotVisualizer
from yellowbrick.features.manifold import Manifold
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.radviz import RadViz

# yellow brick imports
# Feature Visualizers
from yellowbrick.features.rankd import Rank1D, Rank2D

# Model Selection Visualizers
from yellowbrick.model_selection import (
    RFECV,
    CVScores,
    DroppingCurve,
    FeatureImportances,
    LearningCurve,
    ValidationCurve,
)

# Regressor Visualizers
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import ManualAlphaSelection

# Target Visualizers Imports
from yellowbrick.target import (
    BalancedBinningReference,
    ClassBalance,
    FeatureCorrelation,
)

from ...experiment import ExperimentConfig
from ...frameworks.pytorch.score import (
    get_dataset_shape as _get_shape,
)
from ...frameworks.pytorch.score import (
    is_dataloader_like as _is_dataloader_like,
)
from ...frameworks.pytorch.score import (
    is_dataset_like as _is_dataset_like,
)
from ...frameworks.pytorch.score import (
    materialize_dataset as _materialize_dataset_features_labels,
)
from ...frameworks.pytorch.score import (  # noqa: F401
    to_numpy as _to_numpy,
)
from ...plot.base import (
    PlotterMixin,
    _YellowbrickPlotterMarker,
)
from ...utils import BaseConfig

try:
    from torch.utils.data import Subset
except Exception:
    Subset = None

try:
    import torch
except Exception:
    torch = None

feature_viz_types: Final = (
    "rank1d",
    "rank2d",
    "radviz",
    "pcoords",
    "jointplot",
    "pca",
    "manifold",
)

target_viz_types: Final = (
    "class_balance",
    "balanced_binning_reference",
    "feature_correlation",
)

regressor_viz_types: Final = (
    "prediction_error",
    "residuals_plot",
    "alpha_selection",
)

classifier_viz_types: Final = (
    "roc_auc",
    "precision_recall_curve",
    "classification_report",
    "class_prediction_error",
    "discrimination_threshold",
)

cluster_viz_types: Final = (
    "k_elbow",
    "silhouette",
    "intercluster_distance",
)

model_selection_viz_types: Final = (
    "validation_curve",
    "learning_curve",
    "cv_scores",
    "feature_importances",
    "rfecv",
    "dropping_curve",
)

all_viz_types: Final = (
    feature_viz_types
    + target_viz_types
    + regressor_viz_types
    + classifier_viz_types
    + cluster_viz_types
    + model_selection_viz_types
)


YellowBrickVizType = Literal[
    "rank1d",
    "rank2d",
    "radviz",
    "pcoords",
    "jointplot",
    "pca",
    "manifold",
    "class_balance",
    "balanced_binning_reference",
    "feature_correlation",
    "prediction_error",
    "residuals_plot",
    "alpha_selection",
    "roc_auc",
    "precision_recall_curve",
    "classification_report",
    "class_prediction_error",
    "discrimination_threshold",
    "k_elbow",
    "silhouette",
    "intercluster_distance",
    "validation_curve",
    "learning_curve",
    "cv_scores",
    "feature_importances",
    "rfecv",
    "dropping_curve",
]

# Ensure the Literal stays in sync with your runtime list.
_LITERAL_VIZ_TYPES = set(get_args(YellowBrickVizType))
_RUNTIME_VIZ_TYPES = set(all_viz_types)

if _LITERAL_VIZ_TYPES != _RUNTIME_VIZ_TYPES:
    missing_from_literal = _RUNTIME_VIZ_TYPES - _LITERAL_VIZ_TYPES
    extra_in_literal = _LITERAL_VIZ_TYPES - _RUNTIME_VIZ_TYPES

    raise RuntimeError(
        "YellowBrickVizType is out of sync with all_viz_types.\n"
        f"Missing from Literal: {sorted(missing_from_literal)}\n"
        f"Extra in Literal: {sorted(extra_in_literal)}",
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

SUPPORTED_YELLOWBRICK_PLOT_TYPES = all_viz_types
YELLOWBRICK_PLOT_TYPE_DOCS: Final[dict[str, str]] = {
    "rank1d": "Rank single features by score against the target.",
    "rank2d": "Visualize pairwise feature relationships by target class.",
    "radviz": "Project features radially to inspect class separability.",
    "pcoords": "Render parallel coordinates for multi-feature class comparison.",
    "jointplot": "Show 2D feature relationship with marginal distributions.",
    "pca": "Project features to principal components.",
    "manifold": "Project features with nonlinear manifold embedding.",
    "class_balance": "Display per-class sample counts.",
    "balanced_binning_reference": "Visualize target-value binning quality.",
    "feature_correlation": "Measure correlation between each feature and target.",
    "prediction_error": "Compare predicted vs true values for regressors.",
    "residuals_plot": "Inspect regression residual behavior.",
    "alpha_selection": "Inspect regularization alpha sensitivity.",
    "roc_auc": "Plot ROC curves and AUC for classifiers.",
    "precision_recall_curve": "Plot precision-recall tradeoff curves.",
    "classification_report": "Visualize precision/recall/F1 per class.",
    "class_prediction_error": "Show class-wise prediction error matrix.",
    "discrimination_threshold": "Sweep binary decision thresholds.",
    "k_elbow": "Estimate optimal cluster count with elbow metric.",
    "silhouette": "Inspect cluster cohesion and separation.",
    "intercluster_distance": "Visualize inter-cluster distances.",
    "validation_curve": "Plot score vs hyperparameter values.",
    "learning_curve": "Plot score vs training set size.",
    "cv_scores": "Summarize cross-validation scores across folds.",
    "feature_importances": "Display estimator feature importances.",
    "rfecv": "Run recursive feature elimination with CV.",
    "dropping_curve": "Measure score impact while dropping features.",
}

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def _get_shape(obj):  # noqa: F811
    if hasattr(obj, "shape"):
        return obj.shape
    if hasattr(obj, "dataset") and hasattr(obj.dataset, "shape"):
        shape = obj.dataset.shape
        if hasattr(obj, "indices") and len(shape) > 0:
            return (len(obj.indices), *shape[1:])
        return shape
    if Subset is not None and isinstance(obj, Subset):
        # Unwrap nested Subset / TensorDataset / custom dataset wrappers
        base = obj.dataset

        # Try recursive resolution first
        try:
            shape = _get_shape(base)
            if len(shape) > 0:
                return (len(obj.indices), *shape[1:])
            return shape
        except AttributeError:
            pass

        # Handle TensorDataset-like objects (.tensors)
        if hasattr(base, "tensors") and base.tensors:
            tensor = base.tensors[0]
            if hasattr(tensor, "shape"):
                shape = tensor.shape
                if len(shape) > 0:
                    return (len(obj.indices), *shape[1:])
                return shape
        if len(obj) > 0:
            first = obj[0]
            first_x = (
                first[0]
                if isinstance(first, (tuple, list)) and len(first) > 0
                else first
            )
            first_arr = _to_numpy(first_x)
            sample_shape = getattr(first_arr, "shape", ())
            if len(sample_shape) > 0:
                return (len(obj), *sample_shape)
            return (len(obj),)
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and len(obj) > 0:
        first = obj[0]
        first_x = (
            first[0] if isinstance(first, (tuple, list)) and len(first) > 0 else first
        )
        first_arr = _to_numpy(first_x)
        sample_shape = getattr(first_arr, "shape", ())
        if len(sample_shape) > 0:
            return (len(obj), *sample_shape)
        return (len(obj),)
    raise AttributeError(f"{type(obj).__name__} has no shape")


class _YellowbrickModelAdapter(BaseEstimator, ClassifierMixin):
    """Expose deckard model configs with a sklearn-like inference interface.

    Yellowbrick classifier visualizers expect estimators that implement sklearn
    conventions (``fit``, ``score``, and ``predict_proba``/``decision_function``).
    PyTorch-backed deckard models are not sklearn estimators, so this adapter
    bridges that API using the model config's existing prediction path.
    """

    def __init__(self, model_config: Any, classifier: bool = True):
        self.model_config = model_config
        self._estimator_type = "classifier" if classifier else "regressor"
        self.classes_ = None

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        return _to_numpy(value)

    def _is_transformers_config(self) -> bool:
        model_type = type(self.model_config)
        module_name = str(getattr(model_type, "__module__", "")).lower()
        class_name = str(getattr(model_type, "__name__", "")).lower()
        return "transformers" in module_name or "huggingface" in class_name

    @staticmethod
    def _coerce_transformers_input(X: Any) -> Any:
        """Coerce feature payload toward token-id compatible inputs.

        Yellowbrick visualizers may pass float arrays/tensors, but transformer
        token ids are integral. When values appear floating-point, round and
        cast to int64 while preserving non-array payloads.
        """
        if torch is not None and isinstance(X, torch.Tensor):
            if X.dtype.is_floating_point:
                return torch.round(X).to(dtype=torch.long)
            return X

        if isinstance(X, np.ndarray):
            if np.issubdtype(X.dtype, np.floating):
                return np.rint(X).astype(np.int64, copy=False)
            return X

        if isinstance(X, dict):
            coerced = {}
            for key, value in X.items():
                if torch is not None and isinstance(value, torch.Tensor):
                    if value.dtype.is_floating_point:
                        coerced[key] = torch.round(value).to(dtype=torch.long)
                    else:
                        coerced[key] = value
                elif isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype,
                    np.floating,
                ):
                    coerced[key] = np.rint(value).astype(np.int64, copy=False)
                else:
                    coerced[key] = value
            return coerced

        return X

    def _prediction_candidates(self, X: Any) -> list[Any]:
        """Yield progressively more permissive prediction inputs."""
        candidates: list[Any] = [X]

        if self._is_transformers_config():
            coerced = self._coerce_transformers_input(X)
            if coerced is not X:
                candidates.append(coerced)

            if torch is not None and isinstance(coerced, np.ndarray):
                try:
                    device = getattr(self.model_config, "device", "cpu")
                    tensor = torch.as_tensor(coerced, dtype=torch.long, device=device)
                    candidates.append(tensor)
                except Exception:
                    pass
            return candidates

        if (
            torch is not None
            and isinstance(X, np.ndarray)
            and getattr(self.model_config, "library", "") == "pytorch"
        ):
            device = getattr(self.model_config, "device", "cpu")
            candidates.append(torch.as_tensor(X, dtype=torch.float32, device=device))
        return candidates

    def fit(self, X, y=None, **kwargs):
        """No-op fit adapter used by Yellowbrick estimator contracts.

        Args:
            X: Training features (unused for fitting in this adapter).
            y: Optional labels used to infer classifier classes.
            **kwargs: Additional estimator kwargs (ignored).

        Returns:
            The current adapter instance.
        """
        # Experiment preparation performs model fitting already.
        if y is not None and self._estimator_type == "classifier":
            y_arr = self._to_numpy(y).reshape(-1)
            self.classes_ = np.unique(y_arr)
        return self

    def _predict_raw(self, X) -> np.ndarray:
        last_error = None
        for candidate in self._prediction_candidates(X):
            prepared = candidate
            if (
                torch is not None
                and isinstance(prepared, torch.Tensor)
                and getattr(self.model_config, "library", "") == "pytorch"
                and not self._is_transformers_config()
            ):
                # Yellowbrick may pass grayscale image batches as [N, H, W].
                # Conv2d expects [N, C, H, W], so inject a channel dimension.
                if prepared.ndim == 3:
                    if (
                        prepared.shape[0] <= 4
                        and prepared.shape[1] > 4
                        and prepared.shape[2] > 4
                    ):
                        prepared = prepared.unsqueeze(0)
                    else:
                        prepared = prepared.unsqueeze(1)

            try:
                raw = self.model_config.predict(prepared)
                return self._to_numpy(raw)
            except Exception as exc:
                last_error = exc

                # Transformers pipelines may fail on MPS during visualization-only
                # inference paths. Retry on CPU to keep Yellowbrick diagnostics
                # runnable in notebook and docs environments.
                if self._is_transformers_config() and "mps" in str(exc).lower():
                    original_device = getattr(self.model_config, "device", None)
                    try:
                        cpu_candidate = prepared
                        if torch is not None and isinstance(
                            cpu_candidate,
                            torch.Tensor,
                        ):
                            cpu_candidate = cpu_candidate.detach().cpu()

                        if original_device is not None:
                            self.model_config.device = "cpu"
                        raw = self.model_config.predict(cpu_candidate)
                        return self._to_numpy(raw)
                    except Exception as cpu_exc:
                        last_error = cpu_exc
                    finally:
                        if original_device is not None:
                            self.model_config.device = original_device

        raise RuntimeError(
            "Yellowbrick adapter failed to obtain predictions from model config",
        ) from last_error

    def predict_proba(self, X) -> np.ndarray:
        """Return probability-like predictions for Yellowbrick classifiers.

        Args:
            X: Feature payload for inference.

        Returns:
            Probability matrix.

        Raises:
            ValueError: If raw prediction shape cannot be converted to probabilities.
        """
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
            f"Unsupported prediction shape for predict_proba: {_get_shape(raw)}",
        )

    def predict(self, X) -> np.ndarray:
        """Return class predictions for Yellowbrick estimator interfaces.

        Args:
            X: Feature payload for inference.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If raw prediction shape cannot be converted to labels.
        """
        raw = self._predict_raw(X)
        if raw.ndim == 1:
            return (raw > 0.0).astype(int)
        if raw.ndim == 2:
            return np.argmax(raw, axis=1)
        raise ValueError(
            f"Unsupported prediction shape for predict: {_get_shape(raw)}",
        )

    def score(self, X, y) -> float:
        """Return simple classification accuracy for Yellowbrick consumers.

        Args:
            X: Feature payload for inference.
            y: Ground-truth labels.

        Returns:
            Classification accuracy score.
        """
        y_true = self._to_numpy(y).reshape(-1)
        y_pred = self.predict(X).reshape(-1)
        if _get_shape(y_true)[0] == 0:
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


@dataclass(eq=True)
class YellowbrickPlotterMixin(PlotterMixin):
    """Yellowbrick-specific plotter handler for ML model visualization.

    The runtime object provides the active Yellowbrick plot config, including
    experiment context, visualizer type, clustering mode, selected features or
    classes, and visualizer-specific plot parameters.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __call__(
        self,
        *,
        ax: Axes | None = None,
        **kwargs: dict,
    ) -> ScoreDict:
        """Execute yellowbrick visualization.

        Args:
            ax: Matplotlib axis to plot on.
            **kwargs: Additional visualizer parameters.

        Returns:
            Score payload for rendered plot context.

        Raises:
            NotImplementedError: Always raised by the base mixin implementation.
        """
        # This is a placeholder implementation.
        # The actual implementation delegates to runtime config methods
        # (visualize_features, visualize_targets, etc.)
        raise NotImplementedError(
            "YellowbrickPlotterMixin.__call__ must be implemented by subclass",
        )


@dataclass(kw_only=True, eq=False)
class YellowbrickPlotConfig(_YellowbrickPlotterMarker, BaseConfig):
    """Render a single Yellowbrick plot from composed experiment configuration.

    This config stores the experiment context, Yellowbrick visualizer type,
    feature or class selection, rendering metadata, and output path for one
    plot. It inherits ``_YellowbrickPlotterMarker`` so runtime dispatch
    resolves the Yellowbrick plotting mixin when the Yellowbrick backend is
    selected.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    experiment: ExperimentConfig

    # Plot-specific parameters
    plot_type: Literal[YellowBrickVizType]
    clustering: bool = False
    features: Union[List[str], Literal["all"]] = "all"
    classes: Union[List[str], Literal["all"]] = "all"
    title: str = "Yellowbrick Plot"
    save_path: str = "yellowbrick_plot.png"
    rc_config: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: rc_config."},
    )
    plot_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: plot_params."},
    )

    def __post_init__(self):
        self._experiment_prepared = False
        self._experiment_scores = {}
        if not hasattr(self, "score_dict"):
            self.score_dict = ScoreDict()
        else:
            self.score_dict = ScoreDict.from_payload(getattr(self, "score_dict", {}))
        if not hasattr(self, "rc_config"):
            self.rc_config = {}
        if not hasattr(self, "plot_params"):
            self.plot_params = {}
        if not isinstance(self.experiment, ExperimentConfig):
            raise TypeError("experiment must be an ExperimentConfig instance")

    def _experiment_outputs_ready(self) -> bool:
        def _materializable(value: Any) -> bool:
            return value is not None and (
                hasattr(value, "shape")
                or _is_dataset_like(value)
                or _is_dataloader_like(value)
            )

        data_obj = getattr(self.experiment, "data", None)
        if data_obj is None:
            return False

        # Preferred path for plotting APIs: train/test splits.
        split_ready = all(
            hasattr(data_obj, attr) and _materializable(getattr(data_obj, attr))
            for attr in ["X_train", "y_train", "X_test", "y_test"]
        )

        # Fallback path for configs that only materialize full arrays until sampled.
        full_ready = all(
            hasattr(data_obj, attr) and _materializable(getattr(data_obj, attr))
            for attr in ["_X", "_y"]
        )

        data_ready = split_ready or full_ready
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
            self.experiment.data(
                files={
                    "data_file": None,
                    "score_file": None,
                },
            )
        # If full arrays are present but splits are missing, force an in-memory sample split.
        if (
            not self._experiment_outputs_ready()
            and hasattr(self.experiment, "data")
            and callable(getattr(self.experiment.data, "_sample", None))
            and getattr(self.experiment.data, "_X", None) is not None
        ):
            self.experiment.data.sample()
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
            all_features = list(range(_get_shape(X)[1]))
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

        if (
            not hasattr(X, "columns")
            and not hasattr(X, "shape")
            and (_is_dataset_like(X) or _is_dataloader_like(X))
        ):
            X_mat, y_mat = _materialize_dataset_features_labels(X)
            X = X_mat
            if y_mat is not None and (
                y is None or np.asarray(y).reshape(-1).shape[0] != y_mat.shape[0]
            ):
                y = y_mat
        if (
            not hasattr(X_test, "columns")
            and not hasattr(X_test, "shape")
            and (_is_dataset_like(X_test) or _is_dataloader_like(X_test))
        ):
            X_test_mat, y_test_mat = _materialize_dataset_features_labels(X_test)
            X_test = X_test_mat
            if y_test_mat is not None and (
                y_test is None
                or np.asarray(y_test).reshape(-1).shape[0] != y_test_mat.shape[0]
            ):
                y_test = y_test_mat

        y = _to_numpy(y).reshape(-1)
        y_test = _to_numpy(y_test).reshape(-1)

        if self.classes == "all":
            classes = np.asarray(np.unique(y)).reshape(-1).tolist()
        else:
            classes = np.asarray(self.classes).reshape(-1).tolist()
        if self.features == "all":
            if hasattr(X, "columns"):
                features = X.columns.tolist()
            else:
                features = list(range(_get_shape(X)[1]))
        else:
            features = self.features

        # Keep feature labels and matrices aligned.
        if isinstance(features, list):
            if hasattr(X, "columns"):
                X = X.loc[:, features]
                X_test = X_test.loc[:, features]
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

    def visualize_features(self, ax: Axes | None = None) -> None:
        """Render feature-focused Yellowbrick plots.

        Supported ``plot_type`` values:
            ``rank1d``, ``rank2d``, ``radviz``, ``pcoords``, ``jointplot``,
            ``pca``, ``manifold``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If the configured plot_type is not a supported feature visualizer.
        """
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

    def visualize_targets(self, ax: Axes | None = None) -> None:
        """Render target-focused Yellowbrick plots.

        Supported ``plot_type`` values:
            ``class_balance``, ``balanced_binning_reference``,
            ``feature_correlation``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If the configured plot_type is not a supported target visualizer.
        """
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

    def visualize_regressors(self, ax: Axes | None = None) -> None:
        """Render regressor-focused Yellowbrick plots.

        Supported ``plot_type`` values:
            ``prediction_error``, ``residuals_plot``, ``alpha_selection``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If the configured plot_type is not a supported regressor visualizer.
        """
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

    def visualize_classifiers(self, ax: Axes | None = None) -> None:
        """Render classifier-focused Yellowbrick plots.

        Supported ``plot_type`` values:
            ``roc_auc``, ``precision_recall_curve``, ``classification_report``,
            ``class_prediction_error``, ``discrimination_threshold``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If the configured plot_type is not a supported classifier visualizer.
        """
        X, y, _classes, _ = self._get_plot_data()
        X_test, y_test, _, _ = self._get_plot_data(test=True)
        model = self._get_plot_model()
        inferred_classes = np.asarray(np.unique(np.asarray(y).reshape(-1))).reshape(-1)
        if self.plot_type == "classification_report":
            visualizer = ClassificationReport(
                model,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            if getattr(visualizer, "classes_", None) is None:
                visualizer.classes_ = inferred_classes
            visualizer.score(X_test, y_test)
        elif self.plot_type == "roc_auc":
            visualizer = ROCAUC(
                model,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            if getattr(visualizer, "classes_", None) is None:
                visualizer.classes_ = inferred_classes
            visualizer.score(X_test, y_test)
        elif self.plot_type == "precision_recall_curve":
            visualizer = PrecisionRecallCurve(
                model,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            if getattr(visualizer, "classes_", None) is None:
                visualizer.classes_ = inferred_classes
            visualizer.score(X_test, y_test)
        elif self.plot_type == "class_prediction_error":
            visualizer = ClassPredictionError(
                model,
                **self.plot_params,
                ax=ax,
            )
            visualizer.fit(X, y)
            if getattr(visualizer, "classes_", None) is None:
                visualizer.classes_ = inferred_classes
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

    def visualize_clusters(self, ax: Axes | None = None) -> None:
        """Render clustering-focused Yellowbrick plots.

        Supported ``plot_type`` values:
            ``k_elbow``, ``silhouette``, ``intercluster_distance``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If the configured plot_type is not a supported clustering visualizer.
        """
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

    def visualize_model_selection(self, ax: Axes | None = None) -> None:
        """Render model-selection Yellowbrick plots.

        Supported ``plot_type`` values:
            ``validation_curve``, ``learning_curve``, ``cv_scores``,
            ``feature_importances``, ``rfecv``, ``dropping_curve``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If the configured plot_type is not a supported model-selection visualizer.
        """
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

    def parse_cv(self) -> KFold | StratifiedKFold | TimeSeriesSplit | ShuffleSplit:
        """Parse and instantiate the configured cross-validation strategy.

        Accepted runtime formats:
            int -> StratifiedKFold(n_splits=int)
            dict[name=kfold|timeseries|stratifiedkfold|shufflesplit] -> matching splitter
            omitted -> default KFold/StratifiedKFold based on classifier flag

        Returns:
            Instantiated cross-validation splitter.

        Raises:
            AssertionError: If dict cv configuration is missing required name key.
            ValueError: If cv strategy name is unsupported.
        """
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

    def parse_range(self) -> list[float | int]:
        """Parse ``param_range`` configuration into a concrete value sequence.

        Accepted formats:
            - ``[start, stop]``
            - ``[start, stop, "linear"]``
            - ``[start, stop, "log"]``
            - ``[start, stop, step]``

        Returns:
            Concrete range values as a list.

        Raises:
            AssertionError: If param_range is missing or malformed.
            ValueError: If distribution marker is unsupported.
        """
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

    def visualize(self, ax: Axes | None = None) -> None:
        """Dispatch rendering based on ``plot_type`` and save output image.

        See ``YELLOWBRICK_PLOT_TYPE_DOCS`` for human-readable descriptions of
        every supported ``plot_type``.

        Args:
            ax: Optional matplotlib axis for the rendered plot.

        Raises:
            ValueError: If plot_type is unsupported.
        """
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        original_set_clip_path = Artist.set_clip_path

        def _compat_set_clip_path(artist, path, transform=None):
            try:
                return original_set_clip_path(artist, path, transform)
            except TypeError as exc:
                message = str(exc)
                if (
                    "set_clip_path" in message
                    and transform is None
                    and getattr(type(path), "__name__", "") == "Rectangle"
                    and hasattr(path, "get_path")
                    and hasattr(path, "get_transform")
                ):
                    return original_set_clip_path(
                        artist,
                        path.get_path(),
                        path.get_transform(),
                    )
                raise

        Artist.set_clip_path = _compat_set_clip_path
        try:
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 8))

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
        finally:
            Artist.set_clip_path = original_set_clip_path

    def show(self, visualizer: BaseEstimator) -> None:
        """Finalize and save a fitted Yellowbrick visualizer to disk.

        Args:
            visualizer: Fitted yellowbrick visualizer instance.

        Raises:
            AssertionError: If visualizer does not provide required yellowbrick API.
        """
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
            try:
                visualizer.finalize()
            except Exception as exc:
                message = str(exc)
                if "Legend needs either Axes or FigureBase as parent" in message:
                    logger.warning(
                        "Skipping yellowbrick finalize legend step: %s",
                        message,
                    )
                elif "The passed figure is not managed by pyplot" in message:
                    logger.warning(
                        "Skipping yellowbrick finalize pyplot-close step: %s",
                        message,
                    )
                else:
                    raise

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

        figure = None
        if hasattr(visualizer, "ax") and visualizer.ax is not None:
            figure = getattr(visualizer.ax, "figure", None)
        elif hasattr(visualizer, "axes") and visualizer.axes is not None:
            try:
                axes = np.ravel(visualizer.axes)
                if axes.size > 0 and axes[0] is not None:
                    figure = getattr(axes[0], "figure", None)
            except Exception:
                figure = None

        original_from_any = FontProperties._from_any

        @classmethod
        def _fontprops_from_any_compat(cls, arg):
            if getattr(type(arg), "__name__", "") == "FontProperties":
                return arg
            return original_from_any(arg)

        FontProperties._from_any = _fontprops_from_any_compat
        try:
            if figure is not None:
                figure.savefig(self.save_path)
            else:
                plt.savefig(self.save_path)
        finally:
            FontProperties._from_any = original_from_any

    def __len__(self):
        return 1

    def __call__(self) -> ScoreDict:
        """Prepare experiment state, render ``plot_type``, and return score payload.

        The score payload originates from the prepared ``ExperimentConfig`` and
        is returned unchanged except for normalization to ``ScoreDict``.

        Returns:
            Normalized experiment score payload.
        """
        if self.rc_config:
            plt.rcParams.update(self.rc_config)
        scores = self._ensure_experiment_prepared()
        self.visualize()
        return ScoreDict.from_payload(scores)


@dataclass(kw_only=True)
class YellowbrickConfigList(BaseConfig):
    """Render a collection of Yellowbrick plots from one prepared experiment.

    This config composes an ExperimentConfig and prepares it once, then fans that
    state out into child ``YellowbrickPlotConfig`` objects so each plot render
    reuses the same trained data/model/attack artifacts.
    """

    experiment: ExperimentConfig
    plots: (
        dict[str, YellowbrickPlotConfig] | Literal["all"] | list[YellowBrickVizType]
    ) = "all"
    clustering: bool = False
    plot_folder: Optional[str] = None
    rc_config: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: rc_config."},
    )

    def __post_init__(self):
        self._experiment_prepared = False
        self._experiment_scores = {}
        if not hasattr(self, "score_dict"):
            self.score_dict = ScoreDict()
        else:
            self.score_dict = ScoreDict.from_payload(getattr(self, "score_dict", {}))
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
            for attr in ["_X", "_y"]
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
            self.experiment.data(
                files={
                    "data_file": None,
                    "score_file": None,
                },
            )
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
        """Render configured yellowbrick plot collection and return experiment scores.

        Returns:
            Experiment score payload captured during preparation.
        """
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
