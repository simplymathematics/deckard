"""Plot configuration module.

Canonical plot configs are now loaded from examples/*/config/plot/ YAML files
at runtime via deckard.declarations.register_configs().

Reference dictionaries are kept below for documentation only.
"""

from .base import PlotTypePlugin

PLOT_DEFAULT = {
    "backend": "yellowbrick",
    "plot_type": "roc_auc",
    "plot_folder": "plots",
    "features": "all",
    "classes": "all",
    "title": "",
    "plot_params": {},
    "experiment": {
        "data": "${data}",
        "model": "${model}",
        "defense": "${defense}",
        "attack": "${attack}",
        "files": "${files}",
        "experiment_name": "${experiment_name}",
    },
}

PLOT_TYPES = [
    "alpha_selection",
    "balanced_binning_reference",
    "class_balance",
    "class_prediction_error",
    "classfication_report",
    "cv_scores",
    "discrimination_threshold",
    "dropping_curve",
    "feature_correlation",
    "feature_importances",
    "intercluster_distance",
    "jointplot",
    "k_elbow",
    "learning_curve",
    "manifold",
    "pca",
    "pcoords",
    "precision_recall_curve",
    "prediction_error",
    "radviz",
    "rank1d",
    "rank2d",
    "residuals_plot",
    "rfecv",
    "roc_auc",
    "silhouette",
    "validation_curve",
]

# Plugin Declarations
# Seaborn plotter plugin for matplotlib-based plotting
SEABORN_PLOTTER_PLUGIN = PlotTypePlugin(
    mixin_type="deckard.plugins.seaborn.plot._SeabornPlotterMixin",
    backend="seaborn",
    plot_family=None,
    init_params={
        "description": "Seaborn matplotlib backend for statistical plots",
        "supported_types": ["scatter", "line", "hist", "cat", "bar", "heatmap"],
    },
)

# Yellowbrick plotter plugin for ML model visualization
YELLOWBRICK_PLOTTER_PLUGIN = PlotTypePlugin(
    mixin_type="deckard.plugins.yellowbrick.plot._YellowbrickPlotterMixin",
    backend="yellowbrick",
    plot_family=None,
    init_params={
        "description": "Yellowbrick backend for ML model diagnostics and visualization",
        "supported_types": [
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
        ],
    },
)

# Configs are now loaded from YAML files in examples/*/config/plot/
# These dictionaries are kept for reference/legacy code but not registered via safe_store

__all__ = [
    "PLOT_DEFAULT",
    "PLOT_TYPES",
    "SEABORN_PLOTTER_PLUGIN",
    "YELLOWBRICK_PLOTTER_PLUGIN",
]
