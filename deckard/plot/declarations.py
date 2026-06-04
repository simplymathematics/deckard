"""Plot configuration module.

Canonical plot configs are now loaded from examples/*/config/plot/ YAML files
at runtime via deckard.declarations.register_configs().

Reference dictionaries are kept below for documentation only.
"""

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

# Configs are now loaded from YAML files in examples/*/config/plot/
# These dictionaries are kept for reference/legacy code but not registered via safe_store

__all__ = [
    "PLOT_DEFAULT",
    "PLOT_TYPES",
]
