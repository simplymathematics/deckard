"""Static plot configuration declarations and ConfigStore registrations."""

from ..utils import safe_store

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


safe_store(group="plot", name="default", node=PLOT_DEFAULT)
for _plot_name in PLOT_TYPES:
    safe_store(
        group="plot",
        name=_plot_name,
        node={
            "defaults": ["default"],
            "plot_type": _plot_name,
        },
    )
