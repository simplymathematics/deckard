# Yellowbrick Visualization

deckard provides single-run model diagnostics through the Yellowbrick library
via {class}`deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig`. The
Yellowbrick backend operates on a composed
{class}`~deckard.experiment.ExperimentConfig` and renders visualizers directly
from the trained model and prepared dataset.

(yellowbrick-overview)=

## Overview

The {mod}`deckard.plugins.yellowbrick.plot` module provides:

- {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig` — single-plot
  config that wraps an {class}`~deckard.experiment.ExperimentConfig` with a
  Yellowbrick visualizer
- {class}`~deckard.plugins.yellowbrick.plot.YellowbrickConfigList` — ordered list
  of {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig` instances
  that share a common experiment

Yellowbrick plots are selected by ``plot_type`` and are grouped by category:

**Feature analysis** (``visualize_features``):
  ``rank1d``, ``rank2d``, ``radviz``, ``pcoords``, ``jointplot``, ``pca``,
  ``manifold``

**Target / distribution** (``visualize_targets``):
  ``class_balance``, ``balanced_binning_reference``, ``feature_correlation``

**Regression diagnostics** (``visualize_regressors``):
  ``prediction_error``, ``residuals_plot``, ``alpha_selection``

**Classification diagnostics** (``visualize_classifiers``):
  ``classification_report``, ``roc_auc``, ``precision_recall_curve``,
  ``class_prediction_error``, ``discrimination_threshold``

**Clustering** (``visualize_clusters``):
  ``k_elbow``, ``silhouette``, ``intercluster_distance``

**Model selection** (``visualize_model_selection``):
  ``learning_curve``, ``validation_curve``, ``cv_scores``,
  ``feature_importances``, ``rfecv``, ``dropping_curve``

## Examples

```{seealso}

  Notebook-based Yellowbrick visual diagnostics are documented in:

  - {doc}`notebooks/yellowbrick.ipynb </notebooks/yellowbrick>`
  - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`

```
### Troubleshooting

- **Import error**: install yellowbrick with
  ``pip install yellowbrick`` or ``pip install "deckard[plot]"``.
- **Missing experiment outputs**: call ``experiment()`` before passing it to
  {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig`; the config
  calls ``_ensure_experiment_prepared``
  lazily but explicit preparation is cleaner.
- **Unsupported plot type**: check the valid ``plot_type`` values listed in the
  *Overview* section above.
- **cv required**: model selection visualizers (``learning_curve``,
  ``validation_curve``, etc.) require ``"cv"`` in ``plot_params``.
- **Headless environments**: set ``matplotlib.use("Agg")`` before importing
  pyplot to avoid display errors in CI or server contexts.

### See also

* {doc}`plot` — general plotting documentation
* {doc}`seaborn` — multi-run aggregation visualization
* {doc}`experiment` — experiment orchestration
* {doc}`model` — model configuration and training
