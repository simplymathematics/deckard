# Yellowbrick Visualization

deckard provides single-run model diagnostics through the Yellowbrick library
via {class}`deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig`. The
Yellowbrick backend operates on a composed
{class}`~deckard.experiment.ExperimentConfig` and renders visualizers directly
from the trained model and prepared dataset.

## Parent Core Modules and Behavior Deltas

Parent core pages:

- {doc}`../plot/index`
- {doc}`../experiment/index`
- {doc}`../model/index`

Behavior deltas in this integration:

- yellowbrick visualizer selection and parameterization by plot type,
- lazy experiment preparation support for single-run diagnostics,
- backend-specific rendering layered over shared plot/file contracts.

(yellowbrick-overview)=

## Overview

The {mod}`deckard.plugins.yellowbrick.plot` module provides:

- {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig` — single-plot
  config that wraps an {class}`~deckard.experiment.ExperimentConfig` with a
  Yellowbrick visualizer
- {class}`~deckard.plugins.yellowbrick.plot.YellowbrickConfigList` — ordered list
  of {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig` instances
  that share a common experiment

Yellowbrick plots are selected by `plot_type` and are grouped by category:

External references:

- [Yellowbrick documentation](https://www.scikit-yb.org)
- [`yellowbrick.classifier.ROCAUC`](https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html)
- [`yellowbrick.classifier.ClassificationReport`](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html)
- [`yellowbrick.regressor.ResidualsPlot`](https://www.scikit-yb.org/en/latest/api/regressor/residuals.html)
- [`yellowbrick.model_selection.LearningCurve`](https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html)

Related Deckard docs:

- {doc}`../model/index` for model objects consumed by visualizers
- {doc}`../score/index` for metric outputs commonly compared alongside yellowbrick plots
- {doc}`../experiment/index` for composed experiment execution prior to plotting

**Feature analysis** ({meth}[deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig.visualize_features](yellowbrick)):
- [`rank1d`](https://www.scikit-yb.org/en/latest/api/features/index.html)
- [`rank2d`](https://www.scikit-yb.org/en/latest/api/features/index.html)
- [`radviz`](https://www.scikit-yb.org/en/latest/api/features/index.html)
- [`pcoords`](https://www.scikit-yb.org/en/latest/api/features/index.html)
- [`jointplot`](https://www.scikit-yb.org/en/latest/api/features/index.html)
- [`pca`](https://www.scikit-yb.org/en/latest/api/features/index.html)
- [`manifold`](https://www.scikit-yb.org/en/latest/api/features/index.html)

**Target / distribution** ({meth}[deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig.visualize_targets](yellowbrick)):
- [`class_balance`](https://www.scikit-yb.org/en/latest/api/target/index.html)
- [`balanced_binning_reference`](https://www.scikit-yb.org/en/latest/api/target/index.html)
- [`feature_correlation`](https://www.scikit-yb.org/en/latest/api/target/index.html)

**Regression diagnostics** ({meth}[deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig.visualize_regressors](yellowbrick)):
- [`prediction_error`](https://www.scikit-yb.org/en/latest/api/regressor/index.html)
- [`residuals_plot`](https://www.scikit-yb.org/en/latest/api/regressor/index.html)
- [`alpha_selection`](https://www.scikit-yb.org/en/latest/api/regressor/index.html)

**Classification diagnostics** ({meth}[deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig.visualize_classifiers](yellowbrick)):
- [`classification_report`](https://www.scikit-yb.org/en/latest/api/classifier/index.html)
- [`roc_auc`](https://www.scikit-yb.org/en/latest/api/classifier/index.html)
- [`precision_recall_curve`](https://www.scikit-yb.org/en/latest/api/classifier/index.html)
- [`class_prediction_error`](https://www.scikit-yb.org/en/latest/api/classifier/index.html)
- [`discrimination_threshold`](https://www.scikit-yb.org/en/latest/api/classifier/index.html)

**Clustering** ({meth}[deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig.visualize_clusters](yellowbrick)):
- [`k_elbow`](https://www.scikit-yb.org/en/latest/api/cluster/index.html)
- [`silhouette`](https://www.scikit-yb.org/en/latest/api/cluster/index.html)
- [`intercluster_distance`](https://www.scikit-yb.org/en/latest/api/cluster/index.html)

**Model selection** ({meth}[deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig.visualize_model_selection](yellowbrick)):
- [`learning_curve`](https://www.scikit-yb.org/en/latest/api/model_selection/index.html)
- [`validation_curve`](https://www.scikit-yb.org/en/latest/api/model_selection/index.html)
- [`cv_scores`](https://www.scikit-yb.org/en/latest/api/model_selection/index.html)
- [`feature_importances`](https://www.scikit-yb.org/en/latest/api/model_selection/index.html)
- [`rfecv`](https://www.scikit-yb.org/en/latest/api/model_selection/index.html)
- [`dropping_curve`](https://www.scikit-yb.org/en/latest/api/model_selection/index.html)

## Examples

```{seealso}

  Notebook-based Yellowbrick visual diagnostics are documented in:

  - {doc}`notebooks/yellowbrick.ipynb </notebooks/yellowbrick>`
  - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`

```

### Troubleshooting

- **Import error**: install yellowbrick with
  `pip install yellowbrick` or `pip install "deckard[plot]"`.
- **Missing experiment outputs**: call `experiment()` before passing it to
  {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig`; the config
  calls `_ensure_experiment_prepared`
  lazily but explicit preparation is cleaner.
- **Unsupported plot type**: check the valid `plot_type` values listed in the
  *Overview* section above.
- **cv required**: model selection visualizers (`learning_curve`,
  `validation_curve`, etc.) require `"cv"` in `plot_params`.
- **Headless environments**: set `matplotlib.use("Agg")` before importing
  pyplot to avoid display errors in CI or server contexts.

### See also

- {doc}`../plot/index` — general plotting documentation
- {doc}`/api/plugins/seaborn` — multi-run aggregation visualization
- {doc}`../experiment/index` — experiment orchestration
- {doc}`../model/index` — model configuration and training
