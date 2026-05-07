Yellowbrick Visualization
=========================

Deckard provides single-run model diagnostics through the Yellowbrick library
via :class:`deckard.plot.yellowbrick_plots.YellowbrickPlotConfig`. The
Yellowbrick backend operates on a composed
:class:`~deckard.experiment.ExperimentConfig` and renders visualizers directly
from the trained model and prepared dataset.

.. _yellowbrick-overview:

Overview
--------

The :mod:`deckard.plot.yellowbrick_plots` module provides:

- :class:`~deckard.plot.yellowbrick_plots.YellowbrickPlotConfig` — single-plot
  config that wraps an :class:`~deckard.experiment.ExperimentConfig` with a
  Yellowbrick visualizer
- :class:`~deckard.plot.yellowbrick_plots.YellowbrickConfigList` — ordered list
  of YellowbrickPlotConfig instances that share a common experiment

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

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

The recommended workflow is to select a plot type from the presets in
``examples/sklearn/config/plot/``:

.. code-block:: bash

   # ROC-AUC visualization
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=roc_auc

   # Classification report
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=classfication_report

   # Precision-recall curve
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=precision_recall_curve

   # Learning curve (requires cv and param_range in plot_params)
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=learning_curve

   # Feature importance
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=feature_importances

   # Override model and plot type together
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      model.model_type=sklearn.ensemble.GradientBoostingClassifier \
      plot=roc_auc

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

**ROC-AUC for a classifier:**

.. code-block:: python

   from deckard.data import DataConfig
   from deckard.experiment import ExperimentConfig
   from deckard.model import ModelConfig
   from deckard.plot.yellowbrick_plots import YellowbrickPlotConfig

   data = DataConfig(
       dataset_name="make_classification",
       data_params={"n_samples": 200, "n_features": 10, "n_classes": 2},
       train_size=150,
       test_size=50,
       classifier=True,
   )
   model = ModelConfig(
       model_type="sklearn.linear_model.LogisticRegression",
       classifier=True,
       model_params={"max_iter": 100},
   )
   experiment = ExperimentConfig(data=data, model=model)
   experiment()  # train model and prepare data

   plot = YellowbrickPlotConfig(
       experiment=experiment,
       plot_type="roc_auc",
       save_path="plots/roc_auc.png",
   )
   plot()

**Classification report:**

.. code-block:: python

   from deckard.plot.yellowbrick_plots import YellowbrickPlotConfig

   plot = YellowbrickPlotConfig(
       experiment=experiment,
       plot_type="classification_report",
       save_path="plots/classification_report.png",
   )
   plot()

**Feature importance (tree models only):**

.. code-block:: python

   from deckard.data import DataConfig
   from deckard.experiment import ExperimentConfig
   from deckard.model import ModelConfig
   from deckard.plot.yellowbrick_plots import YellowbrickPlotConfig

   data = DataConfig(
       dataset_name="make_classification",
       data_params={"n_samples": 200, "n_features": 10},
       train_size=150,
       test_size=50,
       classifier=True,
   )
   model = ModelConfig(
       model_type="sklearn.ensemble.RandomForestClassifier",
       classifier=True,
       model_params={"n_estimators": 50},
   )
   experiment = ExperimentConfig(data=data, model=model)
   experiment()

   plot = YellowbrickPlotConfig(
       experiment=experiment,
       plot_type="feature_importances",
       save_path="plots/feature_importance.png",
       plot_params={"cv": 3},  # cv is required for model selection plots
   )
   plot()

**Learning curve:**

.. code-block:: python

   from deckard.plot.yellowbrick_plots import YellowbrickPlotConfig

   plot = YellowbrickPlotConfig(
       experiment=experiment,
       plot_type="learning_curve",
       save_path="plots/learning_curve.png",
       plot_params={
           "cv": 5,
           "param_range": [0.1, 1.0],  # train_sizes range
       },
   )
   plot()

**PCA decomposition:**

.. code-block:: python

   from deckard.plot.yellowbrick_plots import YellowbrickPlotConfig

   plot = YellowbrickPlotConfig(
       experiment=experiment,
       plot_type="pca",
       save_path="plots/pca.png",
   )
   plot()

Configuration reference
~~~~~~~~~~~~~~~~~~~~~~~

Key fields of :class:`~deckard.plot.yellowbrick_plots.YellowbrickPlotConfig`:

- **experiment** (:class:`~deckard.experiment.ExperimentConfig`, required):
  the experiment providing model, data, and (optionally) attack outputs
- **plot_type** (str, required): the Yellowbrick visualizer to render; see
  *Overview* for valid values
- **save_path** (str): output file path for the rendered plot (default:
  ``yellowbrick_plot.png``)
- **features** (list | ``"all"``): feature subset to include (default: ``"all"``)
- **classes** (list | ``"all"``): class subset for classifier visualizers
  (default: ``"all"``)
- **title** (str): plot title (default: ``"Yellowbrick Plot"``)
- **plot_params** (dict): extra kwargs forwarded to the Yellowbrick visualizer
  constructor; required keys vary by ``plot_type`` (e.g. ``cv`` for model
  selection plots, ``param_range`` for ``validation_curve``)
- **rc_config** (dict): matplotlib rcParams overrides applied before rendering

``plot_params`` notes
^^^^^^^^^^^^^^^^^^^^^

- ``learning_curve``: requires ``{"cv": <int or dict>}``;
  optionally ``{"param_range": [start, end]}``.
- ``validation_curve``: requires ``{"cv": <int or dict>, "param_name": <str>,
  "param_range": [start, end]}``.
- ``cv_scores``: requires ``{"cv": <int or dict>}``.
- ``rfecv`` / ``dropping_curve``: require ``{"cv": <int or dict>}``.
- ``jointplot``: requires ``{"columns": [col_a, col_b]}``.

YAML preset examples
~~~~~~~~~~~~~~~~~~~~

Each preset in ``examples/sklearn/config/plot/`` is a self-contained plot
config. For example, ``roc_auc.yaml``:

.. code-block:: yaml

   backend: yellowbrick
   plot_type: roc_auc
   plot_folder: plots
   features: all
   classes: all
   title: ""
   plot_params: {}
   experiment:
     data: ${data}
     model: ${model}
     defense: ${defense}
     attack: ${attack}
     files: ${files}
     experiment_name: ${experiment_name}

Create a custom preset by copying ``default.yaml`` and overriding ``plot_type``
and ``plot_params``.

Troubleshooting
~~~~~~~~~~~~~~~

- **Import error**: install yellowbrick with
  ``pip install yellowbrick`` or ``pip install "deckard[plot]"``.
- **Missing experiment outputs**: call ``experiment()`` before passing it to
  ``YellowbrickPlotConfig``; the config calls ``_ensure_experiment_prepared``
  lazily but explicit preparation is cleaner.
- **Unsupported plot type**: check the valid ``plot_type`` values listed in the
  *Overview* section above.
- **cv required**: model selection visualizers (``learning_curve``,
  ``validation_curve``, etc.) require ``"cv"`` in ``plot_params``.
- **Headless environments**: set ``matplotlib.use("Agg")`` before importing
  pyplot to avoid display errors in CI or server contexts.

See also
~~~~~~~~

* :doc:`plot` — general plotting documentation
* :doc:`seaborn` — multi-run aggregation visualization
* :doc:`experiment` — experiment orchestration
* :doc:`model` — model configuration and training

