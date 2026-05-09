Yellowbrick Visualization
=========================

deckard provides single-run model diagnostics through the Yellowbrick library
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
`examples/sklearn/config/plot/ <../examples/sklearn/config/plot/>`_:

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

.. seealso::

   Fully-executed programmatic examples are available in the
   :doc:`notebooks/yellowbrick.ipynb </notebooks/yellowbrick>` notebook, including ROC-AUC, classification
   report, feature importance, learning curve, and PCA plots with rendered output.

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

Each preset in `examples/sklearn/config/plot/ <../examples/sklearn/config/plot/>`_ is a self-contained plot
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

