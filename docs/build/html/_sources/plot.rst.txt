Plot
====

Plotting APIs
-------------

The plotting package exposes two public entry points:

- ``deckard.plot.PlotConfig`` chooses between the Seaborn and Yellowbrick backends.
- ``deckard.plot.yellowbrick_plots.YellowbrickPlotConfig`` and ``deckard.plot.yellowbrick_plots.YellowbrickConfigList`` behave like experiment configs and prepare experiment outputs at most once before rendering plots.

.. automodule:: deckard.plot
   :members:
   :undoc-members:
   :show-inheritance:

Survival Plot Extension
-----------------------

Survival plotting configs are provided in a dedicated optional module.

.. automodule:: deckard.plot.survival
   :members:
   :show-inheritance:

.. automodule:: deckard.plot.yellowbrick_plots
   :members:
   :show-inheritance:

Overview
--------

Plot configs separate plotting intent from execution details. They support:

- Seaborn-driven exploratory and report plots
- Yellowbrick visualizers for model diagnostics
- shared style and rc parameter configuration
- optional experiment-aware setup for plotting from prior outputs

Usage
-----

Yellowbrick Examples (Smoke-Tested)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deckard ships a broad set of Yellowbrick plot presets under
``examples/sklearn/config/plot``. These are designed to be selected through
Hydra's ``plot=<name>`` override when running from
``examples/sklearn/config/default.yaml``.

Frequently used examples include:

- ``plot=roc_auc`` (``roc_auc.yaml``)
- ``plot=precision_recall_curve`` (``precision_recall_curve.yaml``)
- ``plot=classfication_report`` (``classfication_report.yaml``)
- ``plot=class_balance`` (``class_balance.yaml``)
- ``plot=feature_importances`` / ``plot=feature_correlation``
- ``plot=learning_curve`` / ``plot=validation_curve``

Example commands:

.. code-block:: bash

   # ROC-AUC visualization
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=roc_auc

   # Classification report visualization
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=classfication_report

   # Learning-curve visualization
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      plot=learning_curve

Yellowbrick + ExperimentConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yellowbrick plotting is experiment-oriented. In practice, a
:class:`deckard.plot.yellowbrick_plots.YellowbrickPlotConfig` receives an
``experiment`` payload (or a ``YellowbrickConfigList`` of multiple plots), then
uses that :class:`deckard.experiment.ExperimentConfig` object to ensure model and
dataset artifacts are prepared before rendering visual diagnostics.

The default plot config in ``examples/sklearn/config/plot/default.yaml`` shows
the expected shape:

.. code-block:: yaml

   backend: yellowbrick
   plot_type: roc_auc
   experiment:
     data: ${data}
     model: ${model}
     defense: ${defense}
     attack: ${attack}
     files: ${files}
     experiment_name: ${experiment_name}

Base programmatic example:

.. code-block:: python

   from deckard.plot import PlotConfig
   from deckard.experiment import ExperimentConfig
   from deckard.data import DataConfig
   from deckard.model import ModelConfig
   from deckard.file import FileConfig

   exp = ExperimentConfig(
      data=DataConfig(dataset_name="adult", test_size=0.2, classifier=True),
      model=ModelConfig(
         model_type="sklearn.ensemble.RandomForestClassifier",
         classifier=True,
      ),
      files=FileConfig(),
      classifier=True,
   )

   plot_cfg = PlotConfig(
      backend="yellowbrick",
      plot_type="roc_auc",
      experiment=exp,
      plot_folder="plots",
   )
   plot_cfg()

Attack support status for Yellowbrick: attack visualizers are not currently
supported in the Yellowbrick plotting path. Yellowbrick plots operate on model
and dataset experiment outputs; attack-specific plots should be treated as a
future extension.

Seaborn Support
~~~~~~~~~~~~~~~

Seaborn plotting is supported through
:class:`deckard.plot.seaborn_plots.SeabornPlotConfig` and
:class:`deckard.plot.PlotConfig` when ``backend=seaborn`` is selected.

Unlike Yellowbrick presets, Seaborn plots are usually configured by specifying
``plot_type`` (``scatter``, ``line``, ``hist``, ``cat``, ``bar``, ``heatmap``),
``x``/``y`` columns, and a ``data_file``.

Programmatic Seaborn example:

.. code-block:: python

   from deckard.plot import PlotConfig

   cfg = PlotConfig(
      backend="seaborn",
      plot_type="scatter",
      data_file="outputs/scores.csv",
      x="accuracy",
      y="evasion_accuracy",
      hue="defense_alias",
      title="Accuracy vs Evasion Accuracy",
      plot_file="plots/scatter_accuracy_vs_evasion.png",
   )
   cfg()

You can also define a Seaborn plot directly in YAML:

.. code-block:: yaml

   backend: seaborn
   plot_type: scatter
   data_file: outputs/scores.csv
   x: accuracy
   y: evasion_accuracy
   hue: defense_alias
   title: Accuracy vs Evasion Accuracy
   plot_file: plots/scatter_accuracy_vs_evasion.png

Programmatic example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deckard.plot import PlotConfig

   cfg = PlotConfig()
   cfg()

Internals
---------

The plotting module routes to backend-specific config objects and ensures
output files are written consistently. Yellowbrick plotting can hydrate
experiment context lazily before rendering to avoid repeated setup.

Troubleshooting
---------------

- Confirm plotting dependencies are installed for the selected backend.
- For Yellowbrick presets, ensure the selected ``plot=<name>`` exists under
   ``examples/sklearn/config/plot``.
- For Seaborn, ensure ``data_file`` exists and that ``x``/``y``/``hue`` columns
   are present in that dataset.
- Verify input score/data files exist and are in expected schema.
- Use explicit output paths to avoid confusion in multirun directories.

See also
~~~~~~~~

* :doc:`experiment` — experiment orchestration and result generation
* :doc:`score` — scoring framework that produces plotting data
* :doc:`seaborn` — statistical visualization with Seaborn
* :doc:`yellowbrick` — model interpretability visualizations
* :doc:`layers` — advanced workflows
