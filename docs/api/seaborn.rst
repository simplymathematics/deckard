Seaborn Visualization
=====================

deckard provides statistical visualization through Seaborn via the
:class:`deckard.plot.seaborn_plots.SeabornPlotConfig` class. The Seaborn
backend is designed for **multi-run aggregation plots** — visualizing compiled
results across many experiment runs stored in a tabular data file (CSV,
Parquet, etc.).

.. _seaborn-overview:

Overview
--------

The :mod:`deckard.plot.seaborn_plots` module provides:

- :class:`~deckard.plot.seaborn_plots.SeabornPlotConfig` — single-plot
  configuration with x/y columns, plot type, and optional hue/style
- :class:`~deckard.plot.seaborn_plots.SeabornPlotConfigList` — ordered list of
  SeabornPlotConfig instances sharing a common ``data_file``

These configs are intended for post-hoc visualization of compiled experiment
results rather than single-run diagnostics.

Supported plot types
~~~~~~~~~~~~~~~~~~~~

The ``plot_type`` field accepts:

- ``scatter`` — scatter plot (``seaborn.scatterplot``)
- ``line`` — line plot (``seaborn.lineplot``)
- ``hist`` — histogram (``seaborn.histplot``)
- ``cat`` — categorical plot (``seaborn.catplot``)
- ``bar`` — bar plot (``seaborn.barplot``)
- ``heatmap`` — heatmap (``seaborn.heatmap``)

Usage
-----

Seaborn mode requires a tabular data file produced by the
``compile_results`` layer. Run ``compile_results`` first, then invoke the
``plot`` layer with ``backend=seaborn`` (or let the auto-detector choose
when you supply ``data_file``).

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Compile results then plot accuracy vs. epsilon:**

.. code-block:: bash

   # 1. Run multiple experiments (e.g. via Hydra multirun)
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name attack-default \
      --multirun attack.attack_params.eps=0.01,0.05,0.1,0.2

   # 2. Compile results into a single CSV
   python -m deckard compile_results \
      --config-path examples/sklearn/config \
      --output_file results.csv

   # 3. Plot with Seaborn backend
   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot.backend=seaborn \
      plot.data_file=results.csv \
      plot.x=attack.attack_params.eps \
      plot.y=evasion_accuracy \
      plot.plot_type=scatter \
      plot.title="Evasion Accuracy vs Epsilon" \
      plot.plot_file=plots/evasion_vs_eps.png

**Line plot with hue grouping:**

.. code-block:: bash

   python -m deckard plot \
      --config-path examples/sklearn/config \
      --config-name default \
      plot.backend=seaborn \
      plot.data_file=results.csv \
      plot.x=attack.attack_params.eps \
      plot.y=evasion_accuracy \
      plot.plot_type=line \
      plot.hue=model.model_type \
      plot.plot_file=plots/accuracy_by_model.png

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

.. seealso::

   Fully-executed programmatic examples are available in the
   :doc:`notebooks/seaborn.ipynb </notebooks/seaborn>` notebook, including scatter, bar, line, and
   plot-list patterns with rendered output.

Configuration reference
~~~~~~~~~~~~~~~~~~~~~~~

Key fields of :class:`~deckard.plot.seaborn_plots.SeabornPlotConfig`:

- **x** (str, required): column name to use as the x-axis
- **y** (str, required): column name to use as the y-axis
- **data_file** (str): path to compiled results file (CSV / Parquet / etc.);
  mutually exclusive with ``data``
- **data** (pd.DataFrame): in-memory DataFrame; mutually exclusive with
  ``data_file``
- **plot_type** (str): one of ``scatter``, ``line``, ``hist``, ``cat``,
  ``bar``, ``heatmap``
- **hue** (str, optional): column name for color grouping
- **style** (str, optional): column name for style grouping (scatter/line only)
- **title** (str, optional): plot title
- **xlabel** / **ylabel** (str, optional): axis labels
- **xscale** / **yscale** (str, optional): axis scale (``log``, ``linear``, …)
- **legend_title** (str, optional): title for the legend
- **plot_file** (str, optional): path to save the output image
- **rc_config** (dict): matplotlib rcParams overrides
- **kwargs** (dict): extra keyword arguments forwarded to the Seaborn plotter

Styling
~~~~~~~

Pass matplotlib rcParams via the ``rc_config`` field. See the
:doc:`notebooks/seaborn.ipynb </notebooks/seaborn>` notebook for a runnable example with rendered output.

Troubleshooting
~~~~~~~~~~~~~~~

- **AssertionError on column names**: verify that ``x``, ``y``, ``hue``, and
  ``style`` match column names in the data file exactly.
- **File not found**: ensure ``data_file`` path exists before constructing
  ``SeabornPlotConfig``; directories for ``plot_file`` are created automatically.
- **Import error**: install the optional plotting dependencies with
  ``pip install "deckard[plot]"``.

See also
~~~~~~~~

* :doc:`plot` — general plotting documentation
* :doc:`yellowbrick` — single-run diagnostics (ROC, confusion matrix, etc.)
* :doc:`layers` — CLI layer registry (compile_results, plot)
* :doc:`experiment` — experiment orchestration that produces scored outputs

