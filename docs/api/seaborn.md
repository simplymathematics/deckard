# Seaborn Visualization

deckard provides statistical visualization through Seaborn via the
:class:`deckard.plot.seaborn_plots.SeabornPlotConfig` class. The Seaborn
backend is designed for **multi-run aggregation plots** — visualizing compiled
results across many experiment runs stored in a tabular data file (CSV,
Parquet, etc.).

.. _seaborn-overview:

## Overview

The :mod:`deckard.plot.seaborn_plots` module provides:

- :class:`~deckard.plot.seaborn_plots.SeabornPlotConfig` — single-plot
  configuration with x/y columns, plot type, and optional hue/style
- :class:`~deckard.plot.seaborn_plots.SeabornPlotConfigList` — ordered list of
  :class:`~deckard.plot.seaborn_plots.SeabornPlotConfig` instances sharing a
  common ``data_file``

These configs are intended for post-hoc visualization of compiled experiment
results rather than single-run diagnostics.

### Supported plot types

The ``plot_type`` field accepts:

- ``scatter`` — scatter plot (``seaborn.scatterplot``)
- ``line`` — line plot (``seaborn.lineplot``)
- ``hist`` — histogram (``seaborn.histplot``)
- ``cat`` — categorical plot (``seaborn.catplot``)
- ``bar`` — bar plot (``seaborn.barplot``)
- ``heatmap`` — heatmap (``seaborn.heatmap``)

## Examples

.. seealso::

  Notebook-based Seaborn plotting workflows are documented in:

  - :doc:`notebooks/seaborn.ipynb </notebooks/seaborn>`
  - :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>`

### Troubleshooting

- **AssertionError on column names**: verify that ``x``, ``y``, ``hue``, and
  ``style`` match column names in the data file exactly.
- **File not found**: ensure ``data_file`` path exists before constructing
  :class:`~deckard.plot.seaborn_plots.SeabornPlotConfig`; directories for
  ``plot_file`` are created automatically.
- **Import error**: install the optional plotting dependencies with
  ``pip install "deckard[plot]"``.

### See also

* :doc:`plot` — general plotting documentation
* :doc:`yellowbrick` — single-run diagnostics (ROC, confusion matrix, etc.)
* :doc:`layers` — CLI layer registry (compile_results, plot)
* :doc:`experiment` — experiment orchestration that produces scored outputs
