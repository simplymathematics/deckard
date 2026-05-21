# Seaborn Visualization

deckard provides statistical visualization through Seaborn via the
{class}`deckard.plugins.seaborn.plot.SeabornPlotConfig` class. The Seaborn
backend is designed for **multi-run aggregation plots** — visualizing compiled
results across many experiment runs stored in a tabular data file (CSV,
Parquet, etc.).

(seaborn-overview)=

## Overview

The {mod}`deckard.plugins.seaborn.plot` module provides:

- {class}`~deckard.plugins.seaborn.plot.SeabornPlotConfig` — single-plot
  configuration with x/y columns, plot type, and optional hue/style
- {class}`~deckard.plugins.seaborn.plot.SeabornPlotConfigList` — ordered list of
  {class}`~deckard.plugins.seaborn.plot.SeabornPlotConfig` instances sharing a
  common `data_file`

These configs are intended for post-hoc visualization of compiled experiment
results rather than single-run diagnostics.

External references:

- [Seaborn documentation](https://seaborn.pydata.org)
- [`seaborn.scatterplot`](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)
- [`seaborn.lineplot`](https://seaborn.pydata.org/generated/seaborn.lineplot.html)
- [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

Related Deckard docs:

- {doc}`score` for how plotted fields are produced by scorer configs
- {doc}`layers` for compiled-results and plotting layer orchestration

### Supported plot types

The `plot_type` field accepts:

- `scatter` — scatter plot (`seaborn.scatterplot`)
- `line` — line plot (`seaborn.lineplot`)
- `hist` — histogram (`seaborn.histplot`)
- `cat` — categorical plot (`seaborn.catplot`)
- `bar` — bar plot (`seaborn.barplot`)
- `heatmap` — heatmap (`seaborn.heatmap`)

## Examples

```{seealso}

  Notebook-based Seaborn plotting workflows are documented in:

  - {doc}`notebooks/seaborn.ipynb </notebooks/seaborn>`
  - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`

```

### Troubleshooting

- **AssertionError on column names**: verify that `x`, `y`, `hue`, and
  `style` match column names in the data file exactly.
- **File not found**: ensure `data_file` path exists before constructing
  {class}`~deckard.plugins.seaborn.plot.SeabornPlotConfig`; directories for
  `plot_file` are created automatically.
- **Import error**: install the optional plotting dependencies with
  `pip install "deckard[plot]"`.

### See also

- {doc}`plot` — general plotting documentation
- {doc}`yellowbrick` — single-run diagnostics (ROC, confusion matrix, etc.)
- {doc}`layers` — CLI layer registry (compile_results, plot)
- {doc}`experiment` — experiment orchestration that produces scored outputs
