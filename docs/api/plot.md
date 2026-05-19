# Plot

## Plotting APIs

The plotting package exposes two public entry points:

- :class:`~deckard.plot.PlotConfig` chooses between the Seaborn and Yellowbrick backends.
- :class:`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig` and
   :class:`~deckard.plugins.yellowbrick.plot.YellowbrickConfigList` behave like
   experiment configs and prepare experiment outputs at most once before
   rendering plots.

.. automodule:: deckard.plot
   :members:
   :undoc-members:
   :show-inheritance:

## Survival Plot Extension

Survival plotting configs are provided in a dedicated optional module.
See also: :doc:`lifelines`.

.. automodule:: deckard.plot.survival
   :members:
   :show-inheritance:

.. automodule:: deckard.plugins.yellowbrick.plot
   :members:
   :show-inheritance:

## Overview

Plot configs separate plotting intent from execution details. They support:

- Seaborn-driven exploratory and report plots
- Yellowbrick visualizers for model diagnostics
- shared style and rc parameter configuration
- optional experiment-aware setup for plotting from prior outputs

## Examples

.. seealso::

   Notebook-based plotting examples for Yellowbrick and Seaborn are documented
   in:

   - :doc:`notebooks/yellowbrick.ipynb </notebooks/yellowbrick>`
   - :doc:`notebooks/seaborn.ipynb </notebooks/seaborn>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`

## Internals

The plotting module routes to backend-specific config objects and ensures
output files are written consistently. Yellowbrick plotting can hydrate
experiment context lazily before rendering to avoid repeated setup.

## Troubleshooting

- Confirm plotting dependencies are installed for the selected backend.
- For Yellowbrick presets, ensure the selected ``plot=<name>`` exists under
   `examples/sklearn/config/plot <../examples/sklearn/config/plot>`_.
- For Seaborn, ensure ``data_file`` exists and that ``x``/``y``/``hue`` columns
   are present in that dataset.
- Verify input score/data files exist and are in expected schema.
- Use explicit output paths to avoid confusion in multirun directories.

### See also

* :doc:`experiment` — experiment orchestration and result generation
* :doc:`score` — scoring framework that produces plotting data
* :doc:`seaborn` — statistical visualization with Seaborn
* :doc:`yellowbrick` — model interpretability visualizations
* :doc:`layers` — advanced workflows
