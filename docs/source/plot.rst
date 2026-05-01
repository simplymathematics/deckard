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
- Verify input score/data files exist and are in expected schema.
- Use explicit output paths to avoid confusion in multirun directories.

See also
~~~~~~~~

* :doc:`layers`
* :doc:`experiment`
* :doc:`score`
