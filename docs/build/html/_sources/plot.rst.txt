plot
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
   :undoc-members:
   :show-inheritance:
