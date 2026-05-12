Layers
======

The :mod:`deckard.layers` package exposes CLI layer parser/main pairs and the
registry used by the top-level CLI router.

.. automodule:: deckard.layers
   :members:
   :show-inheritance:

Overview
--------

Layers are thin orchestration entrypoints for higher-level tasks, such as:

- optimization runs
- result compilation
- plotting
- survival analysis
- progress monitoring

Each layer is registered in :data:`deckard.layers.layer_dict` as a
``[parser, main]`` pair consumed by the top-level CLI.

Examples
--------

.. seealso::

   Notebook-driven layer execution appears throughout:

   - :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`
   - :doc:`notebooks/seaborn.ipynb </notebooks/seaborn>`

Internals
---------

Layer functions are intentionally small wrappers that parse runtime arguments,
delegate to domain modules, and normalize outputs for CLI and automation.

Troubleshooting
---------------

- Ensure the requested subcommand exists in :data:`deckard.layers.layer_dict`.
- Check config compatibility with the selected layer.
- Verify optional dependencies for survival/plotting extensions are installed.

See also
~~~~~~~~

* :doc:`experiment`
* :doc:`plot`
* :doc:`utils`
