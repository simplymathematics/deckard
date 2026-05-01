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

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~

Invoke a registered layer through the top-level CLI:

.. code-block:: bash

   python -m deckard optimize --config-name experiment

Programmatic example
~~~~~~~~~~~~~~~~~~~~

Access layer dispatch metadata directly:

.. code-block:: python

   from deckard.layers import layer_dict, SUPPORTED_LAYERS

   print(SUPPORTED_LAYERS)
   parser_fn, main_fn = layer_dict["optimize"]
   print(parser_fn, main_fn)

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
