File
====

Overview
--------

The :mod:`deckard.file` module handles persistence for artifacts produced
throughout deckard runs.

It provides helpers for:

- output path resolution
- score/result serialization
- model and data artifact management
- run directory organization

Examples
--------

.. seealso::

   Notebook-based file/artifact workflows are documented in:

   - :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`

API Reference
-------------

.. automodule:: deckard.file
   :members:
   :show-inheritance:

Typical Workflow
----------------

1. Configure file outputs through the active experiment config.
2. Execute experiment/model/attack/score layers.
3. Persist and reload artifacts via file config helpers.

Troubleshooting
---------------

- Ensure output directories are writable.
- Verify artifact paths are consistent across experiment and layer configs.
- Check that expected file formats match the configured save/load behavior.

See also
~~~~~~~~

* :doc:`experiment` — experiment orchestration
* :doc:`data` — dataset artifacts
* :doc:`model` — model artifacts
* :doc:`score` — score persistence and loading
