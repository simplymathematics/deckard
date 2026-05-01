File
====

The :mod:`deckard.file` module defines path configuration primitives used by
Deckard pipelines to persist datasets, models, scores, predictions, and logs.

.. automodule:: deckard.file
   :members:
   :show-inheritance:

Overview
--------

The file layer centralizes artifact naming and output-path management across
experiment components. It helps keep runs reproducible and avoids ad-hoc path
construction in data/model/attack code.

Usage
-----

Programmatic example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deckard.file import FileConfig

   files = FileConfig()
   print(files.score_file)
   print(files.model_file)

Internals
---------

The module provides helpers for deterministic path construction and job-aware
output naming (for example in Hydra multirun contexts).

Troubleshooting
---------------

- Confirm the working directory and configured output folders are writable.
- Validate that generated parent directories exist when running outside default layouts.
- Use absolute paths in config for external storage locations.

See also
~~~~~~~~

* :doc:`experiment`
* :doc:`data`
* :doc:`model`
* :doc:`attack`
* :doc:`score`
