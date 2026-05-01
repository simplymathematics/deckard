Experiment
==========

The :mod:`deckard.experiment` module contains the high-level orchestration
entrypoints for end-to-end experiment execution.

.. automodule:: deckard.experiment
   :members:
   :show-inheritance:

Survival Extension
------------------

Survival-specific experiment orchestration is split into a dedicated optional
module.

.. automodule:: deckard.experiment.survival
   :members:
   :show-inheritance:

Overview
--------

The experiment layer coordinates the full Deckard workflow by composing:

- data loading and sampling via :mod:`deckard.data`
- model training/evaluation via :mod:`deckard.model`
- optional attack execution via :mod:`deckard.attack`
- score aggregation and file outputs via :mod:`deckard.file`

It is the primary integration point for reproducible end-to-end runs.

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~

Run an experiment from the project root:

.. code-block:: bash

   python -m deckard optimize --config-name experiment

Programmatic example
~~~~~~~~~~~~~~~~~~~~

Use the experiment config directly in Python:

.. code-block:: python

   from deckard.experiment import ExperimentConfig

   cfg = ExperimentConfig()
   scores = cfg()
   print(scores)

Internals
---------

The module resolves nested config objects, applies runtime overrides, and
normalizes outputs for downstream scoring/serialization.

Troubleshooting
---------------

- Verify config paths and override keys when Hydra/OmegaConf resolution fails.
- Ensure optional dependencies are installed for selected model/attack backends.
- Check file output paths in :class:`deckard.file.FileConfig` if artifacts are missing.

See also
~~~~~~~~

* :doc:`data`
* :doc:`model`
* :doc:`attack`
* :doc:`file`
* :doc:`score`
* :doc:`utils`
