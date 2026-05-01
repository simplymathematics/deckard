Score
=====

The :mod:`deckard.score` module defines scorer configuration objects used by
model, attack, and experiment pipelines.

.. automodule:: deckard.score
   :members:
   :show-inheritance:

Overview
--------

The score layer provides configurable scorer wrappers so data/model/attack
components can use a consistent scoring interface without hard-coding metric
implementations.

Usage
-----

Programmatic example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deckard.score import ScorerDictConfig

   scorers = ScorerDictConfig()
   callables = scorers.get_callables()
   print(sorted(callables.keys()))

Internals
---------

Score configs normalize definitions into callable maps and support both
classification and regression defaults through dedicated config classes.

Troubleshooting
---------------

- Ensure metric names map to importable scorer callables.
- Check expected prediction shape/type for selected metrics.
- Confirm task type (classifier/regressor) matches the active default scorer set.

See also
~~~~~~~~

* :doc:`model`
* :doc:`attack`
* :doc:`experiment`
