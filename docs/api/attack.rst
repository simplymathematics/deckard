Attack
======

Overview
--------

The attack module orchestrates adversarial example generation across supported
backends and attack families.

It provides:

- attack configuration and instantiation
- attack execution over model/data outputs
- artifact persistence for attacked samples and labels
- attack-aware scoring hooks used by experiment orchestration

Examples
--------

.. seealso::

   Notebook-based attack workflows are documented in:

   - :doc:`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`

API Reference
-------------

.. automodule:: deckard.attack
   :members:
   :show-inheritance:

.. automodule:: deckard.attack.base
   :members:
   :show-inheritance:

.. automodule:: deckard.attack.pytorch
   :members:
   :show-inheritance:

Troubleshooting
---------------

- Ensure the selected attack backend matches the active model backend.
- Confirm attack parameters are valid for the chosen ART/Fairlearn attack type.
- Verify the attack receives compatible input shapes and labels.

See also
~~~~~~~~

* :doc:`experiment` — experiment orchestration
* :doc:`model` — model configuration and execution
* :doc:`data` — data loading and split handling
* :doc:`score` — attack scoring profiles
