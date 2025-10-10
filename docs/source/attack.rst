Attack
=============

The :mod:`deckard.attack` module contains the :class:`~deckard.data.AttackConfig` dataclass and helper
functions for running evasion and inference attacks against scikit-learn
estimators using the Adversarial Robustness Toolbox (ART).

.. automodule:: deckard.attack
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------
:class:`~deckard.data.AttackConfig` provides a configurable interface for setting up and executing
adversarial attacks. It supports:

- Black-box and white-box attacks
- Membership inference and attribute inference attacks
- Customizable attack parameters
- Integration with :mod:`deckard.data` for loading datasets
- Integration with :mod:`deckard.model` for training and evaluating models
- Timing instrumentation for attack execution
- CLI support for one-line attack execution
Supported Attacks
-----------------
Currently supports a selection of attacks from ART, including:
- Evasion Attacks:
  - HopSkipJump
  - BoundaryAttack
  - AutoProjectedGradientDescent
- Membership Inference Attacks:
  - MembershipInferenceBlackBox
  - MembershipInferenceBaseline
- Attribute Inference Attacks:
   - AttributeInferenceBaseline
   - AttributeInferenceBlackBox
(Extendable to other attacks in future versions.)

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~
You can run attacks directly from the terminal:
.. code-block:: bash

   # from the project root
   python -m deckard.attack --attack_config_file blackbox_evasion --attack_size 50


Programmatic example:
~~~~~~~~~~~~~~~~~~~~
You can also use the API programmatically:

.. code-block:: python

   from deckard.attack import initialize_attack_config, AttackConfig
   from deckard.data import initialize_data_config
   from deckard.model import train_and_evaluate

   # initialize/load data
   data = initialize_data_config()
   data(filepath="path/to/data.csv")

   # train a model (example)
   _, _, model = train_and_evaluate(..., train=True, score=False, data=data)

   # create an attack config (uses defaults or overrides)
   attack_cfg = initialize_attack_config()  # or AttackConfig(attack_name="art.attacks.evasion.HopSkipJump", attack_size=50)

   # run the attack against the trained model
   results = attack_cfg(data, model)
   print(results)

Custom Configuration
~~~~~~~~~~~~~~~~~~~~
You can define a YAML file or override config parameters inline.
Example minimal YAML (`blackbox_evasion.yaml`):

.. code-block:: yaml

   _target_: attack.AttackConfig
   attack_name: art.attacks.evasion.HopSkipJump
   attack_size: 100
   attack_params:
     max_iter: 10
     max_eval: 100
     init_eval: 10
     verbose: True

Example inline overrides:

.. code-block:: bash

   python -m deckard.attack --attack_name art.attacks.evasion.HopSkipJump --attack_size 100 --attack_params.max_iter 10 --attack_params.max_eval 100 --attack_params.init_eval 10 --attack_params.verbose True

Internals
---------

Timing and logging
~~~~~~~~~~~~~~~
:class:`~deckard.data.AttackConfig` uses the `time` module to measure execution time for key steps:
- Attack setup time
- Attack execution time
- Attack prediction time
- Attack scoring time
These timings are stored as attributes (e.g. `self._attack_time`) and logged
using Python's built-in `logging` module.

Troubleshooting
---------------
If you encounter issues running attacks, ensure that:
- The specified attack name is valid and corresponds to an ART attack class.
- The model provided is compatible with the chosen attack.
- The data is properly loaded and preprocessed.
- The loaded model is trained before running attacks.


See also
~~~~~~~~
* :doc:`data`
* :doc:`model`