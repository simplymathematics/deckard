Attack
=============

The :mod:`deckard.attack` module contains the :class:`~deckard.attack.AttackConfig` dataclass and helper
functions for running evasion and inference attacks against scikit-learn
estimators using the Adversarial Robustness Toolbox (ART).

.. automodule:: deckard.attack
   :members:
   :show-inheritance:

Overview
--------
:class:`~deckard.attack.AttackConfig` provides a configurable interface for setting up and executing
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
Evasion Attacks:
- HopSkipJump
- BoundaryAttack
- AutoProjectedGradientDescent
Membership Inference Attacks:
- MembershipInferenceBlackBox
- MembershipInferenceBaseline
Attribute Inference Attacks:
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
   python -m deckard optimize --config-name experiment \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_params.eps=0.1 \
      attack.attack_size=20


Programmatic example:
~~~~~~~~~~~~~~~~~~~~~~
You can also use the API programmatically:

.. code-block:: python

   from deckard.attack import AttackConfig
   from deckard.data import DataConfig
   from deckard.model import ModelConfig

   data = DataConfig(
      dataset_name="make_classification",
      data_params={
         "n_samples": 60,
         "n_features": 10,
         "n_informative": 4,
         "n_redundant": 0,
         "n_clusters_per_class": 1,
         "n_classes": 2,
         "random_state": 7,
      },
      train_size=40,
      test_size=20,
      random_state=42,
      stratify=True,
      classifier=True,
   )
   data()

   model = ModelConfig(
      model_type="sklearn.linear_model.LogisticRegression",
      classifier=True,
      model_params={"max_iter": 25},
   )
   model(data)

   attack_cfg = AttackConfig(
      attack_type="art.attacks.evasion.FastGradientMethod",
      attack_params={"eps": 0.1},
      attack_size=20,
   )

   # run the attack against the trained model
   scores = attack_cfg(data=data, model=model)
   print([k for k in scores if k.startswith("evasion_")])

BoundaryAttack example
~~~~~~~~~~~~~~~~~~~~~~

The fairness integration test exercises a small BoundaryAttack configuration:

.. code-block:: python

   boundary_attack = AttackConfig(
      attack_type="art.attacks.evasion.BoundaryAttack",
      attack_params={
         "batch_size": 5,
         "targeted": False,
         "delta": 0.01,
         "epsilon": 0.01,
         "max_iter": 2,
         "num_trial": 5,
         "sample_size": 5,
         "init_size": 5,
         "min_epsilon": 0.0,
         "verbose": False,
      },
      attack_size=5,
   )

Custom Configuration
~~~~~~~~~~~~~~~~~~~~
You can define a YAML file or override config parameters inline.
Example minimal YAML (`blackbox_evasion.yaml`):

.. code-block:: yaml

   _target_: deckard.attack.AttackConfig
   attack_type: art.attacks.evasion.FastGradientMethod
   attack_size: 20
   attack_params:
     eps: 0.1

Example inline overrides:

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_size=20 \
      attack.attack_params.eps=0.1

Internals
---------

Timing and logging
~~~~~~~~~~~~~~~~~~~~
:class:`~deckard.attack.AttackConfig` uses the `time` module to measure execution time for key steps:
- Attack setup time
- Attack execution time
- Attack prediction time
- Attack scoring time
These timings are stored as attributes (e.g. `self._attack_time`) and logged
using Python's built-in `logging` module.

Troubleshooting
---------------
If you encounter issues running attacks, ensure that:
- The specified attack type is valid and corresponds to an ART attack class.
- The model provided is compatible with the chosen attack.
- The data is properly loaded and preprocessed.
- The loaded model is trained before running attacks.


See also
~~~~~~~~
* :doc:`data`
* :doc:`model`
* :doc:`experiment`
* :doc:`utils`
