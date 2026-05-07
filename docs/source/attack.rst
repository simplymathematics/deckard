Attack
=============

The :mod:`deckard.attack` module contains the :class:`~deckard.attack.AttackConfig` dataclass and helper
functions for running evasion, poisoning, extraction, and inference attacks
using the Adversarial Robustness Toolbox (ART).

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
Deckard supports ART attacks across the following families:

- **Evasion attacks** (for example: ``FastGradientMethod``, ``HopSkipJump``,
   ``BoundaryAttack``, ``AutoProjectedGradientDescent``)
- **Poisoning attacks** (for example: ``GradientMatchingAttack``)
- **Extraction attacks** (for example: ART extraction attacks against neural
   classifiers)
- **Inference attacks**:

   - Membership inference (for example: ``MembershipInferenceBlackBox``,
      ``MembershipInferenceBaseline``)
   - Attribute inference (for example: ``AttributeInferenceBaseline``,
      ``AttributeInferenceBlackBox``)
   - Model inversion (for example: ``MIFace``)
   - Database reconstruction (for example: ``DatabaseReconstruction``)

(Extendable to additional ART attack classes in future versions.)

Preset Catalog (examples/sklearn)
---------------------------------

Deckard ships ready-to-run attack presets in ``examples/sklearn/config/attack``
and search-space definitions in ``examples/sklearn/config/search/attacks``.

Common presets include:

- ``boundary``
- ``fgm``
- ``hsj``
- ``membership``
- ``attribute-bb``
- ``model-inversion``
- ``database-reconstruction``
- ``zoo``

Example command selecting a preset:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      attack=hsj

Attack Types And Metrics
------------------------
The output score dictionary depends on the attack family and task type. Metric
keys are prefixed to make results easy to group in downstream analysis.

**Evasion (classification)**

- Prefix: ``evasion_``
- Default metrics: ``evasion_accuracy``, ``evasion_precision``,
   ``evasion_recall``, ``evasion_f1-score``, ``evasion_success``
- ``evasion_success`` is computed as $1 - \text{accuracy}(\text{benign preds}, \text{adversarial preds})$
- Baseline metrics are also emitted with ``benign_`` prefix for consistency,
   for example: ``benign_accuracy``, ``benign_precision``, ``benign_recall``,
   ``benign_f1``

**Evasion (regression)**

- Prefix: ``evasion_``
- Default metrics: ``evasion_mse``, ``evasion_mae``, ``evasion_r2``
- Baseline metrics are also emitted with ``benign_`` prefix, for example:
   ``benign_mse``, ``benign_mae``, ``benign_r2``

**Membership inference**

- Prefix: ``membership_inference_``
- Default metrics: ``membership_inference_accuracy``,
   ``membership_inference_precision``, ``membership_inference_recall``,
   ``membership_inference_f1``

**Attribute inference (classification target attribute)**

- Prefix: ``inferred_<targeted_attribute>_``
- Default metrics: ``inferred_<targeted_attribute>_accuracy``,
   ``inferred_<targeted_attribute>_precision``,
   ``inferred_<targeted_attribute>_recall``,
   ``inferred_<targeted_attribute>_f1``

**Attribute inference (regression target attribute)**

- Prefix: ``inferred_<targeted_attribute>_``
- Default metrics: ``inferred_<targeted_attribute>_mse``,
   ``inferred_<targeted_attribute>_mae``,
   ``inferred_<targeted_attribute>_r2``

**Model inversion**

- Prefix: ``model_inversion_``
- Default summary metrics: ``model_inversion_mse``, ``model_inversion_mae``,
   ``model_inversion_num_targets``
- ``model_inversion_mse`` and ``model_inversion_mae`` compare reconstructed
   samples to per-class prototype means from the selected split.

**Database reconstruction**

- Prefix: ``database_reconstruction_``
- Default summary metrics: ``database_reconstruction_feature_mse``,
   ``database_reconstruction_feature_mae``,
   ``database_reconstruction_num_features``,
   ``database_reconstruction_num_known_rows``,
   ``database_reconstruction_missing_index``
- Optional label metrics (when reconstructed labels are returned):
   ``database_reconstruction_label_accuracy`` (classification) or
   ``database_reconstruction_label_mae`` (regression)
- Reconstructs one held-out row from a selected split (``train`` or ``test``)
   using the known rows and labels from that split.

**Poisoning**

- Compares model quality before and after poisoned retraining on the selected
   evaluation split
- Metric keys are emitted with both ``benign_`` and ``poisoned_`` prefixes,
   for example: ``benign_accuracy``, ``poisoned_accuracy``
- Additional poisoning metadata includes:

   - ``poison_attack_target_class``
   - ``poison_attack_source_class``
   - ``poison_trigger_index``
   - ``poison_trigger_predicted_class``
   - ``poison_trigger_success``
   - ``poison_mode``

**Extraction**

- Compares victim and extracted models on the selected evaluation split
- Metric keys are emitted with ``benign_`` and ``extracted_`` prefixes,
   for example: ``benign_accuracy``, ``extracted_accuracy``
- Additional extraction metadata includes ``extraction_mode``

**Common fields across attacks**

- ``attack_size``
- ``attack_generation_time``
- ``attack_prediction_time``
- ``attack_score_time``

Canonical Runtime Output Fields
-------------------------------

Runtime attack outputs use a canonical naming contract:

- ``attack``: raw attack artifact/result object returned by the backend attack.
- ``attack_predictions``: predictions or inferred outputs produced by the attack
   and consumed by scoring.
- ``attacked_labels``: ground-truth labels aligned with
   ``attack_predictions`` for score computation.

These field names are shared by attack execution and scorer routing to keep
attack score aggregation consistent across attack families and scoring modes.

Fairness-stratified attack scoring is available via
:class:`~deckard.score.attack.FairlearnAttackScorerConfig`, which computes
per-group metrics (for example, by sensitive feature) in addition to overall
attack metrics.

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
      alias="fgm",
   )

   # run the attack against the trained model
   scores = attack_cfg(data=data, model=model)
   print([k for k in scores if k.startswith("evasion_")])

Multi-attack with ExperimentConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute several attacks in one experiment, pass a list to the same
``attack`` field on :class:`deckard.experiment.ExperimentConfig`.

.. code-block:: python

   from deckard.experiment import ExperimentConfig

   cfg = ExperimentConfig(
      data=data,
      model=model,
      attack=[
         AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.05},
            attack_size=20,
            alias="fgm",
         ),
         AttackConfig(
            attack_type="art.attacks.evasion.HopSkipJump",
            attack_params={"max_iter": 5, "verbose": False},
            attack_size=20,
            alias="hsj",
         ),
      ],
   )
   scores = cfg()

   # First key keeps original name; colliding keys use alias suffix.
   print(scores.get("evasion_accuracy"))
   print(scores.get("evasion_accuracy_hsj"))

For multi-attack runs, each attack must define a unique non-empty alias.

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

Membership Inference Attack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute a membership inference attack against a trained model:

.. code-block:: python

   from deckard.attack import AttackConfig
   from deckard.data import DataConfig
   from deckard.model import ModelConfig
   from deckard.score.attack import AttackScorerConfig
   from deckard.score import DefaultClassifierConfig

   data = DataConfig(
      dataset_name="make_classification",
      data_params={
         "n_samples": 100,
         "n_features": 10,
         "n_informative": 5,
         "random_state": 42,
      },
      train_size=70,
      test_size=30,
      classifier=True,
   )
   data()

   model = ModelConfig(
      model_type="sklearn.ensemble.RandomForestClassifier",
      classifier=True,
      model_params={"n_estimators": 50, "random_state": 42},
   )
   model(data)

   # Membership inference with baseline attack
   membership_attack = AttackConfig(
      attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
      attack_size=10,
      scorer=AttackScorerConfig(
         membership_inference=DefaultClassifierConfig(),
      ),
   )

   scores = membership_attack(data=data, model=model)
   print(f"Membership inference success: {scores.get('membership_inference_accuracy', 'N/A')}")

Attribute Inference Attack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute an attribute inference attack against a trained model:

.. code-block:: python

   from deckard.attack import AttackConfig
   from deckard.score.attack import AttackScorerConfig
   from deckard.score import DefaultClassifierConfig

   # Attribute inference targeting the first feature
   attribute_attack = AttackConfig(
      attack_type="art.attacks.inference.attribute_inference.AttributeInferenceBaseline",
      attack_params={"attr_names": ["feature_0"]},
      attack_size=10,
      scorer=AttackScorerConfig(
         attribute_inference=DefaultClassifierConfig(),
      ),
   )

   scores = attribute_attack(data=data, model=model)
   print([k for k in scores if k.startswith("inferred_")])

Model Inversion Attack
~~~~~~~~~~~~~~~~~~~~~~

To run an ART model inversion attack (for example ``MIFace``):

.. code-block:: python

   from deckard.attack import AttackConfig

   inversion_attack = AttackConfig(
      attack_type="art.attacks.inference.model_inversion.mi_face.MIFace",
      attack_size=10,
      attack_params={
         "max_iter": 200,
         "threshold": 1.0,
         "initialization": "average",  # zeros|ones|random|average
         "split": "test",
      },
   )

   scores = inversion_attack(data=data, model=model)
   print(scores["model_inversion_mse"], scores["model_inversion_num_targets"])

Database Reconstruction Attack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run ART database reconstruction against a trained estimator:

.. code-block:: python

   from deckard.attack import AttackConfig

   reconstruction_attack = AttackConfig(
      attack_type="art.attacks.inference.reconstruction.DatabaseReconstruction",
      attack_size=1,
      attack_params={
         "split": "train",
         "missing_index": 0,
      },
   )

   scores = reconstruction_attack(data=data, model=model)
   print(scores["database_reconstruction_feature_mse"])

YAML config shortcuts are available at:

- ``examples/sklearn/config/attack/database-reconstruction.yaml``
- ``examples/pytorch/config/attack/database-reconstruction.yaml``

Poisoning Attack
~~~~~~~~~~~~~~~~

Poisoning attacks retrain/evaluate the model on poisoned data and report both
``benign_`` and ``poisoned_`` metric sets.

.. code-block:: python

   poisoning_attack = AttackConfig(
      attack_type="art.attacks.poisoning.GradientMatchingAttack",
      attack_params={
         "epsilon": 0.05,
      },
      attack_size=10,
   )

   scores = poisoning_attack(data=data, model=model)
   print(scores.get("benign_accuracy"), scores.get("poisoned_accuracy"))

Extraction Attack
~~~~~~~~~~~~~~~~~

Extraction attacks compare victim and extracted model quality on the selected
evaluation split.

.. code-block:: python

   extraction_attack = AttackConfig(
      attack_type="art.attacks.extraction.CopycatCNN",
      attack_params={
         "batch_size_fit": 16,
         "batch_size_query": 16,
         "nb_epochs": 2,
      },
      attack_size=20,
   )

   scores = extraction_attack(data=data, model=model)
   print(scores.get("benign_accuracy"), scores.get("extracted_accuracy"))

CLI Examples
~~~~~~~~~~~~

Run attacks directly from the terminal:

.. code-block:: bash

   # Evasion attack with FastGradientMethod
   python -m deckard optimize --config-name experiment \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_size=50 \
      attack.attack_params.eps=0.2

   # Boundary attack (slow but black-box)
   python -m deckard optimize --config-name experiment \
      attack.attack_type=art.attacks.evasion.BoundaryAttack \
      attack.attack_size=20 \
      attack.attack_params.max_iter=100

   # Membership inference attack
   python -m deckard optimize --config-name experiment \
      attack.attack_type=art.attacks.inference.membership_inference.MembershipInferenceBlackBox \
      attack.attack_size=30

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
