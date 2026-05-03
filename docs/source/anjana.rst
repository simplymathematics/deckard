Anjana Integration
==================

Deckard provides support for anonymization-aware machine learning through the
optional Anjana extension modules. This integration enables privacy-preserving
modeling workflows within the Deckard framework.

.. _anjana-overview:

Overview
--------

The Anjana integration consists of three main extension modules:

- :mod:`deckard.data.anjana` — anonymization-aware dataset configuration
- :mod:`deckard.model.anjana` — anonymization-aware model training and evaluation
- :mod:`deckard.score.anjana` — anonymization-aware scoring metrics

These modules support privacy-preserving adversarial robustness studies by
quantifying model behavior under anonymization constraints.

Key Features
~~~~~~~~~~~~

- **Privacy metrics**: evaluate model robustness to data anonymization
- **Anonymization profiles**: configurable strategies (differential privacy,
  suppression, generalization, etc.)
- **Integrated scoring**: compute both utility and privacy-specific metrics
- **ART compatibility**: work alongside standard ART attacks and defenses
- **Flexible backends**: support sklearn, PyTorch, and custom model types

Data Configuration
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.data.anjana.AnjanaDataConfig` extends
:class:`deckard.data.DataConfig` with anonymization parameters:

- Specify sensitive attributes to anonymize
- Define anonymization strategies (suppression, bucketing, noise injection, etc.)
- Track original vs. anonymized dataset statistics
- Optional validation dataset for privacy measurement

Model Configuration
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.model.anjana.AnjanaModelConfig` supports:

- Standard model training on anonymized data
- Optional utility measurement on original data
- Privacy-utility tradeoff analysis
- Integration with privacy-aware loss functions
- Checkpoint management for utility tracking

Scoring and Metrics
~~~~~~~~~~~~~~~~~~~

The :mod:`deckard.score.anjana` module provides:

- :class:`~deckard.score.anjana.DefaultAnjanaDataScoreConfig` — data-level
  privacy metrics (information loss, suppression rate)
- :class:`~deckard.score.anjana.DefaultAnjanaModelScoreConfig` — model-level
  privacy and utility metrics (accuracy drop, privacy guarantee strength)

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Basic Anjana experiment:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=anjana \
      data.dataset_name=make_classification \
      model=anjana \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      score.data=anjana \
      score.model=anjana

**Anjana with PyTorch backend:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=anjana \
      data.base_data_config=pytorch \
      data.dataset_name=CIFAR10 \
      model=anjana \
      model.base_model_config=pytorch \
      score.data=anjana \
      score.model=anjana

**Anjana with attack evaluation:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=anjana \
      model=anjana \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_params.eps=0.1 \
      score.data=anjana \
      score.model=anjana

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

**Basic Anjana workflow:**

.. code-block:: python

   from deckard.data.anjana import AnjanaDataConfig
   from deckard.model.anjana import AnjanaModelConfig
   from deckard.experiment import ExperimentConfig
   from deckard.score.anjana import (
       DefaultAnjanaDataScoreConfig,
       DefaultAnjanaModelScoreConfig,
   )

   # Configure anonymization-aware data
   data = AnjanaDataConfig(
       dataset_name="make_classification",
       data_params={"n_samples": 500, "n_features": 20, "n_classes": 2},
       train_size=70,
       test_size=30,
       sensitive_attributes=["age", "income"],  # columns to anonymize
       anonymization_strategy="suppression",  # or "bucketing", "noise", etc.
       classifier=True,
       scorer=DefaultAnjanaDataScoreConfig(),
   )

   # Configure model with anonymization awareness
   model = AnjanaModelConfig(
       model_type="sklearn.ensemble.RandomForestClassifier",
       classifier=True,
       model_params={"n_estimators": 50, "max_depth": 10},
       scorer=DefaultAnjanaModelScoreConfig(),
   )

   # Run experiment with anonymization evaluation
   cfg = ExperimentConfig(data=data, model=model)
   scores = cfg()

   print("Original accuracy:", scores.get("accuracy"))
   print("Anonymized accuracy:", scores.get("anonymized_accuracy", "N/A"))
   print("Privacy score:", scores.get("privacy_score", "N/A"))

**Anjana with attacks:**

.. code-block:: python

   from deckard.attack import AttackConfig

   # Define attack on anonymized models
   attack = AttackConfig(
       attack_type="art.attacks.evasion.FastGradientMethod",
       attack_params={"eps": 0.15},
       attack_size=100,
   )

   # Evaluate attack robustness of anonymized models
   cfg = ExperimentConfig(
       data=data,
       model=model,
       attack=attack,
   )
   scores = cfg()

   print("Evasion success rate:", scores.get("evasion_success_rate"))
   print("Privacy-robust robustness:", scores.get("privacy_attack_success", "N/A"))

**Anjana with PyTorch:**

.. code-block:: python

   from deckard.data.anjana import AnjanaDataConfig
   from deckard.data.pytorch import PytorchDataConfig
   from deckard.model.anjana import AnjanaModelConfig
   from deckard.model.pytorch import PytorchModelConfig
   from deckard.experiment.torch_experiment import TorchExperimentConfig

   # Base PyTorch data config
   pytorch_data = PytorchDataConfig(
       dataset_name="CIFAR10",
       train_size=45000,
       test_size=5000,
       device="auto",
       classifier=True,
   )

   # Wrap with Anjana anonymization
   data = AnjanaDataConfig(
       base_data_config=pytorch_data,
       sensitive_attributes=["label_coarse"],
       anonymization_strategy="generalization",
   )

   # PyTorch model with Anjana tracking
   pytorch_model = PytorchModelConfig(
       model_type="torchvision.models.resnet18",
       classifier=True,
       device="auto",
       epochs=10,
   )

   model = AnjanaModelConfig(
       base_model_config=pytorch_model,
   )

   # Run PyTorch + Anjana experiment
   cfg = TorchExperimentConfig(data=data, model=model)
   scores = cfg()

Configuration
~~~~~~~~~~~~~

Key configuration options for :class:`~deckard.data.anjana.AnjanaDataConfig`:

- **base_data_config** (DataConfig, optional): base data config to extend with
  anonymization; if omitted, uses :class:`deckard.data.DataConfig`
- **sensitive_attributes** (list): column names or indices to anonymize
- **anonymization_strategy** (str): one of "suppression", "bucketing", "noise",
  "generalization", "pseudonymization"
- **suppression_rate** (float): fraction of sensitive values to suppress (0–1)
- **bucket_size** (int): group size for bucketing strategy
- **noise_scale** (float): standard deviation for Gaussian noise injection
- **generalization_hierarchy** (dict, optional): domain-specific generalization
  rules

For :class:`~deckard.model.anjana.AnjanaModelConfig`:

- **base_model_config** (ModelConfig, optional): base model to extend
- **track_utility** (bool): compute accuracy on original vs. anonymized data
- **privacy_loss_weight** (float): optional weighting in composite loss function

Interpretation
~~~~~~~~~~~~~~

Anjana scores commonly include:

- **information_loss**: fraction of information removed by anonymization (0–1)
- **suppression_rate**: fraction of suppressed values (0–1)
- **utility_drop**: decrease in model accuracy due to anonymization
- **privacy_guarantee**: estimated privacy protection level (depends on strategy)
- **accuracy_original**: accuracy on original dataset
- **accuracy_anonymized**: accuracy on anonymized dataset

A well-tuned anonymization strategy balances privacy (high information loss) with
utility (low accuracy drop).

Troubleshooting
~~~~~~~~~~~~~~~

- **No sensitive attributes**: Ensure sensitive_attributes list is non-empty and
  matches actual column names in the data.
- **Anonymization too aggressive**: Reduce suppression_rate or bucket_size to
  preserve more utility.
- **Information loss negligible**: Increase anonymization parameters or switch to
  a stronger strategy.
- **Memory issues with large datasets**: Consider batch-wise anonymization or
  sampling.

See also
~~~~~~~~

* :doc:`data` — general data configuration including :mod:`deckard.data.anjana`
* :doc:`model` — general model configuration including :mod:`deckard.model.anjana`
* :doc:`score` — scoring framework including :mod:`deckard.score.anjana`
* :doc:`pytorch` — optional PyTorch integration with Anjana
* :doc:`package` — overview of all extensions
