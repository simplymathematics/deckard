Anjana Integration
==================

deckard provides support for anonymization-aware machine learning through the
optional Anjana extension modules. This integration enables privacy-preserving
modeling workflows within the deckard framework.

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

Score Types Available
~~~~~~~~~~~~~~~~~~~~~

Anjana scoring in deckard is provided by :mod:`deckard.score.anjana` with the
default scorer profiles:

- :class:`deckard.score.anjana.DefaultAnjanaDataScoreConfig`
- :class:`deckard.score.anjana.DefaultAnjanaModelScoreConfig`

These include:

- ``k_anonymity``
- ``l_diversity``
- ``t_closeness``

The scorers operate on pandas DataFrame-backed data and can resolve context
from ``y_pred`` or from ``data._X`` together with quasi-identifier and
sensitive-attribute configuration.

Data Configuration
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.data.anjana.AnjanaDataConfig` extends
:class:`deckard.data.DataConfig` with anonymization parameters:

- Specify sensitive attributes to anonymize
- Define anonymization strategies (suppression, bucketing, noise injection, etc.)
- Track original vs. anonymized dataset statistics
- Optional validation dataset for privacy measurement

Data pipeline and preprocessing support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AnjanaDataConfig`` extends ``DataPipelineConfig``, so it keeps standard
deckard pipeline capabilities while adding anonymization/fairness hooks:

- configurable preprocessing pipeline steps from core data config
- optional ANJANA defense transform via ``anjana_defense`` callable config
- optional fairlearn-style preprocessing insertion via ``fairness_defense``
- hierarchy generation support for quasi-identifiers
- standard split/k-fold/shuffle sampling through the base data stack

Model Configuration
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.model.anjana.AnjanaModelConfig` supports:

- Standard model training on anonymized data
- Optional utility measurement on original data
- Privacy-utility tradeoff analysis
- Integration with privacy-aware loss functions
- Checkpoint management for utility tracking

``AnjanaModelConfig`` wraps ``ModelConfig`` behavior and can still use deckard's
general model defenses via ``model.defense`` (ART preprocessors,
postprocessors, trainers, and detector pipelines) where compatible with the
selected backend/model.

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

.. seealso::

   Fully-executed programmatic examples — including basic Anjana workflows,
   attacks on anonymized models, and PyTorch integration — are available in
   the :doc:`notebooks/anjana.ipynb </notebooks/anjana>` notebook.

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

Defense options for users
~~~~~~~~~~~~~~~~~~~~~~~~~

Anjana workflows can layer three defense categories:

- data-level anonymization defenses (``anjana_defense``)
- fairness preprocessing defenses (``fairness_defense``)
- model-level ART defenses via ``model.defense`` / defense pipelines

This allows privacy transformation and adversarial defense composition in a
single experiment.

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
