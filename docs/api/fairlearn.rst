Fairlearn Integration
=======================

deckard provides support for fairness-aware machine learning through the optional Fairlearn extension modules. This integration enables fairness evaluation and mitigation workflows within the deckard framework.

.. _fairlearn-overview:

Overview
--------

The Fairlearn integration consists of three main extension modules:

- :mod:`deckard.data.fairness` — fairness-aware dataset configuration
- :mod:`deckard.model.fairness` — fairness-aware model training and evaluation
- :mod:`deckard.score.fairness` — fairness-aware scoring metrics

These modules support fairness analysis and mitigation by quantifying and reducing bias in model predictions.

Key Features
~~~~~~~~~~~~

- **Fairness metrics**: evaluate model bias and group fairness
- **Mitigation strategies**: configurable pre-, in-, and post-processing mitigators
- **Integrated scoring**: compute both accuracy and fairness-specific metrics
- **ART compatibility**: work alongside standard ART attacks and defenses
- **Flexible backends**: support sklearn, PyTorch, and custom model types

Score Types Available
~~~~~~~~~~~~~~~~~~~~~

Fairness scoring in deckard is provided by :mod:`deckard.score.fairness` with the default scorer profiles:

- :class:`deckard.score.fairness.DefaultFairnessDataScoreConfig`
- :class:`deckard.score.fairness.DefaultFairnessModelScoreConfig`

These include:

- ``demographic_parity_difference``
- ``equalized_odds_difference``
- ``statistical_parity_difference``
- ``disparate_impact``

The scorers operate on pandas DataFrame-backed data and can resolve context from ``y_pred`` or from ``data._X`` together with sensitive attribute configuration.

Data Configuration
~~~~~~~~~~~~~~~~~~

The :class:`~deckard.data.fairness.FairlearnDataConfig` extends :class:`deckard.data.DataConfig` with fairness parameters:

- Specify sensitive features for fairness analysis
- Define mitigation strategies (preprocessing, in-processing, postprocessing)
- Track group-wise statistics and fairness metrics
- Optional validation dataset for fairness measurement

Data pipeline and preprocessing support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FairlearnDataConfig`` extends ``DataPipelineConfig``, so it keeps standard deckard pipeline capabilities while adding fairness hooks:

- configurable preprocessing pipeline steps from core data config
- optional Fairlearn mitigation transform via ``fairness_defense`` callable config
- optional ANJANA anonymization insertion via ``anjana_defense``
- group-aware sampling and stratification
- standard split/k-fold/shuffle sampling through the base data stack

Model Configuration
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.model.fairness.FairlearnModelConfig` supports:

- Standard model training with fairness constraints
- Optional group fairness measurement
- Fairness-utility tradeoff analysis
- Integration with fairness-aware loss functions
- Checkpoint management for fairness tracking

``FairlearnModelConfig`` wraps ``ModelConfig`` behavior and can still use deckard's general model defenses via ``model.defense`` (ART preprocessors, postprocessors, trainers, and detector pipelines) where compatible with the selected backend/model.

Scoring and Metrics
~~~~~~~~~~~~~~~~~~~

The :mod:`deckard.score.fairness` module provides:

- :class:`~deckard.score.fairness.DefaultFairnessDataScoreConfig` — data-level fairness metrics (group parity, bias)
- :class:`~deckard.score.fairness.DefaultFairnessModelScoreConfig` — model-level fairness and utility metrics (accuracy, group fairness)

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Basic Fairlearn experiment:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=fairlearn \
      data.dataset_name=adult \
      model=fairlearn \
      model.model_type=sklearn.linear_model.LogisticRegression \
      score.data=fairness \
      score.model=fairness

**Fairlearn with PyTorch backend:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=fairlearn \
      data.base_data_config=pytorch \
      data.dataset_name=CIFAR10 \
      model=fairlearn \
      model.base_model_config=pytorch \
      score.data=fairness \
      score.model=fairness

**Fairlearn with attack evaluation:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=fairlearn \
      model=fairlearn \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_params.eps=0.1 \
      score.data=fairness \
      score.model=fairness

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

.. seealso::

   Fully-executed programmatic examples — including basic Fairlearn workflows, attacks on fairness-aware models, and PyTorch integration — are available in the :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>` notebook.

Configuration
~~~~~~~~~~~~~

Key configuration options for :class:`~deckard.data.fairness.FairlearnDataConfig`:

- **base_data_config** (DataConfig, optional): base data config to extend with fairness; if omitted, uses :class:`deckard.data.DataConfig`
- **sensitive_features** (list): column names or indices for fairness analysis
- **mitigation_strategy** (str): one of "preprocessing", "inprocessing", "postprocessing"
- **group_names** (list): names of groups for group fairness analysis
- **fairness_metric** (str): metric to optimize (e.g., "demographic_parity_difference")

For :class:`~deckard.model.fairness.FairlearnModelConfig`:

- **base_model_config** (ModelConfig, optional): base model to extend
- **track_fairness** (bool): compute group fairness metrics
- **fairness_loss_weight** (float): optional weighting in composite loss function

Defense options for users
~~~~~~~~~~~~~~~~~~~~~~~~~

Fairlearn workflows can layer three defense categories:

- data-level fairness mitigations (``fairness_defense``)
- anonymization preprocessing defenses (``anjana_defense``)
- model-level ART defenses via ``model.defense`` / defense pipelines

This allows fairness transformation and adversarial defense composition in a single experiment.

Interpretation
~~~~~~~~~~~~~~

Fairlearn scores commonly include:

- **demographic_parity_difference**: difference in positive outcome rates between groups
- **equalized_odds_difference**: difference in true/false positive rates between groups
- **statistical_parity_difference**: difference in selection rates between groups
- **disparate_impact**: ratio of positive outcome rates between groups
- **accuracy_group_0/1**: accuracy for each group
- **overall_accuracy**: overall model accuracy

A well-tuned fairness strategy reduces group disparities while maintaining high utility.

Troubleshooting
~~~~~~~~~~~~~~~

- **No sensitive features**: Ensure sensitive_features list is non-empty and matches actual column names in the data.
- **Mitigation ineffective**: Try a different mitigation_strategy or adjust group_names.
- **Fairness metric not improving**: Tune fairness_loss_weight or try a different fairness_metric.
- **Memory issues with large datasets**: Consider batch-wise mitigation or sampling.

See also
~~~~~~~~

* :doc:`data` — general data configuration including :mod:`deckard.data.fairness`
* :doc:`model` — general model configuration including :mod:`deckard.model.fairness`
* :doc:`score` — scoring framework including :mod:`deckard.score.fairness`
* :doc:`pytorch` — optional PyTorch integration with Fairlearn
* :doc:`modules` — overview of all extensions
