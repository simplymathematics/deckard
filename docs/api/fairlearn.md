# Fairlearn Integration

deckard provides support for fairness-aware machine learning through the optional Fairlearn extension modules. This integration enables fairness evaluation and mitigation workflows within the deckard framework.
See also: :doc:`pytorch` for torch-backed fairness workflows.

.. _fairlearn-overview:

## Overview

The Fairlearn integration consists of three main extension modules:

- :mod:`deckard.data.fairness` — fairness-aware dataset configuration
- :mod:`deckard.model.fairness` — fairness-aware model training and evaluation
- :mod:`deckard.score.fairness` — fairness-aware scoring metrics

These modules support fairness analysis and mitigation by quantifying and reducing bias in model predictions.

### Key Features

- **Fairness metrics**: evaluate model bias and group fairness
- **Mitigation strategies**: configurable pre-, in-, and post-processing mitigators
- **Integrated scoring**: compute both accuracy and fairness-specific metrics
- **ART compatibility**: work alongside standard ART attacks and defenses
- **Flexible backends**: support sklearn, PyTorch, and custom model types

### Score Types Available

Fairness scoring in deckard is provided by :mod:`deckard.score.fairness` with the default scorer profiles:

- :class:`~deckard.score.fairness.DefaultFairnessDataScoreConfig`
- :class:`~deckard.score.fairness.DefaultFairnessModelScoreConfig`

These include:

- ``demographic_parity_difference``
- ``equalized_odds_difference``
- ``statistical_parity_difference``
- ``disparate_impact``

The scorers operate on pandas DataFrame-backed data and can resolve context from ``y_pred`` or from ``data._X`` together with sensitive attribute configuration.

### Data Configuration

The :class:`~deckard.data.fairness.FairlearnDataConfig` extends :class:`deckard.data.DataConfig` with fairness parameters:

- Specify sensitive features for fairness analysis
- Define mitigation strategies (preprocessing, in-processing, postprocessing)
- Track group-wise statistics and fairness metrics
- Optional validation dataset for fairness measurement

### Data pipeline and preprocessing support

:class:`~deckard.data.fairness.FairlearnDataConfig` extends
:class:`~deckard.data.DataPipelineConfig`, so it keeps standard deckard
pipeline capabilities while adding fairness hooks:

- configurable preprocessing pipeline steps from core data config
- optional Fairlearn mitigation transform via ``fairness_defense`` callable config
- optional ANJANA anonymization insertion via ``anjana_defense``
- group-aware sampling and stratification
- standard split/k-fold/shuffle sampling through the base data stack

### Model Configuration

The :class:`~deckard.model.fairness.FairlearnModelConfig` supports:

- Standard model training with fairness constraints
- Optional group fairness measurement
- Fairness-utility tradeoff analysis
- Integration with fairness-aware loss functions
- Checkpoint management for fairness tracking

:class:`~deckard.model.fairness.FairlearnModelConfig` wraps
:class:`~deckard.model.ModelConfig` behavior and can still use deckard's
general model defenses via ``model.defense`` (ART preprocessors,
postprocessors, trainers, and detector pipelines) where compatible with the
selected backend/model.

### Scoring and Metrics

The :mod:`deckard.score.fairness` module provides:

- :class:`~deckard.score.fairness.DefaultFairnessDataScoreConfig` — data-level fairness metrics (group parity, bias)
- :class:`~deckard.score.fairness.DefaultFairnessModelScoreConfig` — model-level fairness and utility metrics (accuracy, group fairness)

## Examples

.. seealso::

   Notebook-based Fairlearn workflows, including fairness-aware model training,
   data transforms, and fairness attack scoring, are documented in:

   - :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`

### Troubleshooting

- **No sensitive features**: Ensure sensitive_features list is non-empty and matches actual column names in the data.
- **Mitigation ineffective**: Try a different mitigation_strategy or adjust group_names.
- **Fairness metric not improving**: Tune fairness_loss_weight or try a different fairness_metric.
- **Memory issues with large datasets**: Consider batch-wise mitigation or sampling.

### See also

* :doc:`data` — general data configuration including :mod:`deckard.data.fairness`
* :doc:`model` — general model configuration including :mod:`deckard.model.fairness`
* :doc:`score` — scoring framework including :mod:`deckard.score.fairness`
* :doc:`pytorch` — optional PyTorch integration with Fairlearn
* :doc:`modules` — overview of all extensions
