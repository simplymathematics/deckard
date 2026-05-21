# Fairlearn Integration

deckard provides support for fairness-aware machine learning through the optional Fairlearn extension modules. This integration enables fairness evaluation and mitigation workflows within the deckard framework.
See also: {doc}`pytorch` for torch-backed fairness workflows.

(fairlearn-overview)=

## Overview

The Fairlearn integration consists of three main extension modules:

- {mod}`deckard.plugins.fairlearn.data` — fairness-aware dataset configuration
- {mod}`deckard.plugins.fairlearn.model` — fairness-aware model training and evaluation
- {mod}`deckard.plugins.fairlearn.score` — fairness-aware scoring metrics

These modules support fairness analysis and mitigation by quantifying and reducing bias in model predictions.

External references:

- [Fairlearn documentation](https://fairlearn.org/main/)
- [`fairlearn.metrics.demographic_parity_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html)
- [`fairlearn.metrics.equalized_odds_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html)
- [`fairlearn.reductions.ExponentiatedGradient`](https://fairlearn.org/main/api_reference/generated/fairlearn.reductions.ExponentiatedGradient.html)
- [Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.org/) for paired attack/defense workflows

### Key Features

- **Fairness metrics**: evaluate model bias and group fairness
- **Mitigation strategies**: configurable pre-, in-, and post-processing mitigators
- **Integrated scoring**: compute both accuracy and fairness-specific metrics
- **ART compatibility**: work alongside standard ART attacks and defenses
- **Flexible backends**: support sklearn, PyTorch, and custom model types

### Score Types Available

Fairness scoring in deckard is provided by {mod}`deckard.plugins.fairlearn.score` with the default scorer profiles:

- {class}`~deckard.plugins.fairlearn.score.DefaultFairnessDataScoreConfig`
- {class}`~deckard.plugins.fairlearn.score.DefaultFairnessModelScoreConfig`

These include:

- ``demographic_parity_difference``
- ``equalized_odds_difference``
- ``statistical_parity_difference``
- ``disparate_impact``

The scorers operate on pandas DataFrame-backed data and can resolve context from ``y_pred`` or from ``data._X`` together with sensitive attribute configuration.

### Data Configuration

The {class}`~deckard.plugins.fairlearn.data.FairlearnDataConfig` extends {class}`deckard.data.DataConfig` with fairness parameters:

- Specify sensitive features for fairness analysis
- Define mitigation strategies (preprocessing, in-processing, postprocessing)
- Track group-wise statistics and fairness metrics
- Optional validation dataset for fairness measurement

### Data pipeline and preprocessing support

{class}`~deckard.plugins.fairlearn.data.FairlearnDataConfig` extends
{class}`~deckard.data.DataPipelineConfig`, so it keeps standard deckard
pipeline capabilities while adding fairness hooks:

- configurable preprocessing pipeline steps from core data config
- optional Fairlearn mitigation transform via ``fairness_defense`` callable config
- optional ANJANA anonymization insertion via ``anjana_defense``
- group-aware sampling and stratification
- standard split/k-fold/shuffle sampling through the base data stack

### Model Configuration

The {class}`~deckard.plugins.fairlearn.model.FairlearnModelConfig` supports:

- Standard model training with fairness constraints
- Optional group fairness measurement
- Fairness-utility tradeoff analysis
- Integration with fairness-aware loss functions
- Checkpoint management for fairness tracking

{class}`~deckard.plugins.fairlearn.model.FairlearnModelConfig` wraps
{class}`~deckard.model.ModelConfig` behavior and can still use deckard's
general model defenses via ``model.defense`` (ART preprocessors,
postprocessors, trainers, and detector pipelines) where compatible with the
selected backend/model.

### Scoring and Metrics

The {mod}`deckard.plugins.fairlearn.score` module provides:

- {class}`~deckard.plugins.fairlearn.score.DefaultFairnessDataScoreConfig` — data-level fairness metrics (group parity, bias)
- {class}`~deckard.plugins.fairlearn.score.DefaultFairnessModelScoreConfig` — model-level fairness and utility metrics (accuracy, group fairness)

### How Deckard builds MetricFrame

Group fairness aggregation is implemented through
[`fairlearn.metrics.MetricFrame`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html)
inside Deckard's Fairlearn scorer pipeline.

Deckard constructs MetricFrame with:

```python
return MetricFrame(
      metrics=metrics,
      y_true=y_true,
      y_pred=y_pred,
      sensitive_features=sensitive_features,
      control_features=control_features,
      sample_params=sample_params,
      n_boot=n_boot,
      ci_quantiles=ci_quantiles,
      random_state=random_state,
)
```

Parameter semantics in Deckard runtime:

- `metrics`: normalized metric callables from `group_scorers` on
   {class}`~deckard.plugins.fairlearn.score.FairlearnScoreDictConfig`.
- `y_true`: resolved labels from the active scoring mode (`train`, `test`,
   `val`, `attack`, or `attack-val`).
- `y_pred`: resolved predictions aligned to `y_true`.
- `sensitive_features`: protected/group attributes resolved from
   {class}`~deckard.data.DataConfig` (or explicitly passed at call-time).
- `control_features`: optional conditioning columns for conditional group
   evaluation (forwarded directly to MetricFrame).
- `sample_params`: optional per-metric sample kwargs mapping forwarded as-is.
- `n_boot`: optional bootstrap iteration count for confidence intervals.
- `ci_quantiles`: optional quantiles to report when bootstrap is enabled.
- `random_state`: optional RNG seed/state for reproducible bootstrap behavior.

Related reduction controls on
{class}`~deckard.plugins.fairlearn.score.FairlearnScoreDictConfig`:

- `group_reduction`: `difference`, `ratio`, or `none`
- `group_reduction_method`: `between_groups` or `to_overall`
- `include_group_overall`: include MetricFrame `overall` outputs
- `include_group_by_group`: include flattened `by_group` outputs

See also {doc}`score` for scorer profile composition and {doc}`attack` for
attack-time fairness scoring.

## Examples

```{seealso}

   Notebook-based Fairlearn workflows, including fairness-aware model training,
   data transforms, and fairness attack scoring, are documented in:

   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```
### Troubleshooting

- **No sensitive features**: Ensure sensitive_features list is non-empty and matches actual column names in the data.
- **Mitigation ineffective**: Try a different mitigation_strategy or adjust group_names.
- **Fairness metric not improving**: Tune fairness_loss_weight or try a different fairness_metric.
- **Memory issues with large datasets**: Consider batch-wise mitigation or sampling.

### See also

* {doc}`data` — general data configuration including {mod}`deckard.plugins.fairlearn.data`
* {doc}`model` — general model configuration including {mod}`deckard.plugins.fairlearn.model`
* {doc}`score` — scoring framework including {mod}`deckard.plugins.fairlearn.score`
* {doc}`pytorch` — optional PyTorch integration with Fairlearn
* {doc}`modules` — overview of all extensions
