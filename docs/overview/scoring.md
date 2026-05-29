# Scoring Overview

This page is the user-facing guide to how scoring is configured in deckard.
It summarizes scoring scope/stage behavior, default scorer families, and
example YAML patterns before you dive into low-level API/runtime details.

For full callable/runtime signature details, see {doc}`/api/score/index`.

## What Scoring Does

The scoring layer provides configurable scorer wrappers so data/model/attack
components can use a consistent scoring interface without hard-coding metric
implementations.

Core components:

- {class}`~deckard.score.ScorerConfig` wraps one metric callable.
- {class}`~deckard.score.ScorerDictConfig` normalizes a dictionary of scorer
  definitions.
- {class}`~deckard.attack.AttackConfig` delegates attack scoring to
  {class}`~deckard.score.attack.AttackScorerConfig`.

## Canonical Scoring Contract

- Runtime score scope is normalized through
  {func}`~deckard.score.normalize_scorer_mode` with canonical modes:
  `train`, `test`, `val`, `all`, `attack`, `attack-val`, and `pre-sample`.
  Data-profile scorers use `pre-sample` for dataset-level checks before sampling,
  while `train`/`test`/`val`/`all` apply to split-aware payloads.
- Runtime payload shape is represented by
  {class}`~deckard.score.ScorerRuntimeContract`.
- Stage filtering remains stage-token driven and independent from score scope.

## Output Shape Guarantees

- Scalar outputs are always persisted as primitive float-compatible values.
- Dict/Series/DataFrame outputs are flattened into scalar key/value pairs.
- Nested metric payloads are flattened with underscore-joined keys.

## Defaults

This section covers the baseline built-in scorer families used across core
deckard workflows (model and data), before attack-specific and optional plugin
specializations.

### Default Scorer Catalog

The default score profiles registered in Hydra config store are:

- [`scorers/classification`](#scorersclassification-and-scorersregression-yaml-example)
- [`scorers/regression`](#scorersclassification-and-scorersregression-yaml-example)
- [`scorers/fairness`](#scorersfairness-yaml-example)
- [`scorers/survival`](#scorerssurvival-yaml-example)
- [`scorers/privacy`](#scorersprivacy-yaml-example)
- [`scorers/evasion`](#scorersevasion-and-scorersevasion-regression-yaml-example)
- [`scorers/evasion-regression`](#scorersevasion-and-scorersevasion-regression-yaml-example)
- [`scorers/membership-inference`](#scorersmembership-inference-yaml-example)
- [`scorers/attribute-inference`](#scorersattribute-inference-and-scorersattribute-inference-regression-yaml-example)
- [`scorers/attribute-inference-regression`](#scorersattribute-inference-and-scorersattribute-inference-regression-yaml-example)
- [`scorers/fairlearn-attack`](#scorersfairlearn-attack-yaml-example)


### Core Model Defaults

- {class}`~deckard.score.DefaultModelScorerDictConfig`

(scorersclassification-and-scorersregression-yaml-example)=
#### `scorers/classification` and `scorers/regression` YAML example

```yaml
classifier: true
metrics:
  - accuracy: sklearn.metrics.accuracy_score
  - precision:
      function: sklearn.metrics.precision_score
      average: weighted
      zero_division: 0
  - recall:
      function: sklearn.metrics.recall_score
      average: weighted
      zero_division: 0
  - f1:
      function: sklearn.metrics.f1_score
      average: weighted
      zero_division: 0
  - roc_auc:
      function: sklearn.metrics.roc_auc_score
      average: weighted
      multi_class: ovr
      needs_proba: true
  - log_loss:
      function: sklearn.metrics.log_loss
      needs_logits: true

classifier: false
metrics:
  - mse:
      function: sklearn.metrics.mean_squared_error
      greater_is_better: false
  - mae:
      function: sklearn.metrics.mean_absolute_error
      greater_is_better: false
  - r2: sklearn.metrics.r2_score
```

- {class}`~deckard.score.DefaultClassifierScorerDictConfig`
- {class}`~deckard.score.DefaultRegressorScorerDictConfig`

### PyTorch Model Defaults

- {class}`~deckard.score.DefaultPytorchScorerDictConfig`

```yaml
classifier: true
metrics:
  - accuracy: sklearn.metrics.accuracy_score
  - precision: sklearn.metrics.precision_score
  - recall: sklearn.metrics.recall_score
  - f1: sklearn.metrics.f1_score

classifier: false
metrics:
  - mse: sklearn.metrics.mean_squared_error
  - mae: sklearn.metrics.mean_absolute_error
  - r2: sklearn.metrics.r2_score

notes:
  - roc_auc and log_loss are omitted for broad PyTorch wrapper compatibility
  - optimizer_loss from torch framework is included automatically
```

- {class}`~deckard.score.DefaultPytorchClassifierScorerDictConfig`
- {class}`~deckard.score.DefaultPytorchRegressorScorerDictConfig`

### Data Defaults

- {class}`~deckard.score.DefaultDataScorerDictConfig`

```yaml
classifier: true
metrics:
  - num_classes: deckard.score.data.data_num_classes_score
  - class_count_min: deckard.score.data.data_class_count_min_score
  - class_count_max: deckard.score.data.data_class_count_max_score
  - class_imbalance_ratio:
      function: deckard.score.data.data_class_imbalance_ratio_score
      greater_is_better: false
  - mutual_information_mean:
      deckard.score.data.data_mutual_information_mean_score
  - mutual_information_max:
      deckard.score.data.data_mutual_information_max_score

classifier: false
metrics:
  - mutual_information_mean:
      deckard.score.data.data_mutual_information_mean_score
  - mutual_information_max:
      deckard.score.data.data_mutual_information_max_score
  - empirical_cdf: deckard.score.data.data_empirical_cdf_function_score
```

- {class}`~deckard.score.DefaultDataClassificationScorerDictConfig`
- {class}`~deckard.score.DefaultDataRegressionScorerDictConfig`
- {class}`~deckard.score.DefaultPytorchDataScorerDictConfig`

```yaml
classifier: true
metrics:
  - split_count: deckard.score.data.pytorch_split_count_score
  - num_classes: deckard.score.data.data_num_classes_score
  - class_count_min: deckard.score.data.data_class_count_min_score
  - class_count_max: deckard.score.data.data_class_count_max_score
  - class_imbalance_ratio: deckard.score.data.data_class_imbalance_ratio_score

classifier: false
metrics:
  - split_count: deckard.score.data.pytorch_split_count_score
  - empirical_cdf: deckard.score.data.data_empirical_cdf_function_score
```

### DVC System Monitoring Defaults

- {class}`~deckard.score.DVCSystemScorerDictConfig`

When the DVC experiment plugin is enabled, Deckard runs a DVC system snapshot
scorer by default after each available component scoring stage:

- `data-score`
- `model-score`
- `attack-score`
- `detector-score`

The emitted metric names are concise component-stat keys, for example:

- `data_memory`
- `model_cpu`
- `attack_gpu`
- `defense_gpu_power`

As with other scorer outputs, these values are persisted with stage and mode
scope in the score payload, for example:

- `score_dict["data-score"]["test"]["data_cpu"]`

These metrics are derived from normalized DVCLive `system_monitor/*` signals
and include existing power-hook outputs when available (for example
`power/data/cpu_watts` contributes to `data_cpu_power`).

By default, DVC system scoring runs at component after-score stages through
scorer stage filters, and can be extended to other canonical scoring stages by
overriding scorer stage configuration.

### External Scorer References

External scorers referenced in the default profiles and YAML examples:

- scikit-learn:
  [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html),
  [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html),
  [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html),
  [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html),
  [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
  [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html),
  [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html),
  [mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html),
  [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

## Attacks

Attack scoring profiles specialize metric sets for evasion, membership
inference, and attribute inference workflows.

### Attack Defaults

- {class}`~deckard.score.DefaultEvasionAttackScorerDictConfig`

(scorersevasion-and-scorersevasion-regression-yaml-example)=
#### `scorers/evasion` and `scorers/evasion-regression` YAML example

```yaml
classifier: true
metrics:
  - accuracy: sklearn.metrics.accuracy_score
  - precision: sklearn.metrics.precision_score
  - recall: sklearn.metrics.recall_score
  - f1-score: sklearn.metrics.f1_score
  - success: deckard.score.attack.evasion_success_score

classifier: false
metrics:
  - mse: sklearn.metrics.mean_squared_error
  - mae: sklearn.metrics.mean_absolute_error
  - r2: sklearn.metrics.r2_score
```

- {class}`~deckard.score.DefaultEvasionRegressionAttackScorerDictConfig`
- {class}`~deckard.score.DefaultMembershipInferenceAttackScorerDictConfig`

(scorersmembership-inference-yaml-example)=
#### `scorers/membership-inference` YAML example

```yaml
classifier: true
metrics:
  - accuracy: sklearn.metrics.accuracy_score
  - precision: sklearn.metrics.precision_score
  - recall: sklearn.metrics.recall_score
  - f1: sklearn.metrics.f1_score
```

- {class}`~deckard.score.DefaultAttributeInferenceAttackScorerDictConfig`

```yaml
classifier: true
metrics:
  - accuracy: sklearn.metrics.accuracy_score
  - precision: sklearn.metrics.precision_score
  - recall: sklearn.metrics.recall_score
  - f1: sklearn.metrics.f1_score

classifier: false
metrics:
  - mse: sklearn.metrics.mean_squared_error
  - mae: sklearn.metrics.mean_absolute_error
  - r2: sklearn.metrics.r2_score
```

- {class}`~deckard.score.DefaultAttributeInferenceRegressionAttackScorerDictConfig`

(scorersattribute-inference-and-scorersattribute-inference-regression-yaml-example)=
#### `scorers/attribute-inference` and `scorers/attribute-inference-regression` YAML example

(scorersfairlearn-attack-yaml-example)=
#### `scorers/fairlearn-attack` YAML example

```yaml
_target_: deckard.score.FairlearnAttackScorerConfig
evasion:
  # grouped metrics default to accuracy/f1 plus success in base scorers
  group_reduction: difference
membership_inference:
  # grouped metrics default to accuracy/f1
  group_reduction: difference
attribute_inference:
  # grouped metrics default to accuracy/f1
  group_reduction: difference
attribute_inference_regression:
  # grouped metrics default to mse/mae
  group_reduction: difference
```

### Attack Score Naming

Attack metrics are prefixed by attack family at runtime:

- `evasion_*`
- `membership_inference_*`
- `inferred_<attribute>_*`

## Lifelines

Lifelines adds survival-analysis scoring on top of the default scorer families.

- Lifelines config class: {class}`~deckard.score.DefaultLifelinesConfig`
- Lifelines scorer reference:
  [concordance_index](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index)

(scorerssurvival-yaml-example)=
### `scorers/survival` YAML example

```yaml
metrics:
  - concordance
  - aic
  - bic
```

## Anjana

Anjana adds privacy/utility-oriented anonymization scoring profiles.

- Anjana config classes: {class}`~deckard.score.DefaultAnjanaScorerDictConfig`,
  {class}`~deckard.score.DefaultAnjanaDataScorerDictConfig`,
  {class}`~deckard.score.DefaultAnjanaModelScorerDictConfig`

(scorersprivacy-yaml-example)=
### `scorers/privacy` YAML example

```yaml
metrics:
  - k_anonymity
  - l_diversity
  - t_closeness
```

## Fairlearn

Fairlearn extends base model metrics with group-aware disparity
metrics.

In practice, these metrics are commonly computed from grouped performance
summaries (for example, across sensitive-feature slices) and then merged into
deckard score output as scalar keys.

- Fairlearn MetricFrame reference:
  [MetricFrame](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html)
- Common fairness metrics used by deckard fairlearn defaults:
  [demographic_parity_difference](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html),
  [equalized_odds_difference](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html),
  [group_mean_prediction](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.group_mean_prediction.html),
  [mean_absolute_error_group_min](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.mean_absolute_error_group_min.html),
  [mean_squared_error_group_max](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.mean_squared_error_group_max.html)

Typical fairness-profile override:

(scorersfairness-yaml-example)=
### `scorers/fairness` YAML example

```yaml
score:
  _target_: deckard.score.base.ScorerDictConfig
  classifier: true
  scorers:
    demographic_parity_difference:
      score_function: fairlearn.metrics.demographic_parity_difference
      score_params:
        sensitive_features: ${data.sensitive_features}
    equalized_odds_difference:
      score_function: fairlearn.metrics.equalized_odds_difference
      score_params:
        sensitive_features: ${data.sensitive_features}
```

- Fairlearn config classes: {class}`~deckard.score.DefaultFairlearnScorerDictConfig`,
  {class}`~deckard.score.DefaultFairlearnClassificationScorerDictConfig`,
  {class}`~deckard.score.DefaultFairlearnRegressionScorerDictConfig`,
  {class}`~deckard.score.DefaultFairlearnDataScorerDictConfig`

```yaml
inherits:
  - deckard.score.DefaultClassifierScorerDictConfig
  - deckard.score.DefaultRegressorScorerDictConfig
adds:
  classifier:
    - demographic_parity_difference
    - equalized_odds_difference
    - group_mean_prediction_difference
  regressor:
    - group_mae_difference
    - group_mse_difference
```

## Hydra Syntax Cheat Sheet

The following example shows alias-based Hydra composition for configuring data,
model, attack, and Anjana scoring, while also enabling Fairlearn group-aware
scores.

```yaml
# config.yaml
defaults:
  - data: default
  - model: default
  - attack: evasion
  - score: classification
  - _self_

# Swap score aliases per behavior:
# - model defaults: score=classification or score=regression
# - data scoring: score=data-classification or score=data-regression
# - attack scoring: score=evasion-classification or score=evasion-regression
# - fairlearn group scoring: score=fairness-classification or score=fairness-regression
# - anjana scoring: score=anjana

# Required for group fairness metrics
data:
  sensitive_features: [sex, race]
```

### Bash + Hydra CLI Examples

These examples use command-line overrides to activate each scoring behavior.

```bash
# Data scoring behavior
deckard optimize --config-path examples/sklearn/config --config-name default \
  score=data-classification

# Model scoring behavior
deckard optimize --config-path examples/sklearn/config --config-name default \
  score=classification

# Attack scoring behavior
deckard optimize --config-path examples/sklearn/config --config-name default \
  score=evasion-classification

# Anjana scoring behavior
deckard optimize --config-path examples/sklearn/config --config-name default \
  score=anjana

# Fairlearn group scoring behavior (requires sensitive feature columns)
deckard optimize --config-path examples/sklearn/config --config-name default \
  score=fairness-classification \
  "data.sensitive_features=[sex,race]"

# Combined coverage via run-time composition aliases (one run per scoring behavior)
deckard optimize --config-path examples/sklearn/config --config-name default \
  score=data-classification,classification,evasion-classification,anjana,fairness-classification \
  "data.sensitive_features=[sex,race]"
```

To ensure group scores are actually computed:

- Provide `data.sensitive_features` in the runtime payload.
- Select a Fairlearn score alias (for example `score=fairness-classification`).
- Use fairness metrics that consume `sensitive_features`, for example
  `demographic_parity_difference` and `equalized_odds_difference`.

## Next Steps

- Runtime API details and [__call__](../api/modules) signatures: {doc}`/api/score/index`
- Orchestration context: {doc}`experiment`
