# Score

## Introduction

This page is the canonical home for scoring behavior and API details.
It documents scorer configuration, runtime scope/stage semantics, serialization
shape guarantees, and default scorer catalogs.

The {mod}`deckard.score` module defines scorer configuration objects used by
model, attack, and experiment pipelines.

## Overview

The score layer provides configurable scorer wrappers so data/model/attack
components can use a consistent scoring interface without hard-coding metric
implementations.

## Canonical Runtime Contract

Scoring now uses a canonical runtime contract shared across data/model/attack/
detector execution paths.

- Runtime score scope is normalized through
  {func}`~deckard.score.normalize_scorer_mode` with canonical modes:
  `train`, `test`, `val`, `all`, `attack`, `attack-val`, `pre-sample`.
- Runtime payload shape is represented by
  {class}`~deckard.score.ScorerRuntimeContract`.
- Stage filtering remains stage-token driven and independent from score scope.

## Output Shape Guarantees

Score outputs are normalized to remain flat, serializable, and merge-safe.

- Scalar metric outputs are always persisted as primitive float-compatible
  values.
- Dict/Series/DataFrame metric outputs are flattened into scalar key/value
  pairs before merge.
- Nested metric payloads are flattened with underscore-joined keys, then
  prefixed by scorer key when merged into stage outputs.

deckard now treats scoring as a runtime-configured layer rather than a fixed
set of metrics embedded inside each pipeline component.

- {class}`~deckard.score.ScorerConfig` wraps a single metric callable.
- {class}`~deckard.score.ScorerDictConfig` normalizes a mapping of metric names
  into callable scorer definitions.
- {class}`~deckard.model.ModelConfig` and {class}`~deckard.data.DataConfig` accept
  scorer configs directly through their `scorer` fields.
- {class}`~deckard.attack.AttackConfig` delegates all attack scoring to
  {class}`~deckard.score.attack.AttackScorerConfig`.

Attack scoring is now profile-based and attack-kind-aware:

- Evasion attacks use an evasion scorer profile and prefix outputs with
  `evasion_`.
- Membership inference attacks use a membership scorer profile and prefix
  outputs with `membership_inference_`.
- Attribute inference attacks use attribute scorer profiles and prefix outputs
  with `inferred_<attribute>_`.
- Generic {class}`~deckard.score.ScorerDictConfig` instances can be supplied to
  attack scorer profiles; deckard will route the correct `y_true` and
  `y_pred` values for the active attack kind and then prefix the resulting
  metric names.

The default score profiles registered in Hydra's config store are:

- `scorers/classification`
- `scorers/regression`
- `scorers/fairness`
- `scorers/survival`
- `attack_scorers/evasion`
- `attack_scorers/evasion-regression`
- `attack_scorers/membership-inference`
- `attack_scorers/attribute-inference`

These registrations are added through :func:`~deckard.score.safe_store`, which
wraps Hydra's `ConfigStore.instance().store(...)` and tolerates duplicate
import-time registration attempts in tests and repeated imports.

## Common scorer components and references

Specific scorer profiles in this module are designed to compose with
{doc}`model`, {doc}`data`, {doc}`attack`, and plugin APIs such as
{doc}`fairlearn` and {doc}`lifelines`.

Frequently referenced metric callables include:

- [`sklearn.metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [`sklearn.metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- [`fairlearn.metrics.demographic_parity_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html)
- [`fairlearn.metrics.equalized_odds_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html)
- [`lifelines.utils.concordance_index`](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index)

## Default Scorer Catalog

This section documents every built-in `Default*` scorer configuration and the
metrics each one registers by default.

### Core Model Defaults

- {class}`~deckard.score.DefaultModelScorerDictConfig`

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
  - Fixed classifier specialization of `DefaultModelScorerDictConfig`.
- {class}`~deckard.score.DefaultRegressorScorerDictConfig`
  - Fixed regressor specialization of `DefaultModelScorerDictConfig`.

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
   - **optimizer_loss** user-specified loss value from `torch` framework is included in the scoring dict automatically.

  ```

- {class}`~deckard.score.DefaultPytorchClassifierScorerDictConfig`
  - Fixed classifier specialization of `DefaultPytorchScorerDictConfig`.
- {class}`~deckard.score.DefaultPytorchRegressorScorerDictConfig`
  - Fixed regressor specialization of `DefaultPytorchScorerDictConfig`.

### Attack Defaults

- {class}`~deckard.score.DefaultEvasionAttackScorerDictConfig`

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
  - Fixed regression specialization of evasion scorers.
- {class}`~deckard.score.DefaultMembershipInferenceAttackScorerDictConfig`

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
  - Fixed regression specialization of attribute-inference scorers.

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
  - Fixed classification specialization of `DefaultDataScorerDictConfig`.
- {class}`~deckard.score.DefaultDataRegressionScorerDictConfig`
  - Fixed regression specialization of `DefaultDataScorerDictConfig`.
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

### Optional Plugin Defaults

These defaults are available when optional dependencies are installed.

- Fairlearn (`fairlearn` extra):
  - {class}`~deckard.score.DefaultFairlearnScorerDictConfig`

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

  - {class}`~deckard.score.DefaultFairlearnClassificationScorerDictConfig`
    - Fixed classifier specialization.
  - {class}`~deckard.score.DefaultFairlearnRegressionScorerDictConfig`
    - Fixed regressor specialization.
  - {class}`~deckard.score.DefaultFairlearnDataScorerDictConfig`

    ```yaml
    metrics:
      - class_count
      - mutual_info

    ```

- Lifelines (`lifelines` extra):
  - {class}`~deckard.score.DefaultLifelinesConfig`

    ```yaml
    metrics:
      - concordance
      - aic
      - bic

    ```

- Anjana / PyCanon (`anjana` extra):
  - {class}`~deckard.score.DefaultAnjanaScorerDictConfig`

    ```yaml
    metrics:
      - k_anonymity
      - l_diversity
      - t_closeness

    ```

  - {class}`~deckard.score.DefaultAnjanaDataScorerDictConfig`
    - Base data metrics + Anjana privacy metrics
  - {class}`~deckard.score.DefaultAnjanaModelScorerDictConfig`
    - Model-scope specialization of Anjana privacy defaults

### Runtime Naming Notes

- Attack metrics are prefixed by attack family at runtime:
  - `evasion_*`
  - `membership_inference_*`
  - `inferred_<attribute>_*` for attribute inference
- Some metric names in config differ slightly from emitted keys due to
  scorer-name normalization and attack-prefix routing.

## Examples

```{seealso}

   Notebook-based scoring workflows (model metrics, data diagnostics,
   attack scoring, and fairness scoring) are documented in:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## Custom Scoring and Runtime Arguments

Use a {class}`~deckard.score.ScorerDictConfig` when you want full control over
which metrics run, which stage they run in, and which runtime payload is passed
to each metric.

### Minimal YAML Example

```yaml
score:
  _target_: deckard.score.base.ScorerDictConfig
  scorers:
    accuracy:
      score_name: accuracy
      score_function: sklearn.metrics.accuracy_score

```

### Custom scorer patterns

```yaml
score:
  _target_: deckard.score.base.ScorerDictConfig
  classifier : True
  scorers:
    # 1) Simple label-based scorer.
    weighted_f1:
      score_name: f1
      score_function: sklearn.metrics.f1_score
      score_params:
        average: weighted
        zero_division: 0

    # 2) Probability/logit scorer.
    roc_auc:
      score_name: roc_auc
      score_function: sklearn.metrics.roc_auc_score
      needs_proba: true
      needs_logits: false
      score_params:
        average: weighted
        multi_class: ovr

    # 3) Stage-scoped scorer (data-profile style).
    post_pipeline_mse:
      score_name: mse
      score_function: sklearn.metrics.mean_squared_error
      greater_is_better: false
      stage:
        - post-pipeline # scores on scorer(data.X,data.y)

    # 4) Stage-scoped scorer (model validation).
    model_val_accuracy:
      score_name: accuracy
      score_function: sklearn.metrics.accuracy_score
      stage:
        - val # scores on scorer(model.predict(data.X_val),data.y_val)

    # 5) Stage-scoped scorer (post-attack evaluation).
    post_attack_success:
      score_name: success
      score_function: deckard.score.attack.evasion_success_score
      stage:
        - post-attack

    # 6) Callable-from-dict specification.
    custom_metric:
      score_name: my_custom
      score_function:
        _target_: my_package.metrics.build_metric
        threshold: 0.8
      score_params:
        normalize: true

```

### ScorerConfig kwargs reference

Each entry under `scorers.<name>` is normalized into
{class}`~deckard.score.ScorerConfig` with these fields:

- `score_name`: metric name used in emitted score keys.
- `score_function`: callable, import string, or dict spec with `_target_`/`name`.
- `score_params`: default keyword arguments merged into each metric call.
- `stage`: optional stage filter token(s), for example `model-test` or
  `post-attack` (see {class}`~deckard.score.base.ScoringModelStage`,
  {class}`~deckard.score.base.ScoringAttackStage`,
  {class}`~deckard.score.base.ScoringDataStage`,
  {class}`~deckard.score.base.ScoringPipelineStage`,
  {class}`~deckard.score.base.ScoringDefenseStage`, and
  {class}`~deckard.score.base.ScoringDetectorStage`).
- `greater_is_better`: optimization direction hint for downstream consumers.
- `needs_labels`: treat `ind` as label predictions (default behavior unless
  `needs_proba: true`).
- `needs_proba`: scorer expects raw model outputs (probabilities or logits).
- `needs_logits`: convert logits to probabilities before scoring when needed.
- `binary_expand_to_multiclass`: expand 1D binary outputs to two columns for
  multiclass-style probability metrics when applicable.
- `binary_positive_class_index`: positive-class column used for binary ROC AUC.
- `row_sum_atol`: tolerance for detecting probability rows that should sum to
  about `1.0`.
- `probability_clip_eps`: numerical floor used in probability normalization.

### Runtime arguments for __call__

{meth}`~deckard.score.ScorerConfig.__call__` runtime call:

```python
ScorerConfig.__call__(dep=None, ind=None, swap=False, **kwargs)

```

- `dep`: dependent/target values (`y_true` alias is accepted).
- `ind`: independent/prediction values (`y_pred` alias is accepted).
- `swap`: swaps `dep` and `ind` before scoring.
- `**kwargs`: merged with `score_params`; unsupported kwargs are dropped when
  the metric callable does not accept variadic (`args`, `kwargs`) keyword args.

{meth}`~deckard.score.ScorerDictConfig.__call__` runtime call:

```python
ScorerDictConfig.__call__(
    mode=None,
    data=None,
    model=None,
    attack=None,
    ind=None,
    dep=None,
    score_file=None,
    **kwargs,
)

```

- `mode`: runtime scoring mode, for example `train`, `test`, `val`, `attack`,
  `attack-val`, or `pre-sample` (mapped by
  {func}`~deckard.score.base.normalize_scoring_mode`).
- `data`, `model`, `attack`: optional runtime context used to derive inputs
  when `dep`/`ind` are not passed directly.
- `dep`/`ind`: explicit targets and predictions.
- `score_file`: load/update persisted score payloads.
- `kwargs.stage`: optional runtime stage token(s) used for stage filtering.
- `kwargs.y_true`/`kwargs.y_pred`: aliases for `dep`/`ind`.
- `kwargs.y_proba`: optional raw-output override for `needs_proba` scorers.
- Remaining `kwargs`: forwarded to each scorer call along with runtime context
  keys (`data`, `model`, `attack`, `mode`).

## Internals

Score configs normalize definitions into callable maps and support both
classification and regression defaults through dedicated config classes.

The main scoring flow is:

1. A config object normalizes metric declarations into
   {class}`~deckard.score.ScorerConfig` instances.
2. {class}`~deckard.score.ScorerDictConfig` resolves import-string callables,
   filters unsupported keyword arguments against the target metric signature,
   and executes the metric.
3. Pipeline components decide which targets and predictions to pass.
4. Attack scoring adds attack-kind-specific prefixes and timing fields.

Important attack-scoring details:

- {class}`~deckard.score.attack.AttackScorerConfig` owns all attack scoring
  behavior.
- {meth}`~deckard.score.attack.AttackScorerConfig.score_evasion` chooses
  between classification and regression evasion
  profiles based on the detected task type.
- {meth}`~deckard.score.attack.AttackScorerConfig.score_membership` evaluates
  inferred membership labels against the attack
  labels (`attack.attacked_labels`).
- {meth}`~deckard.score.attack.AttackScorerConfig.score_attribute` chooses
  categorical vs regression attribute profiles and
  prefixes metrics with the targeted attribute name.
- All attack score dicts add `attack_size` and `attack_score_time`; some
  attribute paths also include `attack_generation_time`.

Because scorer profiles are normal config objects, the same metric definition
can be reused across model scoring and attack scoring. The attack layer adapts
the inputs and prefixes the outputs instead of requiring a separate metric
implementation for every attack family.

## Troubleshooting

- Ensure metric names map to importable scorer callables.
- Check expected prediction shape/type for selected metrics.
- Confirm task type (classifier/regressor) matches the active default scorer set.
- If a metric needs extra keyword arguments, ensure the callable accepts them;
  deckard drops unsupported keyword arguments based on the metric signature.
- If Hydra overrides appear to be ignored, verify that you are targeting the
  correct store group: `scorers/...` for model/data/fairness/survival and
  `attack_scorers/...` for attack profiles.
- If attack metrics are missing, look for prefixed keys such as `evasion_*`,
  `membership_inference_*`, or `inferred_age_*` rather than the raw metric
  names.
- If evasion scoring is configured with regression metrics for a classification
  attack, or vice versa, the resulting metric calculations may fail or produce
  misleading values. Match the scorer profile to the task type.

### See also

- {doc}`data` — data configuration
- {doc}`model` — model configuration and evaluation
- {doc}`attack` — attack scoring
- {doc}`experiment` — experiment orchestration
- {doc}`lifelines` — survival-specific metrics
- {doc}`anjana` — anonymization metrics

## API Reference

```{eval-rst}
.. automodule:: deckard.score
  :members:
  :show-inheritance:

```
