# Score

## Basic flow state

`normalize mode/stage -> execute scorers -> emit payload`.

## Purpose

Define user-facing scoring runtime owner behavior, including stage and mode
routing, hook-aware scorer execution, and persisted score outputs across core,
framework-adapter, and plugin-integrated runtimes.

## Capabilities

- Normalize scoring mode and stage routing across runtime contexts.
- Execute configurable scorer dictionaries for model/data/attack payloads.
- Support label, probability, and logits-based scoring patterns.
- Persist stage-scoped score outputs for downstream reporting.
- Aggregate metric payloads produced by {doc}`data`, {doc}`model`, {doc}`attack`, and {doc}`detector`.

Implementation-level scoring contracts are documented in {doc}`../developers/score`.

## Outputs

- Score dictionaries keyed by stage/mode/metric.
- Persisted score files and merged runtime score payloads.
- Stage-scoped metric keys for model, attack, detector, and data contexts.

## Introduction

This page describes score behavior from a user/API perspective: how scorers
run, how to configure them, and what outputs they produce.

Developer-level scoring contracts and internals are documented in
{doc}`../developers/score`.

For introductory scoring concepts, default scorer families, and YAML catalog
examples, see {doc}`../overview/scoring`.

## Overview

Use {class}`~deckard.score.ScorerDictConfig` to configure stage-aware and
mode-aware metric execution across data, model, attack, and detector outputs.

The scoring runtime supports:

- declarative metric configuration from import strings or callable builders,
- stage filtering (for example `post-attack`),
- mode-aware execution (`train`, `test`, `val`, and related runtime modes),
- persisted score payload updates through `score_file`.

## Integrations

- Framework integration: {doc}`pytorch`
- Plugin integrations: {doc}`fairlearn`, {doc}`lifelines`, {doc}`anjana`

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
  {class}`~deckard.score.base.ScoringDetectorStage`, and
  {class}`~deckard.score.base.ScoringDVCStage`).
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
  {func}`~deckard.score.canon.normalize_scorer_mode`).
- `data`, `model`, `attack`: optional runtime context used to derive inputs
  when `dep`/`ind` are not passed directly.
- `dep`/`ind`: explicit targets and predictions.
- `score_file`: load/update persisted score payloads.
- `kwargs.stage`: optional runtime stage token(s) used for stage filtering.
- `kwargs.y_true`/`kwargs.y_pred`: aliases for `dep`/`ind`.
- `kwargs.y_proba`: optional raw-output override for `needs_proba` scorers.
- Remaining `kwargs`: forwarded to each scorer call along with runtime context
  keys (`data`, `model`, `attack`, `mode`).

### DVC System Scorer

Use {class}`~deckard.score.DVCSystemScorerDictConfig` to generate stage-scoped
DVC system-monitor snapshots.

Default behavior:

- Runs after DVC score-hook stages when DVC plugin support is enabled.
- Emits concise `<component>_<stat>` metric keys (for example `data_cpu` and
  `defense_gpu_power`).
- Persists values with stage and mode scope in the score payload
  (`score_dict[stage][mode][metric]`).
- Includes normalized DVCLive `system_monitor/*` metrics and pre-existing power
  hook metrics (`power/<namespace>/*`) when available.
- Defaults to component after-score stages through scorer stage filters, while
  allowing canonical stage overrides through custom scorer configuration.

Example YAML:

```yaml
score:
  _target_: deckard.score.dvc.DVCSystemScorerDictConfig
```

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
