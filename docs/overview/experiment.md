# Experiment Guide for Base Config Objects

This guide documents experiment runtime behavior for the base orchestration
configuration:

- ExperimentConfig

It covers composition of core configs, stage-aware orchestration, persistence,
and cache-aware rerun behavior.

Related APIs:

- [Experiment API](../api/experiment)
- [Data API](../api/data)
- [Model API](../api/model)
- [Attack API](../api/attack)
- [Detector API](../api/detector)
- [Score API](../api/score)
- [File API](../api/file)

## Core Concepts

### Canonical Runtime Buckets

ExperimentConfig maintains canonical runtime state:

- files: runtime artifact aliases and persistence targets
- times: canonical timing keys plus optional extensions
- score_dict: merged experiment score payload
- outputs: runtime outputs, hook graph traces, and cache metadata
- params: resolved manifest for reproducibility and cache keys

### Stage-Oriented Composition

Experiment orchestration composes data/model/attack/detector/scoring through
canonical stages:

- load
- sample
- train
- defense
- attack
- score
- persist

Stage hooks are generated programmatically from canon definitions and executed
through HookPlugin/HookBundle composition.

## Defaults

- evaluation_mode defaults to `standard`
- score_mode can override split-scoped scoring behavior
- files-only persistence is enforced via FileConfig
- runtime cache uses stage keys derived from params + stage identity

## Typical Flow

At a high level, an experiment run is:

1. resolve/normalize component configs
2. load data and prepare runtime split(s)
3. train model, apply defense, run attacks/detector
4. run experiment-level scorers
5. persist merged score payload and runtime cache metadata

## Execution Flows

### Flow 1: Single-Pass Standard Orchestration

This path executes one end-to-end run through canonical stages and stage hooks,
then persists merged score output and runtime metadata. Pipeline hooks are
executed inside DataConfig, trainer hooks are executed inside ModelConfig, and
defense hooks run both in model-defense stages and in the experiment-level
detector defense stage.

```mermaid
flowchart TD
  A[Start ExperimentConfig.__call__] --> B[before_load stage hook]
  B --> C[data load]
  C --> D[after_load stage hook]
  D --> E[before_sample stage hook]
  E --> F[data sample]
  F --> G[data pipeline hooks before_pipeline/after_pipeline]
  G --> H[after_sample stage hook]
  H --> I[before_train stage hook]
  I --> J[model trainer hooks before_train_or_load_model/after_train_or_load_model]
  J --> K[model defense stages pre_art_defense/pre_fit/post_fit_pre_predict]
  K --> L[after_train stage hook]
  L --> M[attack stage hooks]
  M --> N[defense stage hooks for detector]
  N --> O[experiment score stage hooks]
  O --> P[before_persist stage hook]
  P --> Q[persist scores/files/cache]
  Q --> R[after_persist stage hook]
```

### Flow 2: Repeated Split/Fold with Cache Reuse

For k-fold or shuffle split runs, each run index can reuse cached stage outputs
when params and stage identity match; otherwise the stage executes and rewrites
cache entries.

```mermaid
flowchart TD
  A[detect n_repeats] --> B[for each run_idx]
  B --> C{sample cache hit?}
  C -- yes --> D[rehydrate sampled split state]
  C -- no --> E[run sample stage and cache]
  D --> F{train cache hit?}
  E --> F
  F -- yes --> G[rehydrate model outputs]
  F -- no --> H[train model and cache]
  G --> I[attack/detector/score stages]
  H --> I
  I --> J[aggregate per-run scores]
```

### Flow 3: Multi-Attack + Detector + Stage-Scoped Scoring

When multiple attacks are configured, ExperimentConfig runs each attack branch,
suffixes colliding score keys by alias, optionally applies detector filtering,
and then executes experiment-level scorers. The detector branch is where the
experiment-level `defense` stage hooks are emitted.

```mermaid
flowchart TD
  A[start pipeline core outputs] --> B{multi-attack configured?}
  B -- yes --> C[iterate attack aliases]
  C --> D[attack stage hooks before/after each alias]
  D --> E[merge attack scores with alias suffix]
  E --> F[build detector attack view]
  F --> G[detector defense stage hooks]
  G --> H[merge detector scores]
  H --> I[experiment score stage hooks + custom scorer]
  I --> J[persist merged outputs]
  B -- no --> K[single attack or no attack path]
```

## Programmatic Example

```python
from deckard.experiment import ExperimentConfig

cfg = ExperimentConfig(
    data=my_data_cfg,
    model=my_model_cfg,
    attack=my_attack_cfg,
    detector=my_detector_cfg,
    files={"score_file": "outputs/experiment_scores.json"},
)

scores = cfg()
print(scores)
```

## YAML Example

```yaml
experiment:
  _target_: deckard.experiment.base.ExperimentConfig
  evaluation_mode: standard
  data: ${data}
  model: ${model}
  attack: ${attack}
  detector: ${detector}
  files:
    score_file: outputs/experiment_scores.json
    params_file: outputs/experiment_params.yaml
```

## Recommended Practices

- Use canonical component configs and avoid alternate orchestration paths.
- Keep stage behavior explicit and hook-driven.
- Persist params and scores for cache-aware reruns and trial reuse.
- Use deterministic bundle ordering for repeatable hook composition.

## Quick Checklist

- Are all runtime components composed through ExperimentConfig?
- Are stage hooks and cache metadata captured in outputs?
- Are scores and params persisted through files-only aliases?
- Are split/trial reruns stable under unchanged params?
