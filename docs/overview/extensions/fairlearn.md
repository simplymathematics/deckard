# Fairlearn Plugin Overview

This overview focuses on Fairlearn execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks.md).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Model API](../../api/model)
- [Training API](../../api/train)
- [Defense API](../../api/defend)
- [Detector API](../../api/detector)
- [Scoring Overview](../scoring.md)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment.md)
- [Plot API](../../api/plot)

## Execution Order

1. Sensitive-feature-aware data and pipeline handling.
2. Fairness-aware trainer/model wrapper execution.
3. Defense stage mapping for fairness policy branches.
4. Group fairness scoring merge.
5. Canonical persistence and fairness diagnostics.

## Execution Flows

### Data Flow

```mermaid
flowchart TD
        A[Data load + sensitive attrs] --> B[fairlearn data policy hooks]
        B --> C[prepared fairness-aware split payload]
```

### Pipeline Flow

```mermaid
flowchart TD
        A[before_pipeline] --> B[fairlearn preprocessing/postprocessing stage]
        B --> C[after_pipeline]
```

### Defense Flow

```mermaid
flowchart TD
        A[fairlearn model runtime] --> B{defense type}
        B -- reductions --> C[pre_fit stage]
        B -- adversarial/postprocessing --> D[post_fit_pre_predict stage]
```

### Scoring Flow

```mermaid
flowchart TD
        A[predictions + groups] --> B[group metric scorer execution]
        B --> C[fairness merge last into score_dict]
```

### Plot Flow

```mermaid
flowchart TD
        A[persisted fairness metrics] --> B[plot adapter]
        B --> C[group fairness visual diagnostics]
```

## YAML Examples

```yaml
data:
    _target_: deckard.plugins.fairlearn.data.FairlearnDataConfig
    sensitive_columns: [sex]

model:
    _target_: deckard.plugins.fairlearn.model.FairlearnModelConfig
```

```yaml
score:
    model:
                _target_: deckard.plugins.fairlearn.score.FairlearnScorerDictConfig
        group_reduction: difference
```
