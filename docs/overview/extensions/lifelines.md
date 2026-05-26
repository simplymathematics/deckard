# Lifelines Plugin Overview

This overview focuses on Lifelines execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks.md).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Model API](../../api/model)
- [Defense API](../../api/defend)
- [Scoring Overview](../scoring.md)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment.md)
- [Plot API](../../api/plot)

## Execution Order

1. Survival-oriented data preparation and optional pipeline transforms.
2. Lifelines model runtime execution.
3. Defense branch delegation (if configured in model).
4. Survival scoring execution.
5. Survival plot rendering and persistence.

## Execution Flows

### Data Flow

```mermaid
flowchart TD
        A[data load/split] --> B[survival target/time preparation]
        B --> C[lifelines-ready payload]
```

### Pipeline Flow

```mermaid
flowchart TD
        A[pipeline transforms] --> B[survival feature engineering]
        B --> C[model runtime input]
```

### Defense Flow

```mermaid
flowchart TD
        A[lifelines model path] --> B{defense configured?}
        B -- yes --> C[delegate to canonical model defense stages]
        B -- no --> D[baseline survival path]
```

### Scoring Flow

```mermaid
flowchart TD
        A[survival predictions] --> B[c-index and survival metrics]
        B --> C[merge and persist score artifacts]
```

### Plot Flow

```mermaid
flowchart TD
        A[persisted survival outputs] --> B[survival plot backend]
        B --> C[render and persist charts]
```

## YAML Examples

```yaml
model:
        _target_: deckard.plugins.lifelines.model.SurvivalModelConfig

score:
    model:
        scorers:
            c_index:
                score_function: lifelines.utils.concordance_index
```
