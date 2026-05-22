# Transformers Framework Overview

This overview focuses on execution order for transformers framework wrappers.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../developers/plugin_hook_execution.md).

Related docs:

- [Data Guide](data.md)
- [Pipeline Guide](pipeline.md)
- [Model Guide](model.md)
- [Trainer Guide](trainer.md)
- [Defense Guide](defense.md)
- [Attack Guide](attack.md)
- [Scoring Guide](scoring.md)
- [Files Guide](file.md)
- [Artifacts Guide](artifacts.md)
- [Experiment Guide](experiment.md)
- [Plot Guide](plot.md)

## Execution Order

1. Data split and tokenizer/encoding preparation.
2. Transformer wrapper and trainer strategy execution.
3. Defense stage mapping where configured.
4. Split/stage scoring and score merge.
5. Artifact persistence and optional plotting.

## Execution Flows

### Data Flow

```mermaid
flowchart TD
        A[DataConfig split payload] --> B[tokenizer and encoding adapters]
        B --> C[transformers runtime inputs]
```

### Pipeline Flow

```mermaid
flowchart TD
        A[pipeline transforms] --> B[encoded feature payload]
        B --> C[transformers trainer consumption]
```

### Defense Flow

```mermaid
flowchart TD
        A[transformers model state] --> B{defense configured?}
        B -- yes --> C[canonical defense stage dispatch]
        C --> D[defended inference/train path]
        B -- no --> E[baseline transformers path]
```

### Scoring Flow

```mermaid
flowchart TD
        A[predictions/probabilities] --> B[mode and stage selection]
        B --> C[scorer execution]
        C --> D[persist score artifacts]
```

### Plot Flow

```mermaid
flowchart TD
        A[persisted transformer outputs] --> B[plot adapter]
        B --> C[render diagnostics]
```

## YAML Examples

```yaml
model:
    _target_: deckard.frameworks.transformers.model.TransformersModelConfig
    model_type: transformers.AutoModelForSequenceClassification
    tokenizer: transformers.AutoTokenizer
```

```yaml
score:
    model:
        scorers:
            f1:
                score_name: f1
                score_function: sklearn.metrics.f1_score
                score_params:
                    average: weighted
```
