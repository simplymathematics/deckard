# Transformers Framework Overview

This overview focuses on execution order for transformers framework wrappers.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](/developers/extensions/hooks).

Related docs:

- [Data API](/api/data/index)
- [Pipeline API](/api/data/pipeline)
- [Model API](/api/model/index)
- [Training API](/api/model/train)
- [Defense API](/api/model/defend)
- [Attack API](/api/attack/index)
- [Scoring Overview](../scoring)
- [File API](/api/file/index)
- [Artifacts API](/api/artifacts/index)
- [Experiment Guide](../experiment)
- [Plot API](/api/plot/index)

## Execution Order

1. Data split and tokenizer/encoding preparation.
2. Transformer wrapper and trainer strategy execution.
3. Defense stage mapping where configured.
4. Split/stage scoring and score merge.
5. Artifact persistence and optional plotting.

```{include} ../flowcharts.md
:start-after: <!-- transformers-execution-flows-start -->
:end-before: <!-- transformers-execution-flows-end -->
```

## YAML Examples

```yaml
model:
    _target_: deckard.frameworks.transformers.model.HuggingFacePytorchModelConfig
    name: transformers.AutoModelForSequenceClassification
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
