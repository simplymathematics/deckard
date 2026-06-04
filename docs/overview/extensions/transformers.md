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

## YAML Example

```yaml
name: deckard.frameworks.transformers.declarations.GenericFlexibleTransformer
model_params:
  model_name: distilbert-base-multilingual-cased
  model_revision: main
  pretrained: true
  out_features: 64
  num_classes: 2
  return_features: false

classifier: true
_target_: deckard.frameworks.transformers.model.HuggingFacePytorchModelConfig
fit_params:
  nb_epochs: 1
  batch_size: 16
  verbose: false
library: pytorch
criterion: CrossEntropyLoss
optimizer:
  name: SGD
  lr: 0.01
  momentum: 0.9
clip_values: [0, 1]
alias: hf_mbart_en_fr
```
```{include} ../flowcharts.md
:start-after: <!-- core-experiment-overview-start -->
:end-before: <!-- core-experiment-overview-end -->
```
