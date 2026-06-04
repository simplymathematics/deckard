# PyTorch Framework Overview

This overview focuses on execution order for PyTorch framework wrappers.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](/developers/extensions/hooks).

Related docs:

- [Data API](/api/data/index)
- [Pipeline API](/api/data/pipeline)
- [Model API](/api/model/index)
- [Training API](/api/model/train)
- [Defense API](/api/model/defend)
- [Attack API](/api/attack/index)
- [Detector API](/api/detector/index)
- [Scoring Overview](../scoring)
- [File API](/api/file/index)
- [Artifacts API](/api/artifacts/index)
- [Experiment Guide](../experiment)
- [Plot API](/api/plot/index)

## Execution Order

1. Data tensors/dataloader payload preparation.
2. PyTorch trainer execution (fit/load strategy).
3. Defense stage mapping and application.
4. Mode/stage scoring and score merge.
5. Checkpoint/artifact persistence and plot consumption.

```{include} ../flowcharts.md
:start-after: <!-- core-experiment-overview-start -->
:end-before: <!-- core-experiment-overview-end -->
```

## YAML Examples

```yaml
model:
    _target_: deckard.frameworks.pytorch.model.PytorchModelConfig
    trainer:
        _target_: deckard.model.trainer.base.PytorchTrainerConfig
    model_params:
        epochs: 3
        lr: 0.001
```

```yaml
score:
    model:
        scorers:
            accuracy:
                score_name: accuracy
                score_function: sklearn.metrics.accuracy_score
```
