# sklearn Framework Overview

This overview focuses on execution order for sklearn framework wrappers.

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
- [Utils API](/api/utils/index)

## Execution Order

1. Data split and optional pipeline transform execution.
2. Model trainer strategy resolution and training/load path.
3. Defense stage mapping and application.
4. Split-scoped scoring and score merge.
5. Artifact/file persistence and optional downstream plotting.

```{include} ../flowcharts.md
:start-after: <!-- core-experiment-overview-start -->
:end-before: <!-- core-experiment-overview-end -->
```

## YAML Examples

```yaml
data:
    _target_: deckard.data.base.DataConfig
    pipeline:
        scale:
            name: sklearn.preprocessing.StandardScaler
            fit_X: true

model:
    _target_: deckard.model.base.ModelConfig
    name: sklearn.ensemble.RandomForestClassifier
    trainer:
        _target_: deckard.model.trainer.base.SklearnTrainerConfig
```

```yaml
score:
    model:
        scorers:
            accuracy:
                score_name: accuracy
                score_function: sklearn.metrics.accuracy_score
```
