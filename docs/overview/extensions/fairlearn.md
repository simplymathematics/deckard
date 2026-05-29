# Fairlearn Plugin Overview

This overview focuses on Fairlearn execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](/developers/extensions/hooks).

Related docs:

- [Data API](/api/data/index)
- [Pipeline API](/api/data/pipeline)
- [Model API](/api/model/index)
- [Training API](/api/model/train)
- [Defense API](/api/model/defend)
- [Detector API](/api/detector/index)
- [Scoring Overview](../scoring)
- [File API](/api/file/index)
- [Artifacts API](/api/artifacts/index)
- [Experiment Guide](../experiment)
- [Plot API](/api/plot/index)

## Execution Order

1. Sensitive-feature-aware data and pipeline handling.
2. Fairness-aware trainer/model wrapper execution.
3. Defense stage mapping for fairness policy branches.
4. Group fairness scoring merge.
5. Canonical persistence and fairness diagnostics.

```{include} ../flowcharts.md
:start-after: <!-- fairlearn-execution-flows-start -->
:end-before: <!-- fairlearn-execution-flows-end -->
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
