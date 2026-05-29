# Lifelines Plugin Overview

This overview focuses on Lifelines execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Model API](../../api/model)
- [Defense API](../../api/defend)
- [Scoring Overview](../scoring)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment)
- [Plot API](../../api/plot)

## Execution Order

1. Survival-oriented data preparation and optional pipeline transforms.
2. Lifelines model runtime execution.
3. Optional auxiliary failure derivation from attack and non-attack signals.
4. Defense branch delegation (if configured in model).
5. Survival scoring execution.
6. Survival plot rendering and persistence.

```{include} ../flowcharts.md
:start-after: <!-- lifelines-execution-flows-start -->
:end-before: <!-- lifelines-execution-flows-end -->
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
