# ANJANA Plugin Overview

This overview focuses on ANJANA execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](/developers/extensions/hooks).

Related docs:

- [Data API](/api/data/index)
- [Pipeline API](/api/data/pipeline)
- [Model API](/api/model/index)
- [Defense API](/api/model/defend)
- [Scoring Overview](../scoring)
- [File API](/api/file/index)
- [Artifacts API](/api/artifacts/index)
- [Experiment Guide](../experiment)
- [Plot API](/api/plot/index)

## Execution Order

1. Data load and pre-sample privacy hook execution.
2. Optional pipeline transforms with policy constraints.
3. Defense stage alignment with model runtime where applicable.
4. Privacy-oriented scorer merge.
5. Canonical persistence and optional plotting.

```{include} ../flowcharts.md
:start-after: <!-- anjana-data-overview-start -->
:end-before: <!-- anjana-data-overview-end -->
```

## YAML Examples

You can use dedicated Config objects to execute anjana workflows for
data-preprocessing:

```yaml
data:
  _target_: deckard.plugins.anjana.data.AnjanaDataConfig
  anjana_defense:
    k: 2
```

and for data scoring:

```yaml
score:
    _target_: deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig
   # These are included by default:
   # k_anonymity
   # t_closesness
   # l_diversity
```

Alternatively, you can incoporate anjana behavior into the base workflow:

```yaml
data:
    _target_: deckard.data.DataConfig
    name: make_classification
    classifier: true
    sensitive_columns: [sex]

model:
    _target_: deckard.model.ModelConfig
    name: sklearn.linear_model.LogisticRegression
    classifier: true

# Compose ANJANA scorers directly into the base score chain.
score:
    _target_: deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig

experiment:
    _target_: deckard.experiment.ExperimentConfig
    score_mode: test
```

Integration notes:

- Use [DefaultAnjanaDataScorerDictConfig](../../api/modules) when you only need data/privacy metrics.
- Use [DefaultAnjanaModelScorerDictConfig](../../api/modules) when you only need model-level privacy metrics.
- Keep `sensitive_columns` aligned with the runtime data split used during scoring.
