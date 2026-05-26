# ANJANA Plugin Overview

This overview focuses on ANJANA execution order.

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

1. Data load and pre-sample privacy hook execution.
2. Optional pipeline transforms with policy constraints.
3. Defense stage alignment with model runtime where applicable.
4. Privacy-oriented scorer merge.
5. Canonical persistence and optional plotting.

## Execution Flows

### Data Flow

```mermaid
flowchart TD
    A[Data load] --> B[before_sample privacy hook]
    B --> C[sampled privacy-aware payload]
```

### Pipeline Flow

```mermaid
flowchart TD
    A[privacy-constrained input] --> B[pipeline stages]
    B --> C[policy-compliant transformed payload]
```

### Defense Flow

```mermaid
flowchart TD
    A[model runtime with ANJANA] --> B[map defense to pre_art_defense]
    B --> C[apply defense path]
```

### Scoring Flow

```mermaid
flowchart TD
    A[privacy run outputs] --> B[anjana scorer execution]
    B --> C[merge privacy metrics into score_dict]
```

### Plot Flow

```mermaid
flowchart TD
    A[persisted privacy metrics/artifacts] --> B[plot adapter]
    B --> C[render privacy diagnostics]
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
TODO
```
