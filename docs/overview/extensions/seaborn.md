# Seaborn Plugin Overview

This overview focuses on Seaborn plugin execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks.md).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Scoring Overview](../scoring.md)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment.md)
- [Plot API](../../api/plot)

## Execution Order

1. Read persisted data/score artifacts from canonical outputs.
2. Apply optional pre-plot data transformation policy.
3. Delegate defense paths to source runtimes (no seaborn defense ownership).
4. Consume scorer outputs for visualization inputs.
5. Render seaborn figures and persist plot artifacts.

## Execution Flows

### Data Flow

```mermaid
flowchart TD
    A[data or experiment artifact input] --> B[seaborn data adapter]
    B --> C[plot-ready dataframe payload]
```

### Pipeline Flow

```mermaid
flowchart TD
    A[plot input payload] --> B[optional pre-plot transform]
    B --> C[seaborn render input]
```

### Defense Flow

```mermaid
flowchart TD
    A[source model/experiment defense outputs] --> B[seaborn reads defended artifacts]
    B --> C[no plugin-local defense execution]
```

### Scoring Flow

```mermaid
flowchart TD
    A[persisted score payload] --> B[metric selection for charting]
    B --> C[annotated seaborn visualization]
```

### Plot Flow

```mermaid
flowchart TD
    A[seaborn backend adapter] --> B[render statistical figure]
    B --> C[persist figure asset and metadata]
```

## YAML Examples

```yaml
plot:
  _target_: deckard.plugins.seaborn.plot.SeabornPlotConfig
  files:
    plot_file: outputs/seaborn_summary.png
```