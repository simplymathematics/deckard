# Yellowbrick Plugin Overview

This overview focuses on Yellowbrick plugin execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../developers/plugin_hook_execution.md).

Related docs:

- [Data Guide](data.md)
- [Pipeline Guide](pipeline.md)
- [Model Guide](model.md)
- [Defense Guide](defense.md)
- [Scoring Guide](scoring.md)
- [Files Guide](file.md)
- [Artifacts Guide](artifacts.md)
- [Experiment Guide](experiment.md)
- [Plot Guide](plot.md)

## Execution Order

1. Read model/experiment artifacts from canonical outputs.
2. Apply optional pre-diagnostic data preparation.
3. Use defense-aware model outputs from source runtimes.
4. Consume scorer outputs and diagnostic targets.
5. Render yellowbrick diagnostics and persist plot artifacts.

## Execution Flows

### Data Flow

```mermaid
flowchart TD
    A[experiment/model artifacts] --> B[yellowbrick input adapter]
    B --> C[diagnostic payload]
```

### Pipeline Flow

```mermaid
flowchart TD
    A[diagnostic payload] --> B[optional pre-diagnostic transform]
    B --> C[yellowbrick runtime input]
```

### Defense Flow

```mermaid
flowchart TD
    A[defended model outputs] --> B[yellowbrick consumes defended predictions]
    B --> C[no plugin-local defense execution]
```

### Scoring Flow

```mermaid
flowchart TD
    A[score artifacts] --> B[diagnostic metric selection]
    B --> C[chart annotations and summaries]
```

### Plot Flow

```mermaid
flowchart TD
    A[yellowbrick backend adapter] --> B[render diagnostic figure]
    B --> C[persist figure asset + metadata]
```

## YAML Examples

```yaml
plot:
  _target_: deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig
  files:
    plot_file: outputs/yellowbrick_diagnostic.png
```
