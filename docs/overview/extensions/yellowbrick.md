# Yellowbrick Plugin Overview

This overview focuses on Yellowbrick plugin execution order.

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

1. Read model/experiment artifacts from canonical outputs.
2. Apply optional pre-diagnostic data preparation.
3. Use defense-aware model outputs from source runtimes.
4. Consume scorer outputs and diagnostic targets.
5. Render yellowbrick diagnostics and persist plot artifacts.

```{include} ../flowcharts.md
:start-after: "### Yellowbrick Execution Flows"
:end-before: "### Seaborn"
```

## YAML Examples

```yaml
plot:
  _target_: deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig
  files:
    plot_file: outputs/yellowbrick_diagnostic.png
```
