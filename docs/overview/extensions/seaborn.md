# Seaborn Plugin Overview

This overview focuses on Seaborn plugin execution order.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Scoring Overview](../scoring)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment)
- [Plot API](../../api/plot)

## Execution Order

1. Read persisted data/score artifacts from canonical outputs.
2. Apply optional pre-plot data transformation policy.
3. Delegate defense paths to source runtimes (no seaborn defense ownership).
4. Consume scorer outputs for visualization inputs.
5. Render seaborn figures and persist plot artifacts.

```{include} ../flowcharts.md
:start-after: <!-- seaborn-execution-flows-start -->
:end-before: <!-- seaborn-execution-flows-end -->
```

## YAML Examples

```yaml
plot:
  _target_: deckard.plugins.seaborn.plot.SeabornPlotConfig
  files:
    plot_file: outputs/seaborn_summary.png
```
