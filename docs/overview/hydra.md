# Hydra Overview

This page summarizes Hydra orchestration behavior for Deckard optimize runs.

Hydra execution supports one default profile that can run:

- a single composed experiment (`--run`)
- a multirun sweep (`--multirun`)

The same callback adapter is used in both modes:

- {class}`deckard.layers.optimize.DefaultOptimizerCallback`

The callback now self-configures from top-level config policy keys such as:

- `directions`
- `optimizers`
- `report_trial_attrs`
- `pruning_enabled`
- `dvclive_enabled`
- `dvclive_dir`

No nested optimizer policy dictionary is required in the callback block.

## Minimal Callback Block

```yaml
hydra:
  callbacks:
    deckard_optuna:
      _target_: deckard.layers.optimize.DefaultOptimizerCallback
```

## Stage-Dependent Experiment Identity

When `experiment_name` is not explicitly provided, optimize runtime computes a
stage-dependent hash payload so only stage-relevant configuration contributes to
identity.

This keeps cache/artifact identity stable across unrelated config edits.

## Related Docs

- [Optimization](optimize)
- [Hydra and Optuna Orchestration Contract](../developers/hydra)
- [Optimization Runtime Contract](../developers/optimization)
- [Pruning Runtime Contract](../developers/pruning)
