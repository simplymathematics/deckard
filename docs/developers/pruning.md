# Pruning Runtime Contract

This document defines the runtime pruning contract for Deckard optimization
workflows.

It covers trial reporting, prune decisions, trainer behavior, and termination
semantics.

## Goals

- Enable early stopping based on Optuna pruning rules.
- Keep pruning behavior deterministic and observable.
- Preserve Hydra and Optuna status semantics for pruned trials.
- Avoid silent prune decisions that continue full execution.

## Runtime Pruning Model

Pruning uses a trial-like runtime object with these operations:

- `report(value, step)`
- {meth}`optuna.trial.Trial.should_prune`

Trainer/runtime integration calls these operations during training checkpoints.

## Trial Access Contract

Runtime trial context must be available where trainers execute.

Preferred approach:

- resolve trial context from study metadata and Hydra job identity
- inject trial object (or adapter) through trainer params

This avoids requiring direct Trial object transport through CLI signatures.

## Trainer Integration Contract

Pruning-capable trainers must:

1. compute or select prune metric value
2. call {meth}`deckard.model._mixins.ModelPrunerMixin.check_prune`
3. mark prune intent in runtime output for diagnostics
4. raise `optuna.TrialPruned` when prune decision is true

Raising `TrialPruned` is required to stop execution and record PRUNED state.

## Metric and Step Semantics

Pruning inputs must be explicit:

- `prune_metric`: runtime key used for `report`
- `prune_step`: step index used for `report`

If metric is missing, behavior must be deterministic and logged.

## Observability Contract

Runtime artifacts should make prune behavior inspectable:

- include prune marker in output payload when decision path is reached
- log metric, step, and prune decision source
- keep trial attributes synchronized for downstream analysis

## Failure Handling

Pruning path should fail safely when:

- no trial context is available
- selected metric is missing
- trial backend update fails

Failure behavior must be explicit and avoid corrupting trial metadata.

## Cross-Document Dependencies

- Execution boundaries and score contract: [Optimization Runtime Contract](optimization)
- Orchestration and callback lifecycle: [Hydra and Optuna Orchestration Contract](hydra)
- Reporting and artifact mapping: [DVC Pipeline Autogeneration Spec](dvc)

## Test Requirements

At minimum, tests must cover:

- prune report + should_prune invocation path
- `TrialPruned` propagation and trial state correctness
- behavior with absent trial context
- metric/step configuration edge cases
- post-prune artifact and logging behavior

