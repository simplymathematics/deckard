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

- {meth}[deckard.model._mixins.PruneTrialProtocol.report](../../api/model/index)
- {meth}[deckard.model._mixins.PruneTrialProtocol.should_prune](../../api/model/index)
- {meth}`optuna.trial.Trial.should_prune`

Trainer/runtime integration calls these operations during training checkpoints.

## Pruning Mixins and Runtime Components

### Trial Protocol Mixin Surface

{class}`deckard.model._mixins.PruneTrialProtocol` defines the minimal
trial contract consumed by pruning runtime logic:

- {meth}[deckard.model._mixins.PruneTrialProtocol.report](../../api/model/index) for intermediate
  metric reporting
- {meth}[deckard.model._mixins.PruneTrialProtocol.should_prune](../../api/model/index) for prune
  decision checks

This protocol keeps trainer and mixin logic compatible with Optuna Trial
objects and trial-like adapters.

### Core Pruning Mixin

{class}`deckard.model._mixins.ModelPrunerMixin` centralizes pruning decision
behavior in {meth}[deckard.model._mixins.ModelPrunerMixin.check_prune](../../api/model/index).

{meth}[deckard.model._mixins.ModelPrunerMixin.check_prune](../../api/model/index) behavior contract:

1. If `trial` is `None`, return `False` (no prune decision).
2. If a metric value is provided and trial supports reporting, call
	{meth}[deckard.model._mixins.PruneTrialProtocol.report](../../api/model/index) with
	default step `0` when `step` is omitted.
3. If trial exposes a callable
	{meth}[deckard.model._mixins.PruneTrialProtocol.should_prune](../../api/model/index), return its
	boolean result.
4. If no callable prune method exists, return `False`.

This design makes pruning deterministic for missing or partially compatible
trial backends.

### Trainer Pruning Components

Pruning-capable trainers are implemented as dedicated trainer classes:

- {class}`deckard.model.trainers.PruningTrainer`
- {class}`deckard.model.trainers.PartialFitPruningTrainer`

Both expose:

- trial
- prune_metric (default training_time)
- prune_step (default 0)

Integration rule:

- Trainers call
  {meth}[deckard.model._mixins.ModelPrunerMixin.check_prune](../../api/model/index) only when
	trial is present and the runtime config exposes
	{meth}[deckard.model._mixins.ModelPrunerMixin.check_prune](../../api/model/index).
- When pruning is requested, trainers mark `output["pruned"] = True`.

## End-to-End Pruning Control Flow

Pruning termination is two-stage by design:

1. Trainer sets `output["pruned"] = True` when prune decision is true.
2. Optimize layer checks payload via
	{meth}`deckard.layers.optimize._should_raise_trial_pruned`.
3. Runtime raises {class}`optuna.TrialPruned` in optimize execution when
	pruning is enabled and payload is marked pruned.

This separation keeps trainer code focused on runtime metrics while centralizing
trial-status termination behavior in optimize orchestration.

## Trial Access Contract

Runtime trial context must be available where trainers execute.

Preferred approach:

- resolve trial context from study metadata and Hydra job identity
- inject trial object (or adapter) through trainer params

This avoids requiring direct Trial object transport through CLI signatures.

## Trainer Integration Contract

Pruning-capable trainers must:

1. compute or select prune metric value
2. call {meth}[deckard.model._mixins.ModelPrunerMixin.check_prune](../../api/model/index)
3. mark prune intent in runtime output for diagnostics
4. raise `optuna.TrialPruned` when prune decision is true

Raising `TrialPruned` is required to stop execution and record PRUNED state.

Practical note:

- trainer layer marks prune intent (pruned=True)
- optimize layer performs the actual {class}`optuna.TrialPruned` raise

## Metric and Step Semantics

Pruning inputs must be explicit:

- prune_metric: runtime key used for
	{meth}[deckard.model._mixins.PruneTrialProtocol.report](../../api/model/index)
- prune_step: step index used for
	{meth}[deckard.model._mixins.PruneTrialProtocol.report](../../api/model/index)

If metric is missing, behavior must be deterministic and logged.

Default pruning trainer semantics:

- prune_metric defaults to training_time
- prune_step defaults to 0
- missing metric yields None value reporting path and should still resolve
	to a deterministic boolean prune result

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

Detailed failure semantics:

- Missing trial: pruning path is a no-op (False), training continues.
- Missing trial
	{meth}[deckard.model._mixins.PruneTrialProtocol.should_prune](../../api/model/index): treated as
	non-pruning backend (False).
- Missing metric key: prune check still executes with deterministic behavior,
	and runtime logging should capture missing key context.
- Backend errors during
	{meth}[deckard.model._mixins.PruneTrialProtocol.report](../../api/model/index): fail explicitly; do
	not silently convert to success status.

## Cross-Document Dependencies

- Execution boundaries and score contract: [Optimization Runtime Contract](../optimization/optimization)
- Orchestration and callback lifecycle: [Hydra and Optuna Orchestration Contract](../optimization/hydra)
- Reporting and artifact mapping: [DVC Pipeline Autogeneration Spec](../optimization/dvc)

## Test Requirements

At minimum, tests must cover:

- prune report + should_prune invocation path
- `TrialPruned` propagation and trial state correctness
- behavior with absent trial context
- metric/step configuration edge cases
- post-prune artifact and logging behavior

Recommended mixin-focused tests:

- {meth}[deckard.model._mixins.ModelPrunerMixin.check_prune](../../api/model/index) with:
	- trial=None
	- missing {meth}[deckard.model._mixins.PruneTrialProtocol.should_prune](../../api/model/index)
	- explicit value and step
	- omitted step (defaults to 0)
- {class}`deckard.model.trainers.PruningTrainer` and
	{class}`deckard.model.trainers.PartialFitPruningTrainer` parity checks for
	prune_metric/prune_step behavior
- optimize-layer guard behavior for pruning-enabled vs pruning-disabled runs
