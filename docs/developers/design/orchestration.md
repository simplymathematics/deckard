# Orchestration Guide

This page documents how Deckard coordinates config, mixin, plugin, and hook
execution at runtime.

## Purpose

Orchestration in Deckard defines ownership boundaries: core config objects own
stage flow, while mixins and plugins extend behavior at explicit boundaries.

## Capabilities

- Stage-aware execution for data, model, defense, attack, detector, and score flows.
- Deterministic extension integration through plugin/mixin composition.
- Consistent runtime payload propagation across framework and plugin layers.
- Reproducible persistence and timing capture at each stage.

## Ownership Model

- Core runtime owners: DataConfig, ModelConfig, AttackConfig, DetectorConfig, ExperimentConfig.
- Mixins: reusable capability units used by configs/plugins.
- Plugins: extension entrypoints that compose mixins and dispatch behavior.
- Hooks: stage-level integration surface for before/after runtime boundaries.

## Design Boundaries

- Keep orchestration ownership in core config runtimes.
- Keep plugin implementations thin and policy-specific.
- Keep mixins capability-focused and non-orchestrating.
- Keep hook contracts explicit and stage-scoped.

## See also

- {doc}`../plugins`
- {doc}`../mixins`
- {doc}`../hooks`
- {doc}`../model`
- {doc}`../data`

## Purpose and Rationale

Define ownership boundaries, design intent, and tradeoffs for this domain.

## Internal Architecture

Describe runtime components, data flow, and orchestration boundaries.

## Execution Model

Describe canonical stage ordering and lifecycle semantics.

## Contracts and Invariants

Define non-negotiable behavior guarantees and invariant runtime contracts.

## Extension Points

Describe framework/plugin extension surfaces and constraints.

## Validation and Guardrails

List failure modes, guardrails, and validating tests.

## Migration and Compatibility

Document migrations, aliases, and compatibility expectations.

## See also

- {doc}`../../api/modules`
