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

## Choosing the Right Abstraction

- Choose a `*Config` when the feature owns lifecycle routing, public runtime identity, or persistence contracts.
- Choose a mixin when the feature is reusable capability logic across multiple owners.
- Choose a plugin when the feature is optional and should attach through explicit hook boundaries.
- Choose a subobject when the feature is a bounded component inside a config-owned flow.

Do not move stage ordering ownership into plugins or subobjects. Keep orchestration
order normalization in core config owners and shared orchestration mixins.

## Design Boundaries

- Keep orchestration ownership in core config runtimes.
- Keep plugin implementations thin and policy-specific.
- Keep mixins capability-focused and non-orchestrating.
- Keep hook contracts explicit and stage-scoped.

## Canonical Orchestration Decisions

- Runtime orchestration keeps canonical component/sub-component ownership boundaries.
- Legacy alias dispatch paths are removed from active refactor slices.
- Plugin/framework command and summary contracts are canonicalized at the CLI surface.
- Optional dependency boundaries remain explicit and are enforced by runtime gating.
- Coercion/default precedence follows child-first normalization and parent finalization for derived values.

## See also

- {doc}`/developers/extensions/plugins`
- {doc}`/developers/extensions/mixins`
- {doc}`/developers/extensions/hooks`
- {doc}`/developers/model/model`
- {doc}`/developers/data/data`
