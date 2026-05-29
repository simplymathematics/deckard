# Plugins

## Purpose and Rationale

Define class-level contract requirements for plugin classes that compose
runtime mixins and expose stable execution surfaces.

## Internal Architecture

Plugin classes are composed through hook/plugin infrastructure in
{mod}`deckard.plugins.base` and consumed by runtime components through explicit
hook names. Composition commonly uses {class}`deckard.plugins.base.HookBundle`
and {class}`deckard.plugins.HookPlugin`.

## Execution Model

Canonical plugin flow is `normalize config -> construct plugin runtime ->
execute explicit callable hooks -> return typed payloads`.

Hook execution order is deterministic and should align with canonical stage
hooks (`before_<stage>`, `after_<stage>`).

## Contracts and Invariants

- Plugins must expose public execution methods (including `__call__` where
	required by plugin type).
- Execution ordering must be explicit and deterministic.
- Plugins must avoid hidden mutation of caller-owned runtime payloads.

## Extension Points

- New plugin classes should reuse mixins and shared base behaviors.
- Plugin capabilities should be added through explicit class composition, not
	dynamic side effects.

Implementation additions should register explicit hook entrypoints rather than
embedding stage logic in classes.

## Validation and Guardrails

Guardrails include plugin hook-order tests, payload typing checks, and
deterministic merge/dispatch tests.

See hook contract validation patterns in {doc}`../hooks`.

## Migration and Compatibility

Naming and callable contracts for plugin classes must remain stable for config
compatibility and reproducible orchestration.
