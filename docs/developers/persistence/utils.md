# Utils Design and Contract

## Purpose and Rationale

Document shared utility contracts that support deterministic configuration,
normalization, and helper flows across the project.

## Internal Architecture

Utilities provide reusable primitives for config handling, coercion, path and
type normalization, and common runtime-safe helper behavior.

## Execution Model

Utilities are side-effect-light helper layers invoked by runtime configs and
orchestration paths.

## Contracts and Invariants

- Utility helpers must remain deterministic for equivalent inputs.
- Shared helper behavior should be centralized to avoid duplicated logic.
- Utility APIs should not mutate caller-owned payloads unless explicitly
	documented.

### Coercion and Default Precedence

Configuration normalization follows this precedence contract:

1. Parse raw inputs.
2. Coerce child-level values first (types, aliases, canonical names).
3. Validate coerced child values.
4. Apply defaults only to unresolved/missing fields.
5. Compose normalized children into the parent.
6. Run parent finalization only for derived values, without overriding explicit
	 child/user values.

This order preserves least-surprise behavior, keeps child configs authoritative
for explicitly provided values, and prevents parent-level finalization from
silently rewriting intentional runtime choices.

### Reproducibility Guardrail

- Fingerprints and hash inputs must be computed after coercion + default
	resolution + parent finalization, so equivalent effective configs produce
	stable identifiers.

## Extension Points

- Domain-specific utilities may extend shared helpers through new functions in
	dedicated modules.
- Plugins/frameworks should prefer shared helpers before introducing local
	utility variants.

## Validation and Guardrails

Guardrails include focused utility tests and regression checks in integration
tests that depend on normalized helper behavior.

## Migration and Compatibility

When utility behavior changes, dependent runtime docs and compatibility notes
must be updated together.

## See also

- {doc}`/api/utils/index`
