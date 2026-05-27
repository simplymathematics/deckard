# Plot Design and Contract

## Purpose

Define internal plotting runtime ownership, backend dispatch behavior, and
artifact persistence constraints.

## Capabilities

- Define deterministic backend stage dispatch from runtime configuration.
- Define plotting runtime owner boundaries for input hydration and rendering.
- Define persistence invariants for plot outputs and metadata.
- Define plugin and framework-adapter boundaries for backend-specific logic.

## Why This Exists

Plotting requires backend-specific preparation paths while preserving a
consistent configuration and output contract.

## Internal Architecture

Plot runtimes route through backend config families and optional plugin
adapters while preserving shared plot output semantics.

## Execution Model

Canonical plot lifecycle:

`resolve backend -> prepare inputs -> render -> persist`.

Experiment-aware backends may hydrate runtime context before rendering.

## Contracts and Invariants

- Backend dispatch must be deterministic from configuration.
- Plot outputs must persist through explicit output-file policies.
- Backend-specific behavior must not alter core runtime contracts.
- Input data preparation must remain explicit and reproducible.

## Extension Points

- Backend plugin additions (for example Yellowbrick or survival plotting).
- Shared style/runtime hooks for plot execution.
- Optional experiment/data hydration adapters.

## Failure Modes and Guardrails

- Missing backend dependencies.
- Input schema mismatch for configured plot type.
- Missing source data or score files.

## Tests and Validation

Primary plotting tests live under `test/test_plot/` and integration coverage in
`test/test_layers/`.

## Migration Notes

Keep backend-specific migrations isolated to plugin modules and avoid changing
core plot config surface without compatibility aliases.

## See also

- {doc}`../api/plot`
- {doc}`orchestration`
- {doc}`persistence`
- {doc}`plugins`
