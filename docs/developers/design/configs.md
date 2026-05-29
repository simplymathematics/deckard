# Config Contract

Detailed contract for public config classes (`*Config`).

## Purpose

Config classes are the main control plane for Deckard runtimes. They let users
compose datasets, models, defenses, and scoring pipelines declaratively while
keeping execution deterministic.

## Capabilities

- Capture runtime knobs in serializable fields.
- Support Hydra override workflows and reproducible experiment replay.
- Encapsulate extension-specific defaults behind a stable public API.

## Standards Followed

- Documentation standards: {doc}`../documentation`
- Naming rules: {doc}`../naming`
- Design boundaries: {doc}`../design`

## Required Sections

Use MyST-native Google-style sections:

- `Attributes:` required
- `Args:` when constructor/runtime parameters exist
- `Returns:` for non-`None` methods
- `Raises:` for explicit failures
- `Note:` for persistence/runtime side effects

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
- {doc}`../template`
