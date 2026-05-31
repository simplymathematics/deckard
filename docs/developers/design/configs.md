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

- Documentation standards: {doc}`/developers/contributor/documentation`
- Naming rules: {doc}`/developers/contributor/naming`
- Design boundaries: {doc}`/developers/design/design`

## Canonical Contract Decisions

- Canonical identity fields use `name`; legacy constructor aliases are removed in active hard-cut slices.
- Defense runtime mappings use name and canonical runtime methods (apply, apply_to, apply_defense).
- Unknown init kwargs remain warning-first where duck-typed integration is required.
- Config finalization preserves explicit user values while deriving reproducibility-critical fields last.
- Coercion/default precedence is child normalize -> validate -> apply defaults -> compose to parent -> parent finalization.

## Required Sections

Use MyST-native Google-style sections:

- `Attributes:` required
- `Args:` when constructor/runtime parameters exist
- `Returns:` for non-`None` methods
- `Raises:` for explicit failures
- `Note:` for persistence/runtime side effects
