# File Design and Contract

## Purpose and Rationale

Define file-path policy, alias normalization, and runtime output wiring across
Deckard components.

## Internal Architecture

File behavior is owned by file config objects that normalize output aliases
and expose stable paths to data/model/attack/detector/experiment runtimes.

Primary implementation classes:

- {class}`deckard.file.FileConfig`
- {class}`deckard.file.AbstractFileHandler`
- {class}`deckard.file.CanonFileHandler`

## Execution Model

Canonical flow is `resolve aliases -> materialize paths -> hand off to runtime
components -> persist outputs`.

## Contracts and Invariants

- Canonical aliases remain stable across modules.
- File path resolution is deterministic for equivalent configs.
- Runtime components should consume resolved aliases instead of ad hoc paths.
- File handler parsing behavior should be centralized in file handler classes,
  not duplicated in runtime modules.

## Extension Points

- New file aliases can be added through explicit config schema updates.
- Framework/plugin runtimes may introduce scoped aliases without changing core
	alias meanings.

## Validation and Guardrails

Guardrails include missing-path detection, alias collision checks, and runtime
output-path consistency tests.

Validate alias resolution for shared keys used by score/artifact persistence.

## Migration and Compatibility

Legacy top-level file kwargs should be normalized at boundaries to canonical
file mappings.

## See also

- {doc}`/api/file/index`
