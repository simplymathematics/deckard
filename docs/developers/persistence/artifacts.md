# Artifacts Design and Contract

## Purpose and Rationale

Define artifact serialization contracts used by runtime components for files,
scores, and metadata payloads.

## Internal Architecture

Artifact handling is centered on canonical loaders/savers in the artifact
layer and consumed by data/model/attack/detector/experiment runtimes.
The architecture separates object normalization from persistence adapters so
runtime code can remain backend-agnostic.

Primary implementation objects include:

- {class}`deckard.artifacts.ScoreDict`
- artifact load/save helpers used by runtime configs and persistence layers

## Execution Model

Canonical flow is `normalize -> serialize/deserialize -> merge -> persist`.
Runtime components delegate artifact persistence rather than implementing
custom file I/O paths in each module.

## Contracts and Invariants

- Artifact payloads must remain JSON/YAML serializable.
- Score payload merges must be deterministic.
- Backward-compatible read paths must tolerate previously persisted schemas.
- ScoreDict envelope fields (`payload`, flattened views) must remain compatible
	with existing consumers.

## Extension Points

- New artifact backends may be added through explicit loader/saver adapters.
- Runtime modules may add payload scopes without breaking base schema keys.

## Validation and Guardrails

Guardrails include invalid payload type checks, schema envelope checks, and
merge safety across nested score payloads.

Validate contract compatibility when modifying score envelope structure or
flattened projection formats.

## Migration and Compatibility

Schema versioning and compatibility envelope fields must be preserved when
adding new artifact projections.

## See also

- {doc}`/api/artifacts/index`
