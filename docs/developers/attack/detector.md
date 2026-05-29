# Detector Design and Contract

## Purpose

Define internal detector runtime owner contracts for detector stage behavior,
mode-scoped scoring inputs, hook integration boundaries, and persistence
invariants across core, plugin, and framework-adapter runtimes.

## Capabilities

- Define canonical detector stage lifecycle semantics.
- Define detector score payload merge and persistence invariants.
- Define deterministic detector hook integration boundaries.
- Define plugin and framework-adapter responsibilities without replacing core detector orchestration.

## Detector Runtime Contract

Canonical detector runtime behavior is centered on
{class}`deckard.detector.base.DetectorConfig`.

Expected runtime outputs include:

- detector artifacts (`detector_model_file`, `detected_predictions_file`)
- detector score payloads merged into canonical score structures
- detector timing keys (`detector_training_time`, `detector_detection_time`)

Detector runtimes must preserve files-only persistence through canonical file
aliases and merge-safe score payload behavior.

## Stage and Hook Semantics

Detector execution is stage-oriented and hook-aware.

- Detector stage tokens must remain canonical and deterministic.
- Hook naming should preserve `before_<stage>` and `after_<stage>` contracts.
- Detector stages must remain distinct from split score modes.

Detector-specific hooks should be composed through the shared plugin runtime
layer rather than by embedding hook dispatch logic in detector family classes.

## Boundaries and Ownership

Detector runtime owners should keep detector orchestration in core detector
configs and place backend-specific behavior in adapters.

Framework adapters and plugin detector wrappers may define:

- backend-specific detector model construction
- adapter-specific payload transforms
- adapter-specific validation and defaults

They should not redefine canonical detector stage ordering, score merge
semantics, or files-only persistence behavior.

## Validation and Guardrails

Primary validation targets:

- detector component tests under `test/test_detector/`
- experiment integration tests under `test/test_experiment/`

Guardrails should verify:

- stage token normalization and deterministic ordering
- score merge compatibility with experiment aggregation
- persistence compatibility for detector artifact aliases

## See also

- {doc}`/api/detector/index`
- {doc}`/developers/experiment/experiment`
- {doc}`/developers/extensions/hooks`
- {doc}`/developers/extensions/plugins`
