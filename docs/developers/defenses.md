# Defense Contract

Detailed contract for defense sub-objects under model runtime.

## Purpose

Defense objects apply robustness transformations around model training or
inference stages to reduce attack success and improve reliability.

## Capabilities

- Support family-specific defense pipelines (preprocessor, trainer, detector, and more).
- Apply stage-scoped transformations deterministically.
- Integrate external defense frameworks behind a stable Deckard interface.

## ART Retrainer Defense vs Trainer Config

The `trainer` defense family in this page refers to ART retrainer defenses.

- ART retrainer defenses are defense objects executed in defense stages.
- Trainer configuration objects define model-fit orchestration and are
	documented separately in {doc}`trainers`.

Both must be documented and maintained independently because they have
different ownership boundaries and runtime semantics.

## Standards Followed

- Docstring standard: {doc}`docstrings`
- Model and defense design: {doc}`model`

## Required Documentation

- Defense family role and stage scope
- `Attributes:` for class-level defense controls
- `Raises:` for deterministic validation failures

## Purpose and Rationale

Define ownership boundaries, design intent, and tradeoffs for this domain.

## Internal Architecture

Defense implementation is centered on these runtime objects:

- {class}`deckard.model.defense.base.DefensePipelineConfig`
- {class}`deckard.model.defense.base.DefenseConfig`
- {class}`deckard.model.defense.preprocessor.PreprocessorDefenseConfig`
- {class}`deckard.model.defense.postprocessor.PostprocessorDefenseConfig`
- {class}`deckard.model.defense.trainer.TrainerDefenseConfig`
- {class}`deckard.model.defense.detector.DetectorDefenseConfig`
- {class}`deckard.model.defense.regularizer.RegularizerDefenseConfig`

Family-specific behavior is delegated through mixins in corresponding modules.

## Execution Model

Defense execution is stage-scoped and deterministic:

1. Normalize defense chain through pipeline behavior mixin.
2. Resolve defense family/subtype.
3. Apply fit-time or predict-time behavior by stage.
4. Propagate defense outputs back into model runtime state.

Trainer-family defenses are executed as defense objects, not as trainer config
objects.

## Contracts and Invariants

- Defense chain ordering must be explicit and preserved.
- Family dispatch must be deterministic for equivalent config payloads.
- Defense execution must preserve canonical model runtime contracts.
- Public defense methods should use verb-style API surfaces for detector paths.

## Extension Points

Describe framework/plugin extension surfaces and constraints.

## Validation and Guardrails

Primary guardrails:

- invalid defense family/type detection before runtime mutation,
- stage mismatch protection for fit-vs-predict behavior,
- deterministic behavior under repeated equivalent configs.

Validation should include defense tests in `test/test_model/` and integration
tests in `test/test_experiment/` for stage-ordered defense behavior.

## Migration and Compatibility

Legacy noun-style detector method names and legacy defense aliases should be
normalized to canonical public defense APIs and family names.

## See also

- {doc}`../api/defend`
