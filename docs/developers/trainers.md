# Trainer Contract

Detailed contract for trainer sub-objects under model runtime.

## Purpose

Trainer objects encapsulate model fitting and related lifecycle behavior,
including pre-fit hooks and optional adversarial training integration.

## Capabilities

- Execute task-aware fit operations (fit, pruned, partial fit).
- Coordinate training-time defenses and callbacks.
- Emit timing and state metadata for orchestration layers.

## Trainer vs ART Retrainer Defense

Trainer configuration objects and ART retrainer defenses are separate layers:

- Trainer configuration objects define fit orchestration and runtime trainer
	behavior.
- ART retrainer defenses are defense-family components configured under
	`model.defense` and documented in {doc}`defenses`.

Do not treat ART retrainer defenses as replacements for trainer configuration
objects; both are required in documentation because they answer different
questions (fit orchestration vs robustness defense behavior).

## Standards Followed

- Docstring standard: {doc}`docstrings`
- Model design: {doc}`model`

## Required Documentation

- Trainer purpose and fit lifecycle scope
- `Attributes:` for trainer controls and runtime knobs
- Method-level argument and return contracts

## Purpose and Rationale

Define ownership boundaries, design intent, and tradeoffs for this domain.

## Internal Architecture

Trainer behavior is split across model runtime mixins and trainer-defense
objects:

- {mod}`deckard.model._mixins` owns fit/predict orchestration helpers.
- {class}`deckard.model._mixins.ModelTrainingMixin` owns core fit execution.
- {class}`deckard.model._mixins.PretrainedModelMixin` owns load-or-train
	fallback behavior.
- {class}`deckard.model._mixins.ModelPrunerMixin` owns trial-pruning checks.
- {class}`deckard.model.defense.trainer.TrainerDefenseConfig` owns ART
	adversarial training wrappers.

## Execution Model

Model-trainer execution follows canonical model stages:

1. Resolve/load model or construct new model object.
2. Execute training via trainer mixin methods.
3. Apply fit-time defense paths when configured.
4. Recompute predictions/scores if retraining occurred.

The fit-time defense path is distinct from standard fit orchestration and may
trigger retraining after a pre-defense state snapshot.

## Contracts and Invariants

- Trainer configuration objects own fit orchestration, not defense-family
	policy.
- Fit execution must preserve timing/state keys consumed by experiment layers.
- Trainer runtime behavior must remain callable through documented public model
	runtime methods.
- Pruning checks must surface deterministic prune behavior for equivalent trial
	state.

## Extension Points

Describe framework/plugin extension surfaces and constraints.

## Validation and Guardrails

Primary guardrails:

- pretrained + fit-defense rerun behavior must preserve pre-defense metrics,
- pruning behavior must propagate deterministic prune events,
- trainer-defense composition must not bypass model runtime contracts.

Validation should include model and experiment tests under
`test/test_model/` and `test/test_experiment/` for fit-time defense and
retraining behavior.

## Migration and Compatibility

Legacy trainer-adjacent aliases should be normalized to canonical trainer
runtime objects. Public trainer behavior must stay documented through
`train` and `defend` API pages without merging the two contracts.

## See also

- {doc}`../api/train`
