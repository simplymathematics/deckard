# Attack Design and Contract

This page captures attack-layer design goals, constraints, and acceptance
criteria for contributor-facing documentation.

## Purpose

Define internal attack runtime owner contracts for mode and stage semantics,
hook-aware orchestration boundaries, persistence behavior, and plugin/framework
adapter extensions.

## Capabilities

- Define canonical attack stage and split mode semantics.
- Define attack scoring and persistence invariants.
- Define attack-family routing guarantees for core runtime owners.
- Define plugin and framework-adapter boundaries for attack execution.

## Design Goals

1. Keep attack behavior split-scoped through mode semantics (`auto`, `train`, `test`, `val`).
2. Keep lifecycle reporting stage-scoped through canonical stage tokens (`pre-attack`, `post-attack`).
3. Keep backend-specific behavior in framework/plugin wrappers, not in core attack orchestration.
4. Preserve files-only persistence contracts for attacked outputs and score artifacts.

## Constraints

1. Attack mode must remain distinct from stage semantics.
2. Attack score payloads must stay merge-safe with experiment-level score aggregation.
3. Attack artifact persistence must route through canonical file aliases (`attack_file`, `attack_predictions_file`, `score_file`).
4. Attack-family routing must preserve explicit evasion/poisoning/extraction/inference branches.

## Acceptance Criteria

1. Core attack configuration remains {class}`deckard.attack.AttackConfig` with stable runtime behavior.
2. Canonical attack stages normalize to lifecycle boundaries and are observable in outputs/hooks.
3. Files-only persistence paths are used for attack outputs and downstream scoring artifacts.
4. Attack scoring outputs remain compatible with experiment-level merge and persistence layers.

## Guardrail Tests

- `test/test_attack/`
- `test/test_experiment/test_experiment.py`
- `test/test_experiment/test_experiment_canon.py`

## Related References

- API: {doc}`/api/attack/index`
- Workflow: {doc}`/developers/experiment/experiment`
- Scoring contract: {doc}[score](../score)
- Plugin and hook execution: {doc}`/developers/extensions/hooks`
