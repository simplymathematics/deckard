# Fairlearn Design and Contract

## Purpose and Rationale

Define fairness integration contracts for Fairlearn-backed data/model/score
extensions.

## Internal Architecture

Fairlearn integration composes plugin configs over canonical runtime ownership,
adding sensitive-feature handling, fairness metrics, and optional mitigation
strategies.

Primary implementation modules/classes:

- {mod}`deckard.plugins.fairlearn.data` and
	{class}`deckard.plugins.fairlearn.data.FairlearnDataConfig`
- {mod}`deckard.plugins.fairlearn.model` and
	{class}`deckard.plugins.fairlearn.model.FairlearnModelConfig`
- {mod}`deckard.plugins.fairlearn.score` and
	{class}`deckard.plugins.fairlearn.score.FairlearnScorerDictConfig`

## Execution Model

Canonical flow is `resolve sensitive-feature inputs -> execute core runtime
stages -> apply fairness scoring/mitigation hooks -> persist fairness outputs`.

## Contracts and Invariants

- Fairness integration must preserve core payload and file contracts.
- Sensitive-feature routing must be explicit and documented.
- Fairness score keys must remain merge-safe with core scoring.
- Fairness mixin behavior should not mutate non-fairness runtime payload fields
	outside documented plugin outputs.

## Extension Points

- Additional Fairlearn scorer profiles and mitigators can be added via plugin
	config groups.
- Fairlearn behavior can compose with sklearn and pytorch model wrappers.

## Validation and Guardrails

Guardrails include sensitive-column validation, fairness-metric schema checks,
and optional dependency gating.

Validate Fairlearn metric namespace stability across data/model scoring modes.

## Migration and Compatibility

Maintain compatibility aliases for fairness config targets and preserve stable
fairness metric naming where possible.

## See also

- {doc}`/api/plugins/fairlearn`
- {doc}`../index`
