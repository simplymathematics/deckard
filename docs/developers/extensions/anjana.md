# Anjana Design and Contract

## Purpose and Rationale

Define anonymization/privacy integration contracts for Anjana plugin behavior.

## Internal Architecture

Anjana integration layers anonymization-specific data and scoring behavior over
canonical data/model/score runtime ownership.

Primary implementation modules/classes:

- {mod}`deckard.plugins.anjana.data` and
	{class}`deckard.plugins.anjana.data.AnjanaDataConfig`
- {mod}`deckard.plugins.anjana.score` and default anonymization scorer configs
- {class}`deckard.plugins.anjana.data.PrivacyBehaviorMixin`

## Execution Model

Canonical flow is `resolve anonymization config -> apply anonymization
transforms -> run downstream model/score stages -> persist anonymization
metrics`.

## Contracts and Invariants

- Anonymization behavior must preserve canonical split and file contracts.
- Privacy/anonymization score keys must remain stable and merge-safe.
- Integration must remain optional-dependency safe.
- Anonymization transforms must be applied through explicit configured stages,
  not implicit mutation during unrelated runtime steps.

## Extension Points

- New anonymization profiles and metrics can be added through plugin config
	extensions.
- Integration can compose with other scoring/model plugins when payload
	contracts are preserved.

## Validation and Guardrails

Guardrails include schema validation for anonymization outputs, optional
dependency checks, and metric compatibility tests.

Validate anonymization score key stability to preserve downstream dashboard and
optimization compatibility.

## Migration and Compatibility

Preserve compatibility aliases and stable anonymization metric naming for
existing reports and optimization traces.

## See also

- {doc}`/api/plugins/anjana`
- {doc}`../index`
