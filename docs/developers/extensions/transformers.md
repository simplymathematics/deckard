# Transformers Design and Contract

## Purpose and Rationale

Document framework-integration boundaries for transformer runtime components
across declaration wrappers, model adaptation, and attack integration paths.

## Internal Architecture

Transformers integration layers tokenizer-aware and ART-compatible adapters on
top of canonical runtime contracts while reusing shared data and experiment
orchestration behavior.

Primary implementation modules/classes:

- {mod}`deckard.frameworks.transformers.declarations`
- {mod}`deckard.frameworks.transformers.model` and
  {class}`deckard.frameworks.transformers.model.HuggingFacePytorchModelConfig`
- {class}`deckard.frameworks.transformers.declarations.HuggingFaceArtModelWrapper`

## Execution Model

Canonical flow follows core runtime order while substituting transformer-aware
wrappers for encoded inputs and ART estimator compatibility.

## Contracts and Invariants

- Core runtime keys (files/times/scores) must remain compatible.
- Tokenized input handling must preserve integer token semantics.
- Wrapper logic must not bypass canonical experiment stage ordering.
- Transformer model adaptation must remain compatible with stage-aware attack
  and scoring integration.

## Extension Points

- New transformer declaration variants can extend
  {mod}`deckard.frameworks.transformers.declarations`.
- Optional plugin integrations may layer additional attack/scoring behavior on
  top of transformer wrappers.

## Validation and Guardrails

Guardrails include tokenizer-shape compatibility checks, device reconciliation,
and ART estimator interface validation for transformer model wrappers.

## Migration and Compatibility

Transformers integration should preserve compatibility aliases for prior config
targets and keep declaration pathways stable for existing experiment configs.

## See also

- {doc}`/api/transformers/index`
- {doc}`/overview/extensions/transformers`
- {doc}`../index`