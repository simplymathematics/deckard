# Pytorch Design and Contract

## Purpose and Rationale

Document framework-integration boundaries for PyTorch runtime components across
data, model, experiment, and plotting paths.

## Internal Architecture

PyTorch integration wraps core runtime contracts with framework-specific model,
data, and checkpoint behavior while preserving top-level API orchestration.

Primary implementation modules/classes:

- {mod}`deckard.frameworks.pytorch.data` and
	{class}`deckard.frameworks.pytorch.data.PytorchDataConfig`
- {mod}`deckard.frameworks.pytorch.model` and
	{class}`deckard.frameworks.pytorch.model.PytorchModelConfig`
- {mod}`deckard.frameworks.pytorch.experiment` and
	{class}`deckard.frameworks.pytorch.experiment.TorchExperimentConfig`

## Execution Model

Canonical flow follows core runtime order while substituting torch-native
constructors, fit/predict loops, and model-state persistence adapters.

## Contracts and Invariants

- Core runtime keys (files/times/scores) must remain compatible.
- Framework-specific state serialization must be explicit.
- Wrapper logic must not bypass canonical experiment stage ordering.
- {meth}`~deckard.frameworks.pytorch.model.PytorchModelConfig.save`/
	{meth}`~deckard.frameworks.pytorch.model.PytorchModelConfig.load` config
	behavior and
	{meth}`~deckard.frameworks.pytorch.model.PytorchModelConfig.save_model`/
	{meth}`~deckard.frameworks.pytorch.model.PytorchModelConfig.load_model` runtime behavior
	must remain separate and explicit.

## Extension Points

- New torch model/data variants can extend existing pytorch config families.
- Optional plugin integrations may layer fairness/survival behavior on top of
	torch wrappers.

## Validation and Guardrails

Guardrails include device mismatch checks, checkpoint compatibility checks, and
shape/type validation across torch and scorer interfaces.

Validate torch estimator compatibility with attack/defense integrations that
expect ART estimator-like behavior.

## Migration and Compatibility

Torch integration should preserve compatibility aliases for prior config
targets and checkpoint conventions when possible.

## See also

- {doc}`/api/pytorch/index`
- {doc}`../index`
