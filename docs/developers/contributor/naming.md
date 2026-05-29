# Naming Conventions

Deckard enforces strict naming conventions for all config, plugin, and mixin
objects to ensure clarity and composability.

## Configs

- Canonical form: `<Framework><Type>Config` (e.g., {class}`deckard.frameworks.pytorch.model.PytorchModelConfig`)
- All configs must inherit from {class}`deckard.utils.BaseConfig`

## Scorers

- Use `Default<Extension>ScoreConfig` (e.g., `DefaultFairlearnScoreConfig`)
- Must be extendable and context-aware

## Mixins

- Canonical form: `<Extension><Capability>Mixin` (e.g., `PipelineMixin`)
- Must be dataclasses with at least one public method

## Plugins

- Canonical form: `<Extension><Capability>Plugin` (e.g., `ScorePlugin`)
- Must compose one or more mixins and implement `__call__`

## YAML and Python Naming

- YAML: snake_case with dashes for aliases (e.g., `model/fairlearn-classifier.yaml`)
- Python: PascalCase for all public classes

## Canonical Identity and Command Naming

- Use `name` as the canonical identity field in runtime-facing config payloads.
- Do not introduce new constructor aliases for `dataset_name`, `model_type`, `attack_type`, or `defense_name`.
- Use `deckard plugins` and `deckard frameworks` as the canonical CLI command surfaces.
- Treat plugin and framework list mode (`--list`) as reporting-only behavior (no install side effects).
