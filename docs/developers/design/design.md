# Design Principles

Deckard is built on a set of architectural and design principles that ensure
reproducibility, extensibility, and clarity for both users and contributors.

## Core Principles

- **Declarative Configuration**: All experiments are defined via YAML configs,
not code.
- **Explicit Orchestration**: All orchestration boundaries are public and deterministic.
- **Extensible by Design**: New frameworks, and metrics can be added without
modifying the core.
- **Separation of Concerns**: Core, framework, and plugin layers are strictly separated.
- **Reproducibility**: All workflows are [DVC](https://dvc.org)- and
  [Hydra](https://hydra.cc)-compatible for reproducible runs.

See the [Refactor Plan](../future/refactor_plan) for implementation details and ongoing goals.

## Canonical Runtime Decisions

These decisions are now treated as stable runtime and contributor policy:

- Unknown init kwargs remain warning-first (duck-typed passthrough allowed where needed).
- Legacy aliases (`dataset_name`, `model_type`, `attack_type`) are removed in favor of `name`.
- Defense runtime naming is standardized on `name` (not `defense_name`) for active consolidation paths.
- {class}`deckard.model.defense.base.DefenseConfig` carries the default runtime target and can host multiple plugin defenses on the same object.
- Defense configs require explicit `name` for runtime defense instantiation, while `_target_` is reserved for Hydra `*Config` initialization; raw `defense_name` fallback and shape inference are removed.
- ArtifactLoaderConfig is replaced by {class}`deckard.artifacts.ArtifactLoaderMixin` without compatibility shims.
- Experiment stage-component mapping is unified to canonical component/sub-component ownership.
- {meth}`deckard.utils.BaseConfig.fingerprint` includes all reproducibility-critical components and runs at post-init finalization.
- Canonical command names are `deckard plugins` and `deckard frameworks`.
- Plugin/framework list mode is non-installing and reports environment-dependent availability.
- Unknown plugin names hard-fail with exit code `2`.
- Install behavior is idempotent/no-op when already installed and emits summary output.
- Optional dependency policy does not auto-install ANJANA and requires both anjana and pycanon.
- Plugin/framework summaries support both human-readable and JSON output.
- Coercion/default order is child normalize -> validate -> apply defaults -> compose to parent -> parent finalization.

______________________________________________________________________

**Related:** [Naming Conventions](../contributor/naming) | {doc}`Mixin and Plugin Rules <../extensions/plugins>`
