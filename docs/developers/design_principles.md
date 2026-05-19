# Design Principles

Deckard is built on a set of architectural and design principles that ensure reproducibility, extensibility, and clarity for both users and contributors.

## Core Principles

- **Declarative Configuration**: All experiments are defined via YAML configs, not code.
- **Explicit Orchestration**: All orchestration boundaries are public and deterministic.
- **Extensible by Design**: New frameworks, plugins, and metrics can be added without modifying the core.
- **Separation of Concerns**: Core, framework, and plugin layers are strictly separated.
- **Reproducibility**: All workflows are DVC- and Hydra-compatible for reproducible runs.

See the [Refactor Plan](refactor_plan.md) for implementation details and ongoing goals.

---

**Related:** [Naming Conventions](naming_conventions.md) | [Adapter Contract](adapter_contract.md) | [Mixin and Plugin Rules](mixin_plugin_rules.md)
