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
- **Reproducibility**: All workflows are DVC- and Hydra-compatible for
  reproducible runs.
- **Reproducibility**: All workflows are [DVC](https://dvc.org)- and
  [Hydra](https://hydra.cc)-compatible for reproducible runs.

See the [Refactor Plan](plan) for implementation details and ongoing goals.

______________________________________________________________________

**Related:** [Naming Conventions](naming) | [Mixin and Plugin Rules](plugins)
