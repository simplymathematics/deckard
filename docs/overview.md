# Overview

This page provides a high-level summary of the deckard package, its architecture, and core API. For conceptual background, see [concepts.md](concepts.md).

## Core API
- Data, Model, Attack, Experiment, File, and ScorerDict configuration classes
- Modular, extensible architecture using `hydra`'s ConfigStore
- Support for adversarial robustness, fairness, privacy, and survival analysis.
- Configurable sweeping, sampling, and optimizaiton with `optuna`.

See also:
- [API Reference](modules.md)
- [Extensions](extensions.md)
- [Notebooks](notebooks/index.md)
- [Developer Docs](development.md)
