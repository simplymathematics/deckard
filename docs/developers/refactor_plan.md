# Refactor Plan

Use this page as the actionable TODO checklist. Keep detailed design/spec text in the dedicated developer docs linked below.

See `.github/refactor_plan` for detailed status notes and historical tracking.

## TODO Checklist

- [x] Keep canonical config declarations in `examples/sklearn/configs` and `examples/pytorch/configs`.
- [x] Remove hardcoded `ConfigStore.instance().store()` registrations from Python declaration modules.
- [x] Register declarations dynamically at package installation from `deckard/declarations.py` via `safe_store()`.
- [ ] Add/verify optional dependency gating for framework-specific registration.
- [ ] Add/verify external config root discovery through `DECKARD_CONFIG_DIRS`.
- [ ] Consolidate per-module declarations (`data`, `model`, `attack`, `defense`, `plot`, `layers`, `experiment`) into canonical YAML groups.
- [ ] Refactor tests to compose canonical configs via Hydra (compose-first, unit, experiment).
- [ ] Enforce naming conventions (`*Config`, `Default*ScoreConfig`, `*Mixin`, `*Plugin`, `modified_snake-case.yaml`).
- [ ] Run coverage + focused refactor test suites and update `.github/refactor_plan` progress.

---

**Design Specs:** [Config Declaration Architecture](config_declaration_architecture) | [Naming Conventions](naming_conventions) | [Core/Framework/Plugin Boundaries](core_framework_plugin_boundaries)
