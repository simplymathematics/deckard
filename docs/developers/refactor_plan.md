# Refactor Plan

Use this page as the actionable TODO checklist. Keep detailed design/spec text in the dedicated developer docs linked below.

See the [full plan](../../.github/refactor_plan.md) for detailed status notes and historical tracking.

## TODO Checklist

- [ ] Keep canonical config declarations in `examples/sklearn/configs` and `examples/pytorch/configs`.
- [ ] Remove hardcoded `ConfigStore.instance().store()` registrations from Python declaration modules.
- [ ] Register declarations dynamically at runtime from `deckard/declarations.py` via `safe_store()`.
- [ ] Add/verify optional dependency gating for framework-specific registration.
- [ ] Add/verify external config root discovery through `DECKARD_CONFIG_DIRS`.
- [ ] Consolidate per-module declarations (`data`, `model`, `attack`, `defense`, `plot`, `layers`, `experiment`) into canonical YAML groups.
- [ ] Refactor tests to compose canonical configs via Hydra (compose-first, unit, experiment).
- [ ] Enforce naming conventions (`*Config`, `Default*ScoreConfig`, `*Mixin`, `*Plugin`, `snake-case.yaml`).
- [ ] Verify adapter/public-attribute boundaries and core/framework/plugin isolation tests.
- [ ] Run coverage + focused refactor test suites and update [full plan](../../.github/refactor_plan.md) progress.

---

**Design Specs:** [Config Declaration Architecture](config_declaration_architecture.md) | [Naming Conventions](naming_conventions.md) | [Adapter Contract](adapter_contract.md) | [Core/Framework/Plugin Boundaries](core_framework_plugin_boundaries.md)
