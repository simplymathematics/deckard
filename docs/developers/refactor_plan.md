# Refactor Plan

Use this page as the actionable TODO checklist. Keep detailed design/spec text in the dedicated developer docs linked below.


## TODO Checklist

- [x] Keep canonical config declarations in `examples/sklearn/configs` and `examples/pytorch/configs`.
- [x] Remove hardcoded `ConfigStore.instance().store()` registrations from Python declaration modules.
- [x] Register declarations dynamically at package installation from `deckard/declarations.py` via `safe_store()`.
- [ ] Ensure that plugin behavior is completely outside of the core modules.
- [ ] Ensure that framework behavior is completely outside of the core modules.
- [ ] Add/verify optional dependency gating for framework-specific registration.
- [ ] Add/verify external config root discovery through `DECKARD_CONFIG_DIRS`.
- [x] Consolidate per-module declarations (`data`, `model`, `attack`, `defense`, `plot`,  `experiment`) into canonical YAML groups.
- [ ] Refactor tests to compose canonical configs via Hydra (compose-first, unit, experiment).
- [ ] Enforce naming conventions (`*Config`, `Default*ScoreConfig`, `*Mixin`, `*Plugin`, `modified_snake-case.yaml`).
- [ ] Run coverage + focused refactor test suites and update `docs/developers/refactor_plan` progress.

---

**Design Specs:** [Config Declaration Architecture](config_declaration_architecture) | [Naming Conventions](naming_conventions) | [Mixin and Plugin Rules](mixin_plugin_rules)
