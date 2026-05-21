# Refactor Plan

Use this page as the actionable TODO checklist.
Keep detailed design/spec text in the dedicated developer docs linked below.
## Data Checklist:

### Phase 1: Core data runtime (`deckard/data`)

- [x] Introduce canonical data scoring stage vocabulary and normalization.
- [x] Route `DataConfig` score execution through canonical stage resolution.
- [x] Add stage-aware score hook dispatch with legacy hook compatibility.
- [ ] Define one explicit runtime attribute contract for all data stages
  (`_X/_y`, train/test/val splits, timing fields, score dict).
- [ ] Make stage-to-split routing explicit and centralized (no per-module
  fallback aliases).
- [ ] Split `DataConfig` orchestration responsibilities into explicit loaders,
  samplers, pipeline runners, and score runners.
- [ ] Move remaining plugin/framework branching logic out of core runtime paths.
- [ ] Replace tests that depend on private internals with public DataConfig
  contract tests.

### Phase 2: Framework data runtimes (`deckard/frameworks/**/data.py`)

- [ ] Align framework data configs to the core DataConfig lifecycle method
  contract (`_load_data`, `_sample`, `_score`, `__call__`).
- [ ] Unify stage semantics with core canonical stage resolver.
- [ ] Unify hook semantics with core stage-driven hook dispatch.
- [ ] Deduplicate sampler logic between framework data modules and
  framework sampler helpers.
- [ ] Ensure framework modules expose only framework-specific adapters (tensor,
  dataloader, device), not alternate orchestration flows.

### Phase 3: Plugin data runtimes (`deckard/plugins/**/data.py`)

- [ ] Convert plugin data configs into policy layers on top of canonical runtime
  behavior (not replacement runtimes).
- [ ] Keep only plugin-specific concerns in plugin modules:
  sensitive features, mitigation transforms, plugin scorers, mode validation.
- [ ] Canonicalize plugin hook names to stage-scoped before/after semantics.
- [ ] Remove non-top-level compatibility aliases/import paths for plugin data
  internals.

### Phase 4: Contracts, docs, and migration guards

- [ ] Add cross-family contract tests asserting unified attributes, methods, and
  control-flow.
- [ ] Add stage and hook conformance tests for core/framework/plugin families.
- [ ] Update API docs (`data`, `pytorch`, `fairlearn`) to reflect canonical
  stage and hook behavior.
- [ ] Document migration constraints: preserve top-level Config APIs only.
- [ ] Run focused suites + coverage, then mark checklist completion.




## Overall Checklist

- [x] Keep canonical config declarations in `examples/sklearn/configs` and
  `examples/pytorch/configs`.
- [x] Remove hardcoded `ConfigStore.instance().store()` registrations from
  Python declaration modules.
- [x] Register declarations dynamically at package installation from
  `deckard/declarations.py` via `safe_store()`.
- [ ] Ensure that plugin behavior is completely outside of the core modules.
- [ ] Ensure that framework behavior is completely outside of the core modules.
- [ ] Add/verify optional dependency gating for framework-specific registration.
- [ ] Add/verify external config root discovery through `DECKARD_CONFIG_DIRS`.
- [x] Consolidate per-module declarations (`data`, `model`, `attack`, `defense`,
  `plot`, `experiment`) into canonical YAML groups.
- [ ] Refactor tests to compose canonical configs via Hydra (compose-first,
  unit, experiment).
- [ ] Enforce naming conventions (`*Config`, `Default*ScoreConfig`, `*Mixin`,
  `*Plugin`, `modified_snake-case.yaml`).
- [ ] Run coverage + focused refactor test suites and update
  `docs/developers/refactor_plan` progress.

______________________________________________________________________

**Design Specs:** [Config Declaration
Architecture](config_declaration_architecture) | [Naming
Conventions](naming_conventions) | [Mixin and Plugin Rules](mixin_plugin_rules)
