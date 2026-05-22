# Refactor Plan

Use this page as the actionable TODO checklist.
Keep detailed design/spec text in the dedicated developer docs linked below.
## Data Checklist:

### Canon Decisions (Locked)

- [x] Persistence contract is files-only in the target runtime shape.
- [x] Runtime timing storage is canonical-plus-extensible (required canonical
  keys plus optional pipeline/plugin keys).
- [x] DataConfig canonical `score_mode` default is `post-pipeline`.
- [x] `score_mode` selects split scope (`train`/`test`/`val`/`all`) and is
  distinct from hook stage lifecycle (`pre-load`/`pre-sample`/`post-sample`/
  `post-pipeline`).

### Phase 1: Core data runtime (`deckard/data`)
MUST: use deckard/artifacts.py for persistence, with appropriately-typed payloads and metadata.
Status update (2026-05-22): DataPipelineConfig is now a legacy alias to DataConfig; runtime pipeline execution is owned by DataConfig via an optional DataPipeline object.
- [x] Introduce canonical data scoring stage vocabulary and normalization.
- [x] Route `DataConfig` score execution through canonical stage resolution.
- [x] Add stage-aware score hook dispatch with legacy hook compatibility.
- [x] Implement and centralize stage helper API in `deckard/data/stages.py`
  (normalization and hook token helpers consumed by runtime imports).
- [x] Define one explicit runtime attribute contract for all data stages
  (`_X/_y`, train/test/val splits, `times`, `scores`, `files`).
- [x] Replace legacy top-level persistence kwargs with files-only persistence orchestration in runtime call paths.
- [x] Set and enforce canonical DataConfig scoring default to
  `score_mode=post-pipeline` while preserving explicit runtime overrides.
- [x] Split `DataConfig` orchestration responsibilities into explicit loaders,
  samplers, pipeline runners, and score runners.
- [x] Move `DataPluginRuntimeMixin` out of `deckard/data/_mixins.py` into
  `deckard/data/stages.py` and wire core data runtimes to use it.
- [x] Move pipeline runtime logic from `deckard/data/_mixins.py` and
  `deckard/data/base.py` into `deckard/data/pipeline/core.py` with thin
  adapters in core data configs.
- [ ] Move remaining plugin/framework branching logic out of core runtime paths.
- [ ] Replace (non-unit) tests that depend on private internals with public DataConfig
  contract tests (including files/times/stage-vs-mode semantics).

### Phase 2: Framework data runtimes (`deckard/frameworks/**/data.py`)

- [ ] Align framework data configs to the core DataConfig lifecycle method
  contract (`load_dataset`, `sample`, `score`, `__call__`).
- [ ] Unify stage semantics with core canonical stage resolver and keep score
  scope semantics separate from hook stage semantics.
- [ ] Unify hook semantics with core stage-driven hook dispatch.
- [ ] Deduplicate sampler logic between framework data modules and
  framework sampler helpers.
- [ ] Ensure framework modules expose only framework-specific adapters (tensor,
  dataloader, device), not alternate orchestration flows.
- [ ] Align framework persistence to files-only and timing metadata to
  canonical-plus-extensible `times` behavior.

### Phase 3: Plugin data runtimes (`deckard/plugins/**/data.py`)

- [ ] Convert plugin data configs into policy layers on top of canonical runtime
  behavior (not replacement runtimes).
- [ ] Keep only plugin-specific concerns in plugin modules:
  sensitive features, mitigation transforms, plugin scorers, mode validation.
- [ ] Canonicalize plugin hook names to stage-scoped before/after semantics.
- [ ] Keep plugin score behavior split-scoped by `score_mode` and avoid using
  hook stage names as score scope aliases.
- [ ] Remove non-top-level compatibility aliases/import paths for plugin data
  internals.

### Phase 4: Contracts, docs, and migration guards

- [ ] Add cross-family contract tests asserting unified attributes, methods, and
  control-flow (`files`, `times`, `scores`, canonical stage/mode behavior).
- [ ] Add stage and hook conformance tests for core/framework/plugin families.
- [ ] Update API docs (`data`, `pytorch`, `fairlearn`) to reflect canonical
  stage and hook behavior, files-only persistence, and score-mode semantics.
  - [x] `docs/api/data.md` updated for DataConfig pipeline ownership and score-mode semantics.
  - [ ] `docs/api/pytorch.md` and `docs/api/fairlearn.md` pending targeted refresh.
- [ ] Document migration constraints: preserve top-level Config APIs only.
- [ ] Run focused suites + coverage, then mark checklist completion.




## Overall Checklist

- [x] Lock and enforce files-only persistence contract for data runtimes
  (no legacy top-level persistence kwargs in target state).
- [x] Lock and enforce canonical-plus-extensible runtime timing model
  (required canonical keys with optional pipeline/plugin keys).
- [x] Lock and enforce DataConfig canonical score default (`post-pipeline`)
  with explicit separation between score scope (`score_mode`) and hook stage
  lifecycle.
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
