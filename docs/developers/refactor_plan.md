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
Status update (2026-05-22): DataConfig is now a legacy alias to DataConfig; runtime pipeline execution is owned by DataConfig via an optional DataPipeline object.
- [x] Introduce canonical data scoring stage vocabulary and normalization.
- [x] Route `DataConfig` score execution through canonical stage resolution.
- [x] Add stage-aware score hook dispatch with legacy hook compatibility.
- [x] Implement and centralize stage helper API in `deckard/data/canon.py`
  (normalization and hook token helpers consumed by runtime imports).
- [x] Define one explicit runtime attribute contract for all data stages
  (`_X/_y`, train/test/val splits, `times`, `scores`, `files`).
- [x] Replace legacy top-level persistence kwargs with files-only persistence orchestration in runtime call paths.
- [x] Set and enforce canonical DataConfig scoring default to
  `score_mode=post-pipeline` while preserving explicit runtime overrides.
- [x] Split `DataConfig` orchestration responsibilities into explicit loaders,
  samplers, pipeline runners, and score runners.
- [x] Move `DataPluginRuntimeMixin` out of `deckard/data/_mixins.py` into
  `deckard/data/canon.py` and wire core data runtimes to use it.
- [x] Move pipeline runtime logic from `deckard/data/_mixins.py` and
  `deckard/data/base.py` into `deckard/data/pipeline/base.py` with thin
  adapters in core data configs.
- [x] Rename `deckard/data/pipeline/core.py` to `deckard/data/pipeline/base.py` and update all references (including tests and docs)
- [x] Rename `deckard/data/stages.py` to `deckard/data/canon.py` and update all references (including tests and docs)
- [x] Move ScoreOrchestratorMixin to a centralized location (into existing deckard/plugins/base.py) and rename it PluginOrchestratorMixin.
- [x] Move remaining plugin/framework branching logic out of core runtime paths (`deckard/data/*`).

### Phase 2: Framework data runtimes (`deckard/frameworks/**/data.py`)

- [x] Align framework data configs to the core DataConfig lifecycle method
  contract (`load_dataset`, `sample`, `score`, `__call__` `fit`).
- [x] Unify stage semantics with core canonical stage resolver and keep score
  scope semantics separate from hook stage semantics.
- [x] Unify hook semantics with core stage-driven hook dispatch.
- [x] Deduplicate sampler logic between framework data modules and
  framework sampler helpers.
- [x] Ensure framework modules expose only framework-specific adapters (tensor,
  dataloader, device), not alternate orchestration flows.
- [x] Align framework persistence to files-only and timing metadata to
  canonical-plus-extensible `times` behavior.

### Phase 3: Plugin data runtimes (`deckard/plugins/**/data.py`)

- [ ] Convert plugin data configs into policy layers on top of canonical runtime
  behavior (not replacement runtimes).
- [ ] Keep only plugin-specific concerns in plugin modules:
  sensitive features, mitigation transforms, plugin scorers, mode validation.
- [x] Canonize plugin hook names to stage-scoped before/after semantics.
- [x] Keep plugin score behavior split-scoped by `score_mode` and avoid using hook stage names as score scope aliases.
- [ ] Keep compatibility aliases for plugin data internals and document their supported import paths.
- [ ] Add plugin availability detection (optional dependency + import checks) and gate plugin re-exports based on installed packages.
- [x] Ensure that top-level *Config objects keep their current behavior
- [x] Implement ANJANA pipeline policy hook at `pre_sample` (before_sample) and keep anonymization logic policy-only.
- [x] Implement Fairlearn pipeline policy hook at `after_pipeline` (post-pipeline) for stage-aligned runtime behavior.
- 
- [x] Add ANJANA score tail hook (post_pipeline) that runs after base/core scores.
- [x] Add Fairlearn score tail hook (`after_score`) and enforce fairlearn metrics merge last.
-
- [x] Normalize plugin scoring calls to split-scoped `score_mode` (`train|test|val|all`) and remove stage-name-as-mode behavior.
- [ ] Keep top-level config behavior stable (`AnjanaDataConfig`, `FairlearnDataConfig`, `LifelinesDataConfig`) via focused plugin suite validation.
- [x] test fairlearn.preprocessing.CorrelationRemover (check first for existing test and extend rather than make a new one)
- [x] register fairlearn.preprocessing.PrototypeRepresentationLearner (max_iter =1 for tests) in examples/*/config/data/pipeline. Test.
- [x] validate anjana scoring/defense 

### Contracts, docs, and migration guards

- [x] Add new design docs in docs/developers about canon, plugins, etc.
- [x] Add cross-family contract tests asserting unified attributes, methods, and
  control-flow (`files`, `times`, `scores`, canonical stage/mode behavior).
- [x] Add stage and hook conformance tests for core/framework/plugin families.
- [x] Update API docs (`data`, `pytorch`, `fairlearn`, `anjana` `pipeline`) to reflect canon.
- [x] Update developer docs to explain the canon file.
- [x] Add new doc files to indices. Add cross-links elsewhere.
- [x] Add hook-based persistence using existing 
  stage and hook behavior, files-only persistence, and score-mode semantics.
  - [x] `docs/api/data.md` updated for DataConfig pipeline ownership and score-mode semantics.
  - [x] `docs/api/pytorch.md` and `docs/api/fairlearn.md` and `docs/api/anjana.md` pending targeted refresh.
- [x] Document migration constraints: preserve top-level Config APIs only.
- [x] Ensure that all examples/*/config/data files are migrated
- [ ] Run  and fix coverage, then mark checklist completion.

## Model Checklist:
- [x] Add a canonical model runtime contract module (`deckard/model/canon.py`).
- [x] Align model runtime fields, timing keys, and score-mode normalization to canon helpers.
- [x] Add focused model canon contract tests for constants and runtime initialization.
- [x] Validate top-level model config behavior remains stable (`ModelConfig` + family aliases).
- [x] Run focused model + phase-4 contract suites and update checklist status.

### Model Phase 1: Trainer Runtime Canon

- [x] Add configurable trainer runtime objects modeled after `BaseSampler` (resolve/compose/execute contract).
- [x] Implement trainer variants: base sklearn, pretrained, partial-fit, partial-fit+pruning, pruning, and base pytorch.
- [x] Route `ModelConfig` training/load flow through trainer composition without replacing core orchestration.
- [x] Reuse existing utility/artifact primitives from `deckard/utils.py` and `deckard/artifacts.py` (no duplicate IO/config logic).

### Model Phase 2: Thin Wrapper Family Rule

- [ ] Apply thin-wrapper policy to framework/plugin `*ModelConfig` families (plugin/framework specific behavior only).
- [ ] Centralize shared model canon/runtime helper logic in one import path and consume from wrappers.

### Model Phase 3: Defense Stage Semantics

- [x] Treat ANJANA data pipeline defenses as pre-ART model defense stage in model-defense orchestration.
- [x] Normalize fairlearn defense stages: `fairlearn.reductions` as pre-fit and `fairlearn.adversarial` as post-fit.
- [x] If a pretrained model receives a defense step with `apply_fit=True`, snapshot the pre-defense state and retrain before applying that defense.
- [x] Persist the pre-defense score/timing/prediction snapshot under an explicit key such as `pre-defense` or `pre-<alias>-defense` before rerunning.
- [x] Add focused model/defense stage tests validating trainer and defense-stage canon behavior.

### Model Documentation and Finalization

- [x] Add a new design doc in docs/developers about model canon, defenses, trainers.
- [x] Add a new overview doc in docs/overview about model canon, defenses, trainers using docs/overview/scoring.md as a guide.
- [x] Add cross-family contract tests asserting unified attributes, methods, and
  control-flow (`files`, `times`, `scores`, canonical stage/mode behavior).
- [x] Add stage and hook conformance tests for core/framework/plugin families.
- [x] Update API docs (`model`, `pytorch`, `fairlearn`, `anjana` `pipeline`) to reflect canon.
- [x] Add new doc files to indices. Add cross-links elsewhere.
- [x] Add hook-based persistence using existing 
  stage and hook behavior, files-only persistence, and score-mode semantics.
  - [x] `docs/api/data.md` updated for DataConfig pipeline ownership and score-mode semantics.
  - [x] `docs/api/pytorch.md` and `docs/api/fairlearn.md` and `docs/api/anjana.md` targeted refresh completed.
- [x] Document migration constraints: preserve top-level Config APIs only.
- [x] Ensure that all examples/*/config/data files are migrated
- [x] Run  and fix coverage, then mark checklist completion

## Attack Checklist:
- [ ] Define a canonical attack runtime contract (`files`, `times`, `scores`, `stage`, `mode`).
- [ ] Normalize attack-stage hooks and defense/attack ordering semantics across core and framework adapters.
- [ ] Keep attack configuration resolution thin so backend-specific attack logic stays in wrappers.
- [ ] Add focused attack contract tests for persistence, timing, and split-scoped scoring behavior.
- [ ] Update attack docs and examples to reflect canonical attack/runtime behavior.

## Detector Checklist:
- [ ] Define a canonical detector runtime contract and align detector state fields to shared model/runtime helpers.
- [ ] Normalize detector-stage hook dispatch and defense application ordering.
- [ ] Ensure detector configs preserve files-only persistence and canonical timing metadata.
- [ ] Add detector contract tests covering train/load behavior, wrapper reuse, and score-state preservation.
- [ ] Update detector docs and example configs to match the final canon.

## Scorer Checklist:
- [ ] Define a canonical scorer runtime contract for `score_mode`, stage hooks, and score aggregation.
- [ ] Normalize scorer output shapes so score dictionaries remain flat, serializable, and merge-safe.
- [ ] Keep scorer policy logic split-scoped and separate from hook-stage lifecycle names.
- [ ] Add scorer contract tests for stage dispatch, score merging, and persistence of score artifacts.
- [ ] Update scorer docs and examples to describe the canonical scoring path.

## Experiment Checklist
- [ ] Define a canonical experiment runtime contract for `files`, `times`, `scores`, and component orchestration.
- [ ] Normalize experiment loading so data/model/attack/detector/scorer configs are composed through the canonical runtime.
- [ ] Ensure experiment persistence and score collection remain files-only and stage-aware.
- [ ] Add experiment contract tests for end-to-end orchestration, cross-family composition, and rerun stability.
- [ ] Update experiment docs and example workflows to match the final canonical flow.

## Plot Checklist
- [ ] Define a canonical plot runtime contract for `files`, `times`, plot output state, and backend selection.
- [ ] Normalize plot backend dispatch so Seaborn, Yellowbrick, and survival plotting remain thin wrappers over the shared plot API.
- [ ] Keep experiment-preparation and output hydration policy-only in backend-specific modules.
- [ ] Add plot contract tests for lazy setup, files-only persistence, and one-time experiment preparation.
- [ ] Update plot docs and example configs to reflect canonical plot behavior.

## File Checklist
- [ ] Decentralize the file-schema `TypedDict`s into module-local canon definitions instead of centralizing them in one registry file (These currently exist in *canon.py files).
- [ ] Introduce a shared abstract file handler that can operate on the canon `TypedDict`s for disk-status checks, parsing, string replacement, and validation.
- [ ] Keep `FileConfig` as the public typed file registry while removing legacy group-specific path assumptions from runtime call sites.
- [ ] Align file placeholders (`{num}`, `{timestamp}`, `{hash}`, replacements) with the final runtime identity and multirun behavior.
- [ ]Ensure that {num}/{#} and {hash}/{*} work with run, multirun, and without hydra (UUID fallback).
- [ ] Update file-related tests to validate allowed keys, placeholder resolution, handler behavior, and cross-module persistence aliases.
- [ ] Refresh file docs/examples so top-level config APIs use the final decentralized file-schema contract.


## Utils Checklist
- [ ] Ensure `deckard/artifacts.py` owns persistence load/save behavior for all runtime artifact payloads.
- [ ] Move/retain config coercion and normalization in `ConfigBase` and shared utility helpers only.
- [ ] Centralize class resolution, plugin instantiation, score merging, and device resolution in `deckard/utils.py`.
- [ ] Remove duplicate coercion/persistence helpers from core, framework, and plugin modules.
- [ ] Add utility contract tests for artifact IO, config coercion, plugin spec normalization, and resolver behavior.


## Final framework/plugin migration:
- [ ] Keep all *fair* behavior in lightweight Fairlearn mixins, plugins, and wrappers outside the core modules.
- [ ] Keep all *torch* behavior in lightweight PyTorch wrappers outside the core modules.
- [ ] Keep all *anjana* behavior in lightweight privacy mixins, plugins, and wrappers outside the core modules.
- [ ] Keep compatibility aliases and add explicit deprecation policy only where behavior must change.
- [ ] Detect available framework/plugin packages at import/runtime and conditionally register or re-export convenience objects from core modules.
- [ ] Add tests for re-export gating: installed plugins are re-exported, missing optional dependencies fail gracefully without breaking core imports.
- [ ] Document the compatibility alias and re-export matrix in developer/API docs.
- [ ] Verify framework/plugin modules only own backend-specific device, estimator, and policy logic.
- [ ] Add final migration tests that assert core modules import cleanly without framework/plugin fallback behavior.


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
