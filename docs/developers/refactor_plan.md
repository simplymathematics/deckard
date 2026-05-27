# Refactor Plan

Use this page as the actionable TODO checklist.
Keep detailed design/spec text in the dedicated developer docs linked below.

## Data Checklist

### Canon Decisions (Locked)

- [x] Persistence contract is files-only in the target runtime shape.

- [x] Runtime timing storage is canonical-plus-extensible (required canonical

  keys plus optional pipeline/plugin keys).

- [x] DataConfig canonical `score_mode` default is `test`.

- [x] `score_mode` selects split scope (`train`/`test`/`val`/`all`) and is

  distinct from hook stage lifecycle (`pre-load`/`pre-sample`/`post-sample`/
  `post-pipeline`).

### Phase 1: Core data runtime (`deckard/data`)

MUST: use deckard/artifacts.py for persistence, with appropriately-typed payloads and metadata.
Status update (2026-05-22): DataConfig is now a legacy alias to DataConfig; runtime pipeline execution is owned by DataConfig via an optional DataPipeline object.

- [x] Introduce canonical data scoring stage vocabulary and normalization.

- [x] Route {class}`deckard.data.DataConfig` score execution through canonical stage resolution.

- [x] Add stage-aware score hook dispatch with legacy hook compatibility.

- [x] Implement and centralize stage helper API in `deckard/data/canon.py`

  (normalization and hook token helpers consumed by runtime imports).

- [x] Define one explicit runtime attribute contract for all data stages

  (`_X/_y`, train/test/val splits, `times`, `scores`, `files`).

- [x] Replace legacy top-level persistence kwargs with files-only persistence orchestration in runtime call paths.

- [x] Set and enforce canonical DataConfig scoring default to

  `score_mode=test` while preserving explicit runtime overrides.

- [x] Split {class}`deckard.data.DataConfig` orchestration responsibilities into explicit loaders,

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

- [x] Convert plugin data configs into policy layers on top of canonical runtime

  behavior (not replacement runtimes).

- [x] Keep only plugin-specific concerns in plugin modules:

  sensitive features, mitigation transforms, plugin scorers, mode validation.

- [x] Canonize plugin hook names to stage-scoped before/after semantics.

- [x] Keep plugin score behavior split-scoped by `score_mode` and avoid using hook stage names as score scope aliases.

- [x] Keep compatibility aliases for plugin data internals and document their supported import paths.

- [x] Add plugin availability detection (optional dependency + import checks) and gate plugin re-exports based on installed packages.

- [x] Ensure that top-level *Config objects keep their current behavior

- [x] Implement ANJANA pipeline policy hook at `pre_sample` (before_sample) and keep anonymization logic policy-only.

- [x] Implement Fairlearn pipeline policy hook at `after_pipeline` (post-pipeline) for stage-aligned runtime behavior.

- [x] Add ANJANA score tail hook (post_pipeline) that runs after base/core scores.

- [x] Add Fairlearn score tail hook (`after_score`) and enforce fairlearn metrics merge last.

- [x] Normalize plugin scoring calls to split-scoped `score_mode` (`train|test|val|all`) and remove stage-name-as-mode behavior.

- [x] Keep top-level config behavior stable (`AnjanaDataConfig`, `FairlearnDataConfig`, `LifelinesExperimentConfig`, etc.) via focused plugin suite validation.

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

## Model Checklist

- [x] Add a canonical model runtime contract module (`deckard/model/canon.py`).

- [x] Align model runtime fields, timing keys, and score-mode normalization to canon helpers.

- [x] Add focused model canon contract tests for constants and runtime initialization.

- [x] Validate top-level model config behavior remains stable ({class}`deckard.model.ModelConfig` + family aliases).

- [x] Run focused model + phase-4 contract suites and update checklist status.

### Model Phase 1: Trainer Runtime Canon

- [x] Add configurable trainer runtime objects modeled after `BaseSampler` (resolve/compose/execute contract).

- [x] Implement trainer variants: base sklearn, pretrained, partial-fit, partial-fit+pruning, pruning, and base pytorch.

- [x] Route {class}`deckard.model.ModelConfig` training/load flow through trainer composition without replacing core orchestration.

- [x] Reuse existing utility/artifact primitives from `deckard/utils.py` and `deckard/artifacts.py` (no duplicate IO/config logic).

### Model Phase 2: Thin Wrapper Family Rule

- [x] Apply thin-wrapper policy to framework/plugin `*ModelConfig` families (plugin/framework specific behavior only).

### Model Phase 3: Defense Stage Semantics

- [x] Treat ANJANA data pipeline defenses as pre-ART model defense stage in model-defense orchestration.

- [x] Normalize fairlearn defense stages: `fairlearn.reductions` as pre-fit and `fairlearn.adversarial` as post-fit.

- [x] If a pretrained model receives a defense step with `apply_fit=True`, snapshot the pre-defense state and retrain before applying that defense.

- [x] Persist the pre-defense score/timing/prediction snapshot under an explicit key such as `pre-defense` or `pre-<alias>-defense` before rerunning.

- [x] Add focused model/defense stage tests validating trainer and defense-stage canon behavior.

- [x] Ensure a single defense chain with `apply_fit=True` does not trigger repeated training passes under normal runtime flow.

- [x] Restrict forced retraining-on-fit-defense to loaded pretrained-model flows (no extra retrain for standard already-fitted model paths).

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

## Attack Checklist

- [x] Define a canonical attack runtime contract (`files`, `times`, `scores`, `stage`, `mode`).

- [x] Normalize attack-stage hooks and defense/attack ordering semantics across core and framework adapters.

- [x] Keep attack configuration resolution thin so backend-specific attack logic stays in wrappers.

- [x] Add focused attack contract tests for persistence, timing, and split-scoped scoring behavior.

- [x] Update attack docs and examples to reflect canonical attack/runtime behavior.

## Detector Checklist

- [x] Define a canonical detector runtime contract and align detector state fields to shared model/runtime helpers.

- [x] Normalize detector-stage hook dispatch and defense application ordering.

- [x] Ensure detector configs preserve files-only persistence and canonical timing metadata.

- [x] Add detector contract tests covering train/load behavior, wrapper reuse, and score-state preservation.

- [x] Update detector docs and example configs to match the final canon.

- [x] Support config-driven detector runtime execution (coerce/execute `*Config` objects) before detector fit/detect orchestration.

- [x] Add detector filtering paths for poisoning and evasion with canonical side effects: poison filtering retrains model, evasion filtering post-processes attack predictions.

- [x] Emit `poison_filter_success` and `evasion_filter_success` scores and ensure successful filtering yields unperturbed inputs/labels for downstream scoring consistency.

## Scorer Checklist

- [x] Define a canonical scorer runtime contract for `score_mode`, stage hooks, and score aggregation.

- [x] Normalize scorer output shapes so score dictionaries remain flat, serializable, and merge-safe.

- [x] Keep scorer policy logic split-scoped and separate from hook-stage lifecycle names.

- [x] Add scorer contract tests for stage dispatch, score merging, and persistence of score artifacts.

- [x] Update scorer docs and examples to describe the canonical scoring path.

### Scorer Serialization Contract (Pending)

- [x] create a native ScoreDict class to make this behavior consistent everywhere and unify all score-transformation functions in that object as helper methods. Use this object to type score_dict in BaseConfig (score_dict: ScoreDict).

- [x] Add a callable lifecycle method that replaces the current read_or_initialize_scores and general persistence flows so each ScoreDict call loads/saves from disk when a score file is specified; otherwise it returns the nested runtime score dictionary.

- [x] ensure no other data scoring paths still rely on legacy-only y_true/y_pred paths

- [x] Add vector-valued scoring support in runtime and persistence layers.

- [x] Ensure score persistence uses human-readable and easy-to-parse formats (formatted JSON/YAML first).

- [x] Implement explicit score de-serialization and serialization contract for scalar, vector, and nested score payloads.

- [x] Guarantee final persisted score output includes:

  - [x] a flat dictionary keyed by runtime scope (mode/stage dependent)
  - [x] a dot.list OmegaConf-style dictionary for easy downstream parsing of all scores across stages, modes, and splits.
  - [x] Create a docs/notebooks/scoring.ipynb that explains major paths.
  - [x] Update docs/developers/score with a design spec
  - [x] Update overview/scoring with the new contract

## Optuna DB Checklist

- [x] Define canonical Optuna study dataframe loading helpers and use them as shared runtime APIs.

- [x] Route DataConfig dataset loading to support optuna-backed sources (`optuna`, `.db`, `.sqlite3`, or explicit `optuna_storage`).

- [x] Treat Seaborn plot configs as DataConfig extensions by accepting/resolving DataConfig runtime payloads directly.

- [x] Treat Yellowbrick plot configs as ExperimentConfig extensions and keep experiment-only preparation logic in Yellowbrick modules.

- [x] Add focused tests for Optuna-backed DataConfig loading and Seaborn plotting from Optuna/DataConfig sources.

- [x] Update plotting and data docs/examples to describe canonical `optuna.db` query paths.

## Plot Checklist

- [x] Define a canonical plot runtime contract for `files`, `times`, plot output state, and backend selection.

- [x] Normalize plot backend dispatch so Seaborn, Yellowbrick, and survival plotting remain thin wrappers over the shared plot API.

- [x] Keep experiment-preparation and output hydration policy-only in backend-specific modules.

- [x] Add plot contract tests for lazy setup, files-only persistence, and one-time experiment preparation.

- [x] Update plot docs and example configs to reflect canonical plot behavior.

## File Checklist

- [x] Decentralize the file-schema `TypedDict`s into module-local canon definitions instead of centralizing them in one registry file (These currently exist in *canon.py files).

- [x] Introduce a shared abstract file handler that can operate on the canon `TypedDict`s for disk-status checks, parsing, string replacement, and validation.

- [x] Keep {class}`deckard.file.FileConfig` as the public typed file registry while removing legacy group-specific path assumptions from runtime call sites.

- [x] Align file placeholders (`{num}`, `{timestamp}`, `{hash}`, replacements) with the final runtime identity and multirun behavior.

- [x]Ensure that {num}/{#} and {hash}/{*} work with run, multirun, and without hydra (UUID fallback).

- [x] Update file-related tests to validate allowed keys, placeholder resolution, handler behavior, and cross-module persistence aliases.

- [x] Refresh file docs/examples so top-level config APIs use the final decentralized file-schema contract.

## Utils Checklist

- [x] Ensure `deckard/artifacts.py` owns persistence load/save behavior for all runtime artifact payloads.

- [x] Move/retain config coercion and normalization in {class}`deckard.utils.BaseConfig` and shared utility helpers only.

- [x] Centralize class resolution, plugin instantiation, score merging, and device resolution in `deckard/utils.py`.

- [x] Remove duplicate coercion/persistence helpers from core, framework, and plugin modules.

- [x] Add utility contract tests for artifact IO, config coercion, plugin spec normalization, and resolver behavior.

### Experiment Runtime Composition Plan (Detailed)

#### Phase 1: Canon Runtime Contract

- [x] Add `deckard/experiment/canon.py` with canonical experiment runtime fields and helper APIs.

- [x] Canonize experiment state buckets:

  - [x] `files` (artifact paths, cache keys, persistence aliases)
  - [x] `times` (canonical timing keys plus extensible stage timings)
  - [x] `scores` (mode/stage-aware score payloads)
  - [x] `outputs` (cached intermediate runtime payloads)
  - [x] `params` (resolved config + runtime kwargs manifest)

- [x] Add mode/stage normalization helpers for experiment-level orchestration (single run + multi-trial semantics).

#### Phase 2: Native Config + HookPlugin + Bundle Composition

- [x] Define native `*Config` composition entry points for data/model/attack/detector/scorer/pipeline/defense.

- [x] Support runtime composition from supplied kwargs without replacing canonical config behavior.

- [x] Add a HookPlugin execution graph generated programmatically from canonical stage definitions in Data/Model/Attack/Score/Detector `*Config` runtimes (no hard-coded stage list).

- [x] Introduce Bundle definitions that group stage hooks + component configs into reusable runtime policies.

- [x] Ensure Bundle composition is additive/overridable and deterministic by explicit order.

#### Phase 3: Caching and Reuse Across Stages/Trials

- [x] Define cache keys for sample/pipeline/train/defense/attack outputs using resolved params + stage identity.

- [x] Persist and reload intermediate outputs for stage skipping and rerun acceleration.

- [x] Ensure cached outputs can be selectively invalidated by component/stage-level parameter changes.

- [x] Ensure all training, defense, pipeline, fold, and attack score artifacts are cache-aware and rehydratable.

#### Phase 4: YAML Serialization Contract

- [x] Add YAML serialization for experiment persistence including:

  - [x] resolved params
  - [x] runtime attributes
  - [x] cached output metadata and pointers

- [x] Keep JSON/YAML score artifacts human-readable and parse-friendly.

- [x] Define explicit de/serialization schema versioning for forward-compatible restores.

- [x] Add load-time migration guards for older score/cache payload structures.

#### Phase 5: DVC Pipeline Autogeneration

Design spec: [DVC Pipeline Autogeneration Spec](dvc)

- [x] Add utility to generate `dvc.yaml` from experiment persistence values and runtime stage graph.

- [x] Map canonical experiment stages to DVC stages with deps/outs/params wiring.

- [x] Emit reproducible stage commands for single experiment execution and multi-trial sweeps.

- [x] Support optional cached-output reuse by pointing DVC outs to canonical runtime file aliases.

- [x] Enable Vega-Lite plot spec outputs (`*.vl.json`) for browser-renderable DVC plot artifacts (yellowbrick and seaborn plots should be supported, but not required).

- [x] Create specs according to [DVC Pipeline Autogeneration Spec](dvc) and deckard-native functionality. Create runnable Hydra YAML files for each plot with names like `attack_alias_vs_metric` or [`roc_auc`](https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html).

- [x] Ensure generated DVCLive/DVC output directories use runtime identity.

- [x] Keep DVC metrics policy canonical.

- [x] targeted tests

- [x] create high-level docs/overview/dvc file

- [x] create demonstration notebook using examples/sklearn context in docs/notebooks/dvc.ipynb

- [x] update docs/developers/dvc

Concrete rewrite steps to align runtime implementation with the design spec:

- [x] Introduce `DVCPluginBundle` as first/last hook bundles for {class}`deckard.experiment.ExperimentConfig` orchestration.

- [x] Instantiate `dvclive.Live` at runtime hook entry (`before_load` in first-position wrapper) and keep a single runtime session per experiment call.

- [x] Pull DVC dependencies in the first wrapper using DVC-native commands (`dvc pull`) before stage execution proceeds.

- [x] Log DVCLive params/metrics/artifacts with native methods (`log_params`, `log_metric`, `log_artifact`, `next_step`, `end`) from runtime hook callbacks.

- [x] Push newly created outputs in the final wrapper (`after_persist`, last-position) via DVC-native commands (`dvc add`, `dvc push`).

- [x] Keep report generation optional and mode-aware (`make_summary`, `make_report`, `make_dvcyaml`, report extension mapping).

- [x] Align canonical DVC stage naming for score/persist stages with `experiment__*` naming.

- [x] Align command emission with Command Templates (`deckard optimize ... stage=... [--multirun] params_file=... dvc_file=...`).

- [x] Enforce Vega-Lite output contract to hydra-resolvable JSON artifacts (`*.vl.json`) even when callers pass YAML-like paths.

- [x] Update and extend targeted tests for bundle construction, lifecycle hook execution, stage naming, and Vega-Lite output normalization.

- [x] Make pruning status contingent on optuna pruner configuration

- [x] Auto-enable dvclive if dvc_plugin is in the default.yaml

- [x] Add native support for fetching/loading artifacts using Live().log_image, Live().log_metric, Live().log_params, Live().log_artifact

- [x] Ensure that run and multi-run both generate reproducible `dvc.yaml` and `params.yaml` files.

Goal:

- [x] generated dvc `cmd` should use `deckard optimize` syntax.

- [x] Integrate experiment/power.py

- [x] Update developers/dvc.md to reflect canon and change the narrative from a plan to a finalized design spec.

- [x] Update dvc.ipynb to reflect the new canon

- [x] Generate dvc.yaml and params.yaml for several experiments (run AND multirun) that demonstrate several Vega-Lite graphs.

#### Phase 6: Hydra Single-Default Multi-Stage Execution

Design specs: [Optimization Runtime Contract](optimization) | [Hydra and Optuna Orchestration Contract](hydra) | [Pruning Runtime Contract](pruning)

- [x] Rename/default callback contract to `DefaultOptimizerCallback` as the configurable Hydra callback adapter.

- [x] Define {class}`deckard.layers.optimize.OptimizerConfig` as the dedicated runtime optimization policy object

  (metadata, optimizers, directions, trial reporting/pruning policy, optional DVCLive integration).

- [x] Keep callback behavior adapter-thin:

  - [x] callback owns lifecycle hooks
  - [x] callback delegates policy behavior to {class}`deckard.layers.optimize.OptimizerConfig`

- [x] Define one Hydra default profile that can execute:

  - [x] a single experiment through selected hook stages
  - [x] multiple trials with cached intermediate reuse

- [x] Ensure stage selection and trial fan-out are controlled by runtime kwargs/overrides, not alternate orchestration paths.

- [x] Preserve files-only persistence and stage-aware score collection for both single and multi-trial flows.

- [x] targeted tests

- [x] create/update high-level docs/overview/hydra file

- [x] create/update demonstration notebook using examples/sklearn context in docs/notebooks/hydra.ipynb

- [x] update docs/developers/hydra

- [x] rename any existing *optuna docs to *optimize

- [x] create/update high-level docs/overview/optimize file

- [x] create/update demonstration notebook using examples/sklearn context in docs/notebooks/optimize.ipynb

- [x] update docs/developers/optimize

#### Phase 7: Validation and Documentation

- [x] Add contract tests for HookPlugin stage ordering, Bundle merge behavior, and native `*Config` composition.

- [x] Add integration tests for cache reuse, YAML round-trip restores, and DVC autogeneration correctness.

- [x] Add Hydra compose tests validating single-default stage selection and multi-trial execution behavior.

- [x] Add optimization callback/config tests for adapter + policy split:

  - [x] `DefaultOptimizerCallback` lifecycle delegation
  - [x] {class}`deckard.layers.optimize.OptimizerConfig` trial resolution/report/prune behavior

- [x] Add pruning integration tests that assert prune termination raises `TrialPruned`.

- [x] Add DVC contract tests that assert:

  - [x] Vega-Lite plot path generation (`*.vl.json`)
  - [x] identity-derived output directories in run and multirun
  - [x] metrics file policy and optimizer-keyed selector behavior
  - [x] summary generation
  - [x] report generation (.html, .ipynb, and .md)

- [x] Document experiment canon, bundle authoring, hook contracts, serialization schema, and DVC workflow in developer docs.

#### Execution Plan: Docs -> Notebooks -> Testing -> Coverage

##### Step 1: Overview and API Documentation

- [x] Add overview docs for each core module modeled after `docs/overview/scoring.md`:

  - [x] `docs/overview/data.md`
  - [x] `docs/overview/model.md`
  - [x] `docs/overview/attack.md`
  - [x] `docs/overview/detector.md`
  - [x] `docs/overview/experiment.md`
  - [x] `docs/overview/file.md`
  - [x] `docs/overview/plot.md`
  - [x] `docs/overview/utils.md`
  - [x] `docs/overview/scoring.md`

- [x] Ensure each overview page includes: defaults, runtime contract, stage/mode semantics (when applicable), persistence behavior, examples, and quick checklist.

- [x] Cross-link every overview page with matching API and developer canon docs.

- [x] Update overview index/toctree reading order to include all core-module pages.

##### Step 2: Notebook Demonstration Suite

- [x] Add a DVC-focused notebook under `docs/notebooks/` for persistence and cache behavior verification (single-run and multi-trial).

- [ ] Ensure that existing notebooks demonstrate:

  - [ ] files-only persistence aliases
  - [ ] canonical timing keys plus extensibility
  - [ ] stage/mode normalization and hook ordering
  - [ ] cache-key determinism and selective invalidation
  - [ ] YAML/JSON human-readable score and params artifacts
  - [ ] stage-based fingerprinting
  - [ ] mode based scoring
  - [ ] data pipeline, model defenses for anjana/fairlearn

- [ ] Add notebook index page entries and per-notebook run expectations.

##### Step 3: Test Plan Aligned to Notebook Scenarios

- [ ] Convert notebook scenarios into focused instruction aboout various workflow paths.

- [ ] Add/extend experiment tests for stage graph generation, bundle merge order, cache reuse, and cache invalidation.

- [ ] Add end-to-end tests that compare fresh runs vs cache-hit reruns and assert equivalent outputs.

- [ ] Add persistence contract tests for scalar/vector/nested score serialization and de-serialization.

- [ ] Add regression tests that validate dot-list/OmegaConf-style score flattening output.

- [ ] Add docs-build tests to ensure new overview and notebook references resolve.

- [ ] Build docs

##### Step 4: Coverage Closure and Exit Criteria

- [ ] Run focused test suites for all touched core modules and notebook-derived scenarios.

- [ ] Run coverage and identify remaining gaps for new runtime branches (hooks, cache paths, persistence codecs).

- [ ] Add targeted tests to close the remaining branch/line gaps.

- [ ] Capture final coverage deltas and mark checklist items complete.

- [ ] Publish final migration summary in developer docs with links to overview docs, notebooks, and tests.

## Final framework/plugin migration

- [ ] Implement, test, and document persistence workflows with/without attacks/defenses and with and without pre-trained models for sklearn and pytorch frameworks in a new examples/*/config/pretrained-default.yaml and test them.

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

  `deckard/declarations.py` via {func}`deckard.utils.safe_store`.

- [ ] Ensure that plugin behavior is completely outside of the core modules.

- [ ] Ensure that framework behavior is completely outside of the core modules.

- [ ] Add/verify optional dependency gating for framework-specific registration.

- [ ] Add/verify external config root discovery through `DECKARD_CONFIG_DIRS`.

- [x] Consolidate per-module declarations (`data`, `model`, `attack`, `defense`,

  `plot`, `experiment`) into canonical YAML groups.

- [ ] Refactor tests to compose canonical configs via Hydra (compose-first,

  unit, experiment).

- [ ] Enforce naming conventions (`*Config`, `Default*ScorerConfig`, `*Mixin`,

  `*Plugin`, `modified_snake-case.yaml`).

- [ ] Run coverage + focused refactor test suites and update

  `docs/developers/refactor_plan` progress.

