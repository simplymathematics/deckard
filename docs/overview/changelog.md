# Changelog

## .98.3

- Documentation hardening, notebook refresh/lockfile sync, runtime/config consistency fixes, and CI/test stabilization across core workflows.
- Scope highlights: improved API/extension coverage (including Fairlearn, Lifelines, Yellowbrick, and Anjana), refined sklearn/lifelines fixtures, and targeted cleanup in DVC experiment lint/test paths.
- Phase 3 closure TODOs:
  - [x] Add anjana matrix coverage.
  - [x] Remove fairlearn compatibility aliases.
  - [x] Add .deckard_rc env loading.
  - [x] Add targeted attack metric labels.
  - [x] Hash runtime artifact file names.
  - [x] Sanitize params export configs.
  - [x] Reduce file.py to canon types.
  - [x] Expand pytorch data loader tests.
  - [x] Run fail-fast targeted tests.
  - [x] Update Phase 3 checklist statuses.
## .98.2

- Finalized repository enforcement pass for core scope checks and verified no baseline violations in `deckard/` via `scripts/repository_enforcement.py --scope deckard/`.
- Updated core runtime architecture so base config objects expose public methods, a documented execution order, and separate plugin/scoring hooks.
- Standardized canonical contracts in `canon.py` files to define inputs, outputs, and execution order for base config objects.
- Clarified orchestration responsibilities: `orchestration.py` applies scoring hooks in canonical method order during `Config.__call__`.
- Clarified artifact responsibilities: `artifacts.py` manages persistence and pipeline dependency chains.
- Consolidated base runtime behaviors in {class}`deckard.utils.BaseConfig` for coercion, fingerprinting, and runtime instantiation.
- Promoted the data sampler to a canonical runtime object.
- Expanded {class}`deckard.model.ModelConfig` with configurable `defense` and `trainer` composition for defense chains, pruning, and pre-trained model analysis.
- Completed documentation for {class}`deckard.attack.AttackConfig` and updated {class}`deckard.detector.DetectorConfig` to support training and pre-trained filtering modes.
- Extended {class}`deckard.layers.optimize.OptimizerConfig` for top-level `optuna` configuration and documented {class}`deckard.layers.optimize.DefaultOptimizerCallback` for user-configurable optimization flows.
- Added canonical {class}`deckard.artifacts.ScoreDict` helpers for parsing, viewing, storing, and updating runtime score dictionaries.
- Reworked [fairlearn](extensions/index) and [anjana](extensions/index) integration to use the plugin architecture.
- Drafted {class}`deckard.experiment.dvc.DVCExperimentConfig` for reproducible experiment generation, `dvclive` monitoring integration, training-flow updates, and `Vega-lite` plotting support (WIP).

## .98.1

- Core package updates in `deckard/` (entrypoints, config/declaration handling,
data/experiment utilities, and fairlearn scoring).
- Documentation updates in `docs/` (Sphinx config and multiple notebook
  refreshes, including Hydra/Optuna/Lifelines/Artifacts flows).
- Build artifact refresh in `build/` and notebook pipeline state updates
  (`docs/notebooks/dvc.lock`, `docs/notebooks/dvc.yaml`).
- isolated sklearn/pytorch from code model code
- isolated plugins from core model code
- unified scoring interface
- improved run-time of test-suite and documentation build

## Known TODOs

### Phase 1: Quick Wins (Docs + Small Test/Notebook Fixes)

- [x] `docs/overview/extensions/anjana.md`: replace placeholder `TODO` section with concrete integration guidance.
- [x] `docs/notebooks/pytorch.ipynb`: resolve `TODO: Fix torch default data scoring` (`scorer=None` placeholders in notebook examples) in the frameworks/pytorch/score.py file.
- [x] `docs/notebooks/yellowbrick.ipynb`: resolve `TODO: Add default cluster-scoring` (`scorer=None` placeholder in notebook examples) in a new deckard/score/cluster.py file.
- [x] `test/test_plot/test_init.py`: repair broken plot init test.
- [x] `papers/compression_distance/find_best.py`: add optimization direction argument support. Ensure to include `diff` directions for inclusion in the paretoset according to paretoset python api (`diff` is not compatible with optuna). There should be no `diff` directions set by default.
- [x] `deckard/__main__.py`: add comprehensive CLI help strings for the main and in layers/ main loop.
- [x] fix or remove xpassed/xfailed tests

### Phase 2: Documentation Coverage + Repo Settings

- [x] Audit individual files for missing cross-references (e.g., [AttackConfig](../api/modules) or [fairlearn](extensions/index) references should link to docs), create a way to track this per-file, and add it to repository enforcement so that we can systematically eliminate the lack of cross-references. Also audit for broken `:mod` and `:doc` syntax inside all docs/ *.md files and all docs/notebooks/*.ipynb files
- [x] Ensure that all docs/ files cross-reference instead of merely wrap objects with ``.
- [x] Fix broken :mod syntax
- [x] Fix broken :doc syntax
- [x] Ensure that notebook .md cells properly cross-link to the docs.
- [x] Add dependency licenses and hyperlinks across all top-level docs and relevant index pages.
- [x] Add links to plugins and their licenses.
- [x] Create a new LICENSES file for this content and move the existing LICENSES section from docs/index.md there and link to it from the index.
- [x] Document `matplotlibrc` behavior and provide extension examples.
- [x] Update [lifelines](extensions/index), `art_attacks`, and `art_defenses` notebooks for clarity and scope.
- [x] `docs/developers/workflows.md`: document Codecov (or equivalent) repository settings checklist.
- [x] `docs/developers/workflows.md`: wire security scanning workflow/checklist updates in docs.
- [x] `docs/developers/workflows.md`: document security scanning integration with repository enforcement and CI gates.
- [x] `docs/developers/workflows.md`: document Dependabot repository settings and update flow.
- [x] audit index files and toctrees to mirror the source code
- [x] Ensure that the flow between base module mirrors the experiment flow
- [ ] Ensure that the flow from base -> frameworks is obvious and parallel
- [ ] Ensure that the flow from base -> plugins is obvious and parallel

### Phase 3: Targeted Code + Workflow Updates

- [x] `.github/workflows/test-optional-dependencies.yml`: add [anjana](extensions/index) optional dependency test coverage in the existing matrix job.
- [x] `deckard/plugins/fairlearn/score.py`: remove temporary compatibility logic (`TODO: Remove this`) at lines 40-43 and apply all required downstream call site, docs, tests, and config updates in the same effort.
- [x] `deckard/__main__.py`: read from existing `.deckard_rc` when present to set environment defaults. Support both dotenv-style key/value and YAML. Precedence must be CLI args > environment variables > `.deckard_rc` defaults. Test and document.
- [x] `deckard/attack/base.py`: set labels to distinguish targeted from non-targeted evasion and poisoning attacks with a shared Mixin. Use label pattern `<target>_evasion_<metric>`. Test and document.

- [x] files: reduce `deckard/file.py` to thin runtime helpers only and unify canon files with the TypeDict objects in favor of authoritative schemas in each module's `canon.py`.
- [x] `test/test_frameworks/test_pytorch/test_pytorch_data.py`: add coverage for map-style dataset, iterable dataset, and pre-split dataloaders with unknown split tags.
- [x] Ensure DVC params exports contain only initialization configuration data so runtime pipelines are fully reproducible from emitted config artifacts.
- [x] Run targeted tests on touched files using fail-fast over core touched files.

### Phase 4: Refactors + Runtime Composition
papers/ folder is out of scope
This is a hard-cut. Tests, docs, and configs must be updated.

Testing standards for this phase were moved to:
- {doc}`../developers/contributor/testing`

- [ ] `deckard/*/base.py`: replace manual init parsing with `inspect`-based parsing and fast-failure during post_init.
- [ ] `deckard/*/base.py`: dataset_name, model_type, attack_type init params/attributes with "name" for uniformity.
- [ ] Unite coercion logic and default logic in BaseConfig.
- [ ] Rename ArtifactLoaderConfig to --> ArtifactLoaderMixin
- [ ] Reduce the number of other Mixins in core modules for simplicity, but keep run-time behavior the same.
  - [ ] Which mixins are mandatory to keep for behavior parity (must-not-break list)?
  - [ ] Which mixins can be merged into BaseConfig versus kept as thin runtime adapters?
  - [ ] Which contract tests should be treated as gating coverage for each removed/merged mixin?
- [ ] `deckard/experiment/canon.py`: correctly map components and subcomponents to existing `*Config` objects.
- [ ] `deckard/experiment/base.py`: compose data behavior at run time from mixins.
- [ ] `deckard/frameworks/pytorch/data.py`: compose data behavior at run time from mixins.
- [ ] `deckard/frameworks/pytorch/experiment.py`: compose data behavior at run time from mixins.
- [ ] Add a formal @fingerprint property to BaseConfig that contains the *Config initialization hash.
- [ ] Add fingerprint property to {class}`deckard.utils.BaseConfig` for run-time access (This should correctly calculate hashes on subsets of composition objects: e.g. `_apply_fit` and `_apply_predict` integration, partial pipelines, partial defense chains, sampling, splitting, folding, partial attack chains, etc.).
- [ ] Update all hashing flows to consume this public property.
- [ ] Create plugin install script with stable re-exports and discovery for core module plugins.
- [ ] Update tests accordingly. 
- [ ] Generate a canonical folder-by-folder fail-fast test flow from the docs/api/index flow and set it as the source of truth for Phase 4 acceptance + CI parity.

Phase 4 decisions (resolved):

- Unknown init kwargs: do not hard-fail yet; emit warnings and allow duck-typed passthrough where needed.
- Legacy aliases (`dataset_name`, `model_type`, `attack_type`): remove in Phase 4 and standardize on `name`.
- `ArtifactLoaderConfig` rename: cut directly to `ArtifactLoaderMixin` and update imports now (no compatibility shim).
- `deckard/experiment/canon.py` mapping: unify aliases to one canonical component/subcomponent mapping.
- `BaseConfig.fingerprint`: hash all components required for reproducibility and compute as the last post-init step (or the existing post-init-finalization hook).
- Plugin install interface: add a general script under `deckard/layers` that supports all existing plugin folders.

Open questions for Phase 4:

- Coercion/default precedence in `BaseConfig`: should defaults resolve before child composition coercion, or after child coercion with parent override/finalization?

### Phase 5: Stabilization + Contract Finalization

- [ ] Remove backwards-compatible logic that is no longer required.
- [ ] Audit code for redundancy and clarity after functionality stabilizes.
- [ ] Finalize run-time contracts for consistent API across composition targets (WIP).
