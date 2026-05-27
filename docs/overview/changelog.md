# Changelog

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

- [ ] Audit individual files for missing cross-references (e.g., [AttackConfig](../api/modules) or [fairlearn](extensions/index) references should link to docs), create a way to track this per-file, and add it to repository enforcement so that we can systematically eliminate the lack of cross-references. Also audit for broken `:mod` and `:doc` syntax inside all docs/ *.md files and all docs/notebooks/*.ipynb files
- [ ] Ensure that all docs/ files cross-reference instead of merely wrap objects with ``.
- [ ] Fix broken :mod syntax
- [ ] Fix broken :doc syntax
- [ ] Ensure that notebook .md cells properly cross-link to the docs.
- [x] Add dependency licenses and hyperlinks across all top-level docs and relevant index pages.
- [x] Add links to plugins and their licenses.
- [x] Create a new LICENSES file for this content and move the existing LICENSES section from docs/index.md there and link to it from the index.
- [x] Document `matplotlibrc` behavior and provide extension examples.
- [x] Update [lifelines](extensions/index), `art_attacks`, and `art_defenses` notebooks for clarity and scope.
- [x] `docs/developers/workflows.md`: document Codecov (or equivalent) repository settings checklist.
- [x] `docs/developers/workflows.md`: wire security scanning workflow/checklist updates in docs.
- [x] `docs/developers/workflows.md`: document security scanning integration with repository enforcement and CI gates.
- [x] `docs/developers/workflows.md`: document Dependabot repository settings and update flow.
- [ ] audit index files and toctrees to mirror the source code
- [ ] Ensure that the flow between base module mirrors the experiment flow
- [ ] Ensure that the flow from base -> frameworks is obvious and parallel
- [ ] Ensure that the flow from base -> plugins is obvious and parallel

### Phase 3: Targeted Code + Workflow Updates


- [ ] `.github/workflows/test-optional-dependencies.yml`: add [anjana](extensions/index) optional dependency test coverage.
- [ ] `deckard/plugins/fairlearn/score.py`: remove temporary compatibility logic (`TODO: Remove this`).
- [ ] `deckard/__main__.py`: read from existing `.deckard_rc` when present.
- [ ] `deckard/attack/base.py`: set labels to distinguish targeted from non-targeted evasion and poisoning attacks with a shared Mixin.
- [ ] `deckard/layers/optimize.py`: ensure `data/model/attack *_file` names are hashes when present in `cfg.files`.
- [ ] `test/test_frameworks/test_pytorch/test_pytorch_data.py`: add dataloader and custom tensor mixin coverage for PyTorch data paths.
- [ ] Ensure that dvc params file exports do not include in-memory objects and that everything is reproducible from the emitted config and hashed

### Phase 4: Refactors + Runtime Composition

- [ ] `deckard/model/base.py`: replace manual init parsing with `inspect`-based parsing.
- [ ] `deckard/experiment/canon.py`: correctly map components and subcomponents to existing `*Config` objects.
- [ ] `deckard/frameworks/pytorch/data.py`: compose data behavior at run time from mixins.
- [ ] Add a formal @fingerprint property to BaseConfig that contains the *Config initialization hash.
- [ ] Add fingerprint property to {class}`deckard.utils.BaseConfig` for run-time access (This should correctly calculate hashes on subsets of composition objects: e.g. `_apply_fit` and `_apply_predict` integration, partial pipelines, partial defense chains, sampling, splitting, folding, partial attack chains, etc.).
- [ ] Update all hashing flows to consume this public property.
- [ ] Create plugin install script with stable re-exports and discovery for core module plugins.

### Phase 5: Stabilization + Contract Finalization

- [ ] Remove backwards-compatible logic that is no longer required.
- [ ] Audit code for redundancy and clarity after functionality stabilizes.
- [ ] Finalize run-time contracts for consistent API across composition targets (WIP).
