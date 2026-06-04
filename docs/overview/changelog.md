# Changelog

## 0.98.4

- Landed a broad hardening and centralization sweep across core config/runtime surfaces, including DataConfig, ModelConfig, ScoreConfig, AttackConfig, ExperimentConfig, CLI composition paths, plot config handling, and plugin canonical-method integration.
- Consolidated resolver and warning initialization behavior into shared canonical paths and improved config resolution consistency across entrypoints.
- Completed path/artifact centralization updates, including artifact IO helper consolidation and compatibility cleanup for repeated-save/update flows.
- Expanded shared sub-component resolution and runtime composition behavior, with improved compatibility for trainer/sampler/defense-style declaration paths.
- Improved PyTorch integration robustness, including better pre-split handling and sensitive-column parsing hardening.
- Refreshed and stabilized tests during the hardening pass (compose/integration/plugin coverage updates and targeted regression fixes).
- Refreshed notebook and docs pipelines repeatedly throughout the window (multiple notebook reruns/fixes, flaky notebook cleanup, docs updates, and local docs build/version-sync maintenance).
- Updated dataset/config inputs used by examples and notebook flows (adult dataset source/config refresh, paper/config refreshes, and example updates).
- Removed broken cache behavior discovered during cleanup and aligned cache behavior with current runtime expectations.
- Applied additional core/runtime safety improvements and compatibility fixes discovered during regression-driven refactoring.
- Closed TODOs from changelog: targeted vs non-targeted attack label normalization in `deckard/attack/base.py`, constructor-parameter extraction stabilization in `deckard/model/base.py`, and component/sub-component runtime manifest mapping completion in `deckard/experiment/canon.py`.


## 0.98.3

- Improved documentation quality, cross-linking, licensing information, and developer workflow guidance.
- Refreshed and stabilized notebooks, examples, and documentation build pipelines.
- Expanded integration support and test coverage for Anjana, Fairlearn, Lifelines, and Yellowbrick.
- Simplified configuration architecture by standardizing naming, reducing indirection, and consolidating ownership of runtime behavior.
- Improved runtime composition across experiments, data pipelines, samplers, attacks, defenses, scoring, and framework integrations.
- Strengthened reproducibility through cleaner configuration exports, deterministic fingerprinting, structured runtime manifests, and artifact hashing.
- Added support for loading defaults from .deckard_rc files with documented precedence alongside environment variables and CLI arguments.
- Improved attack and defense runtime consistency, including clearer targeted-attack metrics, flattened configuration surfaces, and compatibility cleanup.
- Expanded automated validation with broader PyTorch coverage, regression testing, compose-contract testing, and fail-fast verification workflows.
- Reduced technical debt through removal of legacy compatibility paths, module responsibility cleanup, runtime consistency fixes, and general codebase hardening.

## 0.98.2

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

## 0.98.1

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

Repository TODO audit completed from source, test, docs, scripts, workflow, and top-level project files.

Scan scope:
- Included: `deckard`, `docs`, `examples`, `scripts`, `test`, `.github`, `README.md`, `pyproject.toml`
- Excluded: `.git`, `.venv`, `.dvc`, `build`, `docs/build`, `outputs`, generated notebooks and HTML


#### Backlog classification notes

- [ ] `docs/developers/future/refactor_plan.md`: maintain as forward backlog only; do not duplicate backlog items in changelog.
