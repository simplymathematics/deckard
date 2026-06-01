# Changelog

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

### Audited Open TODO List

#### Runtime and core code

- [ ] `deckard/attack/base.py`: resolve targeted/non-targeted attack label TODO in runtime payload path.
- [ ] `deckard/model/base.py`: replace inspect-based fallback TODO in `_sync_model_signature_from_estimator` with a stable constructor-parameter extraction strategy.
- [ ] `deckard/experiment/canon.py`: complete component/sub-component manifest mapping TODO in experiment runtime manifest builder.
- [ ] `deckard/__main__.py`: remove stale config discovery TODO comment now that `.deckard_rc` defaults are supported and tested.

#### API documentation content debt

- [ ] `docs/api/layers/index.md`: complete all inline TODO walkthrough placeholders for optimize, compile-results, progress-bar, pareto, survival, and plotting sections.

#### Test implementation debt

- [ ] `test/test_frameworks/test_pytorch/test_pytorch_data.py`: implement placeholder custom dataset/dataloader/tensorset mixin tests.

#### Tooling and docs-process cleanup

- [x] `scripts/fix_docs_crosslinks.py` + `docs/developers/contributor/documentation.md`: replace `TODO-BROKEN-LINK` placeholder fallbacks with concrete docs index fallbacks and document the follow-up review expectation.

#### Backlog classification notes

- [ ] `docs/developers/future/refactor_plan.md`: maintain as forward backlog only; do not duplicate backlog items in changelog.
