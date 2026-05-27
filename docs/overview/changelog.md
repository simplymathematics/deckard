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
- Reworked `fairlearn` and `anjana` integration to use the plugin architecture.
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

- finalize run-time contracts for consistent API across all composition targets
  (WIP)
- deckard/attack/base.py: set labels to distinguish targeted attacks from
  non-targeted attacks.
- add fingerprint property to {class}`deckard.utils.BaseConfig` for run-time access.
  \_apply_fit and \_apply_predict~~.
- Add licenses, hyperlinks, and paper references throughout the docs
  objects adhere to contract~~
- Update survival, art_attacks, art_defenses notebooks for clarity and scope
-scores like *demgraphic_parity* and *equalized odss* should not get group
scores since they are calculated across groups.
They currently have `nan` scores, and should not be calculated across sensitive
groups.
- `__main__.py` read from existing .deckard_rc
- document .matplotlibrc and demonstrate extension.
- `layers/` add comprehensive helps strings for the CLI
- `deckard/frameworks/pytorch/data.py` Compose data behavior at run-time from mixins
- Remove backwards-compatible logic
- create install script for plugins that:
  - Adds stable re-exports for user convenience
  - discovers data, model, defense, and or attacks
- Audit code for redundancy and clarity after functionality has stablized.
