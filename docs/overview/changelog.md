# Changelog

## .98.2

- Finalized repository enforcement pass for core scope checks.
- Verified no baseline code quality enforcement violations for `deckard/`.
  via `scripts/repository_enforcement.py --scope deckard/`.
- Core runtime architecture updated-- each base Config object now has public methods, a documented execution order, and separate plugin and scoring hooks for easy extensibility.
- `canon.py` files now define inputs, outputs, and execution order of all base Config objects.
- `orchestration.py` handles scoring hooks based off canonical method order during Config.__call\__.
- `artifacts.py` handles persistence and pipeline dependency chains.
- `utils.py` contains a renamed BaseConfig object that handles coercion, fingerprinting, and runtime instantiation.
- Data sampler is now a canonical run-time object.
- ModelConfig objects now have configurable .defense and .trainer architectures for complex defense chains (including plugins), pruning, and analysis on pre-trained models.
- AttackConfig objects are fully documented.
- DetectorConfig objects now support training and filtering (pre-trained) modes.
- OptimizerConfig object now handles top-level `optuna` configuration and DefaultOptimizerCallback is fully documented for user-configurable optimzaiton.
- Created new ScoreDict canonical object with helper functions for parsing, viewing, storing, and updating runtime score dictionaries.
- `fairlearn` and `anjana` packages rewritten to use new plugin architecture.
- Drafted a DVCExperimentConfig for generating reproducible experiment files, `dvclive` integration for system monitoring, model-training updates, and added some `Vega-lite` specs for generating dvclive plots (WIP).

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
- add fingerprint property to BaseConfig for run-time access.
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
