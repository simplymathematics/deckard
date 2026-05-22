# Changelog

## .98.2

- Data runtime architecture updated: `DataConfig` is the canonical runtime owner
  for loading, sampling, optional pipeline execution, and score orchestration.
- `DataConfig` is now documented as a legacy compatibility alias of
  `DataConfig`.
- Data pipeline execution now documented around the runtime
  `deckard.data.pipeline.base.DataPipeline` stage order
  (`fit_pre_sample`, `fit_X`, `fit_y`, `fit_Xy`).
- API documentation refresh for data and pipeline pages to remove stale
  `DataPipelineMixin` inheritance references.
- Added a new overview guide page: `docs/overview/data.md`, styled to match the
  scoring overview documentation.

## .98.1

- Core package updates in `deckard/` (entrypoints, config/declaration handling,
data/experiment utilities, and fairlearn scoring).
- Documentation updates in `docs/` (Sphinx config and multiple notebook
  refreshes, including Hydra/Optuna/Lifelines/Artifacts flows).
- Build artifact refresh in `build/` and notebook pipeline state updates
  (`docs/notebooks/dvc.lock`, `docs/notebooks/dvc.yaml`).
- isolated skelarn/pytorch from code model code
- isolated plugins from core model code
- unified scoring interface
- improved run-time of test-suite and documentation build

## Known TODOs

- finalize run-time contracts for consistent API across all composition targets
  (WIP)
- deckard/attack/base.py: set labels to distinguish targeted attacks from
  non-targeted attacks.
- deckard/layers/optimize.py: ensure data/model/attack \*\_file names are hashes
  when present in cfg.files.
- deckard/model/defend.py: make defense context-aware since ART defenses have
  \_apply_fit and \_apply_predict.
- deckard/plugins/fairlearn/score.py: remove temporary TODO-marked code path.
- Add licenses, hyperlinks, and paper references throughout the docs
- Remove adapter layer, move sklearn-only logic to framworks, ensure \*Config
  objects adhere to contract
- Update survival, art_attacks, art_defenses notebooks for clarity and scope
-scores like *demgraphic_parity* and *equalized odss* should not get group
scores since they are calculated across groups.
They currently have `nan` scores, and should not be calculated across sensitive
groups.
- `__main__.py` read from existing .deckard_rc
- `layers/` add comprehensive helps strings for the CLI
- `deckard/frameworks/pytorch/data.py` Compose data behavior at run-time from mixins
- Remove backwards-compatible logic
- Add stable re-exports for user convenience
