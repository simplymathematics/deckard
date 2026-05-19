# Changelog

## .98.1

- Core package updates in `deckard/` (entrypoints, config/declaration handling, data/experiment utilities, and fairlearn scoring).
- Documentation updates in `docs/` (Sphinx config and multiple notebook refreshes, including Hydra/Optuna/Lifelines/Artifacts flows).
- Build artifact refresh in `build/` and notebook pipeline state updates (`docs/notebooks/dvc.lock`, `docs/notebooks/dvc.yaml`).
- isolated skelarn/pytorch from code model code
- isolated plugins from core model code
- unified scoring interface
- improved run-time of test-suite and documentation build


## Known TODOs
- finalize run-time contracts for consistent API across all composition targets (WIP)
- deckard/attack/base.py: set labels to distinguish targeted attacks from non-targeted attacks.
- deckard/layers/optimize.py: ensure data/model/attack *_file names are hashes when present in cfg.files.
- deckard/model/defend.py: make defense context-aware since ART defenses have _apply_fit and _apply_predict.
- deckard/plugins/fairlearn/score.py: remove temporary TODO-marked code path.
- Add licenses, hyperlinks, and paper references throughout the docs
- Remove adapter layer, move sklearn-only logic to framworks, ensure *Config objects adhere to contract
- Update survival, art_attacks, art_defenses notebooks for clarity and scope
