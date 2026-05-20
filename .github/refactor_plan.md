# Deckard Refactor Plan (Condensed)

Repository: https://github.com/simplymathematics/deckard/tree/refactor-squashed

## Completed Tasks

- Canonical config migration completed: authoritative declarations moved to `examples/sklearn/config` and `examples/pytorch/config`.
- Runtime declaration lifecycle implemented in `deckard/declarations.py`:
  - config root discovery
  - `DECKARD_CONFIG_DIRS` support
  - YAML parsing + `safe_store` registration
  - duplicate and malformed declaration handling
- Core/framework/plugin boundary refactor substantially completed:
  - plugin family extraction under `deckard/plugins`
  - framework family structure under `deckard/frameworks`
  - broad import/isolation smoke coverage added
- Adapter/core decoupling work completed in primary paths:
  - runtime typing extraction to `frameworks/types.py`
  - adapter/contract inheritance removed from primary runtime `*Config` classes
  - widespread call-site migration away from legacy contract shims
- Naming/enforcement infrastructure established:
  - repository enforcement checks and CI wiring
  - naming checks for `Config`/`Mixin`/`Plugin` conventions
  - adapter boundary checks and docstring/type checks added
- Shared mixin infrastructure consolidated for major cross-family behavior:
  - sensitive-columns shared mixin extraction
  - model runtime mixin integration and focused mixin tests
- Test modernization largely completed:
  - many compose/unit/import-smoke updates
  - representative focused slices already green in current branch state

Reference docs for full details:
- `docs/developers/config_declaration_architecture.md`
- `docs/developers/core_plugin_boundaries.md`
- `docs/developers/mixin_plugin_rules.md`
- `docs/developers/naming_conventions.md`
- `docs/developers/docstring_standard.md`
- `docs/developers/design_principles.md`
- `docs/developers/development.md`

## TODO

- Final shim cleanup:
  - remove remaining compatibility shims
  - finish migrating any lingering `_target_` and import call sites to canonical plugin/framework paths
- Data pipeline semantics:
  - complete validation of `fit_*` staged behavior (`fit_X`, `fit_y`, `fit_Xy`, `fit_pre-sample`, `fit_post-sample`, `dtype`-isolated transforms)
  - add/expand tests for stage-flag combinations and dtype routing behavior
- Scoring roadmap items:
  - configurable data/model/attack-only scorers
  - scoring mode for pre/post-defense
  - post-sample scoring mode support for transformed `X`/`y`
- Attack scoring consistency:
  - enforce context-aware scoring mode behavior by attack type
- Experiment layer follow-through:
  - finish `_Mixin -> Plugin -> Config` rule enforcement
  - complete compose + experiment refactor/test consolidation tasks still open
- Plugin-family completion:
  - ensure all plugin YAML naming/path conventions are finalized
  - complete plugin discovery/external config loading tasks where still open
  - verify plugin-family isolation in absence of sibling families across remaining families
- Enforcement backlog:
  - enforce `Default*ScoreConfig` naming convention repository-wide
  - complete MyST-native Google docstring and explicit typing backlog in remaining hotspots
  - continue reducing ambiguous `Any`/`object` runtime payload usage where feasible
- Final integration gate:
  - run full compose/unit/experiment suite in order
  - execute representative sklearn/pytorch/plugin smoke matrix
  - document final pass/fail counts and close remaining checklist items

## Pass Update (2026-05-19)

- [x] Final shim cleanup (targeted subset):
  - migrated lifelines survival tests to canonical imports from `deckard.plugins.lifelines.experiment`
  - migrated legacy experiment declarations shim to canonical re-export (single-source constants)
  - removed test dependency on legacy `deckard.plot.yellowbrick_plots` shim
- [x] Data pipeline semantics follow-through (targeted subset):
  - enforced documented validation: `fit_y` and `fit_Xy` cannot both be true
  - included `fit_Xy` stages in legacy `_init_pipeline()` X-stage collection
  - preserved untyped/unknown-dtype steps in dtype-routed pipeline construction (`data/base.py` and `data/_mixins.py`)
  - added tests for stage-flag execution and dtype-routing retention of untyped steps
- [x] Scoring roadmap initial implementation (low-risk subset):
  - added `score.scoring_type` routing support (`data`/`model`/`attack`/`detector`/`experiment`) in `ExperimentConfig`
  - added focused unit test for `scoring_type: data` routing
- [x] Framework/core finalization:
  - deleted legacy `frameworks/adapters.py` and `frameworks/core.py`
  - confirmed all runtime/test imports migrated to canonical paths
- [ ] Next Phase - Priority Stack:
  1. **Expanded scoring modes** (high-value, in-flight)
     - implement pre/post-defense scoring mode support
     - implement pre/post-pipeline scoring mode support
     - post-sample scoring mode for transformed `X`/`y` payloads
  2. **Experiment layer follow-through** (dependency for final gate)
     - finish `_Mixin -> Plugin -> Config` rule enforcement
     - complete compose + experiment refactor/test consolidation
  3. **Final integration gate** (gating release)
     - full compose/unit/experiment execution matrix
     - representative sklearn/pytorch/plugin smoke matrix
     - document pass/fail counts and close remaining checklist items
  4. **Enforcement backlog** (ongoing)
     - enforce `Default*ScoreConfig` naming convention repository-wide
     - MyST-native Google docstring and explicit typing backlog
  5. **Audit Redundant Code** prefer mixins and run-time resolution/configuration over complex inheritance schemes with unclear orchestration.
  6. **Normalize Configs** ensure that kwargs/methods/attributes are consistent across core/framework *Config objects and with the docs. This should only be handled after the Enforcement backlog.
  7. **Delete Redundant Code** Check the repository for redundant/legacy code
  8. **Full Coverage test** use scripts/coverage.sh. The goal is every file in deckard/ above 90% coverage.
  9. **Minimize tests** minimize repeated tests/configurations.
  10. **Port papers to new configs** 
  10. **Test and Fix all workflows** Final gate for 