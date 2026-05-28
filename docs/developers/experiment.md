# Experiment Canon and Validation Workflow

This guide consolidates Phase 7 validation and documentation contracts for experiment runtime behavior.

## Purpose

Define internal experiment runtime owner contracts for stage ordering, mode
propagation, hook orchestration, cache and persistence schema, and boundaries
between core orchestration and extension adapters.

## Capabilities

- Define canonical stage execution order and defense-stage retraining policy.
- Define deterministic hook bundle composition and hook event contracts.
- Define runtime serialization and cache compatibility guarantees.
- Define extension boundaries for plugin and framework-adapter orchestration.

Related docs:

- [Experiment Runtime Contract](optimization)
- [Hydra and Optuna Orchestration Contract](hydra)
- [Pruning Runtime Contract](pruning)
- [DVC Pipeline Autogeneration Spec](dvc)
- [Refactor Plan](refactor_plan)

## Experiment Canon

Canonical experiment runtime state is organized into five buckets:

1. `files`: output and artifact aliases resolved through {class}`deckard.file.FileConfig`
2. `times`: canonical timing keys with optional extension keys
3. [scores](../api/data): stage-aware and mode-aware score payloads
4. `outputs`: cache metadata, hook trace, and stage intermediates
5. `params`: manifest generated from composed runtime config

The canonical stage order remains:

1. [load](../api/data)
2. [sample](../api/sample)
3. [train](../api/train)
4. `apply_fit_defense`
5. `apply_predict_defense`
6. [attack](../api/attack)
7. [detect](../api/detector)
8. [score](../api/score)
9. `persist`

Defense-stage training rule:

1. `apply_fit_defense` is the fit-time defense stage and is the only defense stage that may trigger a new training step.
2. `apply_predict_defense` is a predict-time defense stage and must not trigger retraining.
3. Retraining is limited to pretrained-load flows where a configured defense requires fit application.

## Bundle Authoring

Hook bundles are authored with {class}`deckard.plugins.base.HookBundle` and {class}`deckard.plugins.HookPlugin` and composed through `deckard.plugins.base.compose_hook_plugins`.

Authoring rules:

1. canonical bundle first
2. user bundles next
3. explicit plugins last
4. duplicate hook signature entries are deduplicated by `(hook_name, method_name)`

This ordering keeps runtime behavior deterministic while allowing additive extension.

## Hook Contract

Experiment-stage hook names follow:

- `before_<stage>`
- `after_<stage>`

Runtime trace entries are stored under `outputs["hooks"]["trace"]` and include:

1. `component`
2. `stage`
3. `event`
4. `run`

Score-stage hooks (`after_data_score`, `after_model_score`, `after_attack_score`, `after_detector_score`) may return score dictionaries that are merged into component score buckets.

## Serialization Schema

Experiment runtime state serialization is schema-versioned.

Expected runtime YAML top-level fields:

1. `schema_version`
2. `experiment`
3. `params`
4. `runtime`

Load-time rules:

1. current and prior supported schema major versions may load
2. future schema major versions are rejected
3. malformed payloads fail fast with explicit errors

## DVC Workflow

DVC generation is additive and uses runtime manifests:

1. stage plan is derived from canonical experiment stages
2. generated command surface uses `deckard optimize ...`
3. `dvc.yaml` and `params.yaml` generation is deterministic for equivalent manifests
4. identity-derived output directories differ correctly for run and multirun modes
5. Vega-Lite outputs are normalized to `.vl.json`

Metrics and report policy:

1. default metrics are file-based score artifacts
2. optimizer-aware naming is used for plot token generation
3. summary and report generation are mode-aware (`.html`, `.ipynb`, `.md`)

## Validation Checklist Mapping

Phase 7 validation is covered by targeted tests in:

- `test/test_experiment/test_experiment_canon.py`
- `test/test_experiment/test_experiment.py`
- `test/test_experiment/test_experiment_dvc.py`
- `test/test_layers/test_optimize.py`

These tests cover:

1. hook stage ordering and bundle merge behavior
2. native config composition and cache reuse
3. Hydra compose behavior for stage selection and multitrial settings
4. callback/policy delegation between {class}`deckard.layers.optimize.DefaultOptimizerCallback` and {class}`deckard.layers.optimize.OptimizerConfig`
5. pruning propagation via `TrialPruned`
6. DVC contract behavior for plots, identities, metrics policy, and reports

## See also

- {doc}`../api/experiment`
