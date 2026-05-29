# Hydra and Optuna Orchestration Contract

This document defines the Hydra-native orchestration contract for Deckard
optimization workflows.

It covers sweeper configuration, callback lifecycle behavior, custom search-space
usage, and run/multirun execution semantics.

## Goals

- Keep Hydra integration first-class and explicit.
- Preserve compatibility with `hydra-optuna-sweeper` behavior.
- Separate parameter proposal concerns from runtime execution concerns.
- Make run metadata, output paths, and params persistence deterministic.

## Runtime Layers

Hydra orchestration is composed of:

- CLI routing and `@hydra.main` dispatch.
- Sweeper config for Optuna trial generation.
- Hydra callback lifecycle for setup/sync/persistence behavior.

## Sweeper Contract

Expected sweeper fields:

- `study_name`
- `storage`
- `direction` or `directions`
- `sampler`
- `n_trials`, `n_jobs`
- optional `params`
- optional `custom_search_space`

Study/storage are required for multirun trial synchronization behavior.

## custom_search_space Contract

`custom_search_space` is used for search-space logic only.

Responsibilities:

- define conditional parameter relationships
- perform Trial-driven `suggest_*` calls during proposal

Non-responsibilities:

- runtime trainer Trial transport
- score payload persistence
- post-run trial attribute synchronization

This keeps proposal logic separate from execution lifecycle behavior.

## Callback Lifecycle Contract

Hydra callback remains the canonical place for:

- multirun study initialization
- objective metric-name setup
- per-job output path resolution
- params file persistence
- per-job score payload persistence
- trial user-attribute synchronization

Callbacks should stay thin and delegate complex runtime behavior to dedicated
runtime objects when needed.

Current default policy shape:

- {class}`deckard.layers.optimize.DefaultOptimizerCallback` is adapter-thin and delegates optimization policy
    values to {class}`deckard.layers.optimize.OptimizerConfig`.
- {class}`deckard.layers.optimize.OptimizerConfig` self-configures from top-level composed config keys
    (`directions`, `optimizers`, `report_trial_attrs`, `pruning_enabled`,
    `dvclive_enabled`, `dvclive_dir`) without requiring a nested optimizer
    dictionary in callback YAML.

## Mode Normalization

Config normalization behavior differs by mode:

- Run mode: keep single-run config shape.
- Multirun mode: enforce sweeper requirements and resolve per-trial output paths.

Path resolution should prefer explicit callback constructor overrides, then Hydra
output directories.

## Trial Identity Normalization

Hydra job ids may vary by launcher.

Contract:

- numeric ids map directly to trial number
- suffixed ids (for example `__main___0`) resolve to trailing numeric component
- unresolved ids must fail gracefully for trial sync and log clear diagnostics

## Output and Params Persistence

Per-job params and score files must be resolved before runtime execution and
written into configured paths.

Persistence should include:

- params snapshot (`params.yaml`)
- score payload (`scores.json`)
- error/log path normalization through file config

## Cross-Document Dependencies

- Runtime optimization contract: [Optimization Runtime Contract](../optimization)
- Pruning contract: [Pruning Runtime Contract](../pruning)
- DVC/DVCLive contract: [DVC Pipeline Autogeneration Spec](../dvc)

## Test Requirements

At minimum, tests must validate:

- run vs multirun mode normalization
- callback path resolution precedence
- metric name setup and objective filtering consistency
- trial id normalization across launcher formats
- stable params and score persistence



## See also

- {doc}`/developers/optimization/optimization`
- {doc}`/developers/optimization/dvc`
- {doc}`/developers/design/orchestration`
