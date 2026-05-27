# Optimization Runtime Contract

This document defines the runtime contract for Deckard optimization flows.

It clarifies how Hydra orchestration, Optuna study metadata, experiment execution,
and score serialization interact during single-run and multirun workflows.

## Goals

- Keep optimization runtime deterministic and config-first.
- Preserve Hydra-native callback compatibility.
- Ensure trial metadata sync is stable across launchers.
- Keep objective extraction and score payload handling explicit.
- Support pruning and DVCLive integration without coupling to CLI entrypoint signatures.

## Architecture

Optimization runtime is split into three concerns:

- Orchestration: Hydra sweeper + callback lifecycle.
- Runtime optimization policy: {class}`deckard.layers.optimize.OptimizerConfig` (or equivalent config object)
    that owns trial resolution, reporting, and pruning policy behavior.
- Execution: `deckard.layers.optimize.optimize_main` instantiates and runs {class}`deckard.experiment.ExperimentConfig`.
- Post-run synchronization: objective filtering, trial user attributes, persisted score payload.

This split allows composition changes without changing the user-facing command surface.

## OptimizerConfig Contract

{class}`deckard.layers.optimize.OptimizerConfig` is the canonical runtime policy object for optimization behavior.

Responsibilities:

- define runtime optimization policy independent of CLI routing
- resolve runtime trial context from Hydra/Optuna metadata
- expose report/prune decision hooks used by trainer/runtime layers
- coordinate optional DVCLive logging behavior

Non-responsibilities:

- Hydra callback lifecycle ownership
- direct CLI argument parsing
- replacing {class}`deckard.experiment.ExperimentConfig` as execution root

Design intent:

- keep {class}`deckard.layers.optimize.DefaultOptimizerCallback` as the Hydra-native adapter
- keep callback methods thin and delegate optimization policy logic into
    {class}`deckard.layers.optimize.OptimizerConfig`
- keep `default.yaml` configuration as the source of runtime policy values

## Entrypoint Contract

`optimize_main(cfg)` remains config-only and does not require a direct Trial argument.

Requirements:

- Input is any DictConfig/dict-like payload coercible to a dictionary.
- Runtime target is forced to {class}`deckard.experiment.ExperimentConfig`.
- Runtime object must return a dict-like score payload.
- Raw score payload must be preserved for callback hooks that run after sweeper wrapping.
- Runtime optimization policy should be configured through {class}`deckard.layers.optimize.OptimizerConfig`
    rather than additional positional function arguments.

## Trial Resolution Contract

Runtime trial context is derived from Hydra metadata and Optuna storage:

1. Resolve sweeper config (`storage`, `study_name`).
2. Normalize Hydra job identity into trial-number-compatible form.
3. Resolve selected trial by trial number.
4. Update trial-level user attributes (for example `experiment_name`).

This logic must be resilient to launcher-specific job id formats.

## Objective Contract

Optimization objectives are controlled by `optimizers` and `directions`.

Rules:

- `diff` directions are filtered from Optuna objective dimensions.
- Missing optimize scores use direction-aware fallback values.
- Non-optimized scores are preserved as trial attributes.
- Study metric names are set from filtered objective keys.

## Runtime Score Contract

Runtime score payloads must remain unfiltered at execution boundaries and only be
filtered when deriving optimization values.

Required behavior:

- Persist full score payload to score artifact.
- Inject normalized experiment id into payload.
- Keep auxiliary timing/count/metadata fields available for DVC metrics expansion.

## Integration With Pruning

Pruning behavior is owned by trainer/runtime flow and documented in
[Pruning Runtime Contract](pruning).

Optimization runtime must provide trial context that supports:

- reporting intermediate values
- prune decision checks
- clean propagation of `TrialPruned` status

## Integration With Hydra

Hydra orchestration details (sweeper, callback lifecycle, custom search space)
are documented in [Hydra and Optuna Orchestration Contract](hydra).

Hydra integration model for this contract:

- callback remains Hydra-native lifecycle adapter
- {class}`deckard.layers.optimize.OptimizerConfig` remains runtime optimization policy object
- callback delegates optimization policy behavior to configured runtime object

## Integration With DVC and DVCLive

DVC stage generation and DVCLive reporting contract are documented in
[DVC Pipeline Autogeneration Spec](dvc).

Optimization runtime is the source of truth for:

- metrics payload shape
- objective key semantics
- trial identity and study metadata




# Optimize Developer Guide

This page is the implementation-oriented companion to the optimization runtime contract.

Related specs:

- [Optimization Runtime Contract](optimization)
- [Hydra and Optuna Orchestration Contract](hydra)
- [Pruning Runtime Contract](pruning)
- [DVC Pipeline Autogeneration Spec](dvc)

## Purpose

Use this guide when implementing or reviewing optimize flows in code, notebooks, and tests.

Core expectations:

- one Hydra default profile should support run and multirun
- stage and trial fan-out control must come from runtime overrides
- callback lifecycle stays adapter-thin and delegates policy to {class}`deckard.layers.optimize.OptimizerConfig`
- persisted params and scores stay deterministic and reproducible

## Canonical Surfaces

Primary code paths:

- `deckard/layers/optimize.py`
- `deckard/layers/optuna_callback.py`
- `deckard/experiment/base.py`
- `deckard/experiment/canon.py`

Primary config source:

- `examples/sklearn/config/default.yaml`

Primary demonstration notebooks:

- `docs/notebooks/hydra.ipynb`
- `docs/notebooks/optimize.ipynb`

## Runtime Contract Checks

When validating optimize behavior, confirm:

1. callback target resolves to `deckard.layers.optimize.DefaultOptimizerCallback`
2. callback directions and optimizers align with {class}`deckard.layers.optimize.OptimizerConfig`
3. single-run stage selection works via runtime override (`stage=...`)
4. multirun fan-out works via sweeper overrides (`hydra.sweeper.*`)
5. files-only persistence aliases are used for params, scores, logs, and errors
6. Optuna storage and study metadata produce stable trial identity mapping
7. pruning-enabled runs can record or surface `TrialPruned` states without corrupting artifacts
8. run and multirun params materialize deterministically for equivalent settings

## Recommended Notebook Validation Sequence

1. run `docs/notebooks/hydra.ipynb` to verify compose and command templates
2. run `docs/notebooks/optimize.ipynb` sections for single-run, multirun, and pruning
3. verify generated params snapshots under `docs/build/`

## Test Requirements

At minimum, tests must cover:

- deterministic objective filtering and metric naming
- trial resolution across launcher id formats
- score payload preservation through callback post-processing
- trial attribute sync on persisted studies
- compatibility when optional components are absent