# Optimize Developer Guide

This page is the implementation-oriented companion to the optimization runtime contract.

Related specs:

- [Optimization Runtime Contract](optimization.md)
- [Hydra and Optuna Orchestration Contract](hydra.md)
- [Pruning Runtime Contract](pruning.md)
- [DVC Pipeline Autogeneration Spec](dvc.md)

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

## Checklist Mapping

Phase 6 checklist coverage is provided by:

- Hydra notebook updates in `docs/notebooks/hydra.ipynb`
- Optimize notebook implementation in `docs/notebooks/optimize.ipynb`
- This developer guide as the optimize-facing implementation reference
