# DVC Overview

This page explains DVC semantics used in Deckard and serves as the generic DVC introduction.

For notebook-pipeline wiring details in this repository, see the notebook-focused guide in [docs/notebooks/dvc.ipynb](../notebooks/dvc.ipynb).

## Cross-References

For command semantics and edge-case flags, see official DVC docs:

- [DVC Command Reference](https://dvc.org/doc/command-reference)
- [`dvc repro` documentation](https://dvc.org/doc/command-reference/repro)
- [`dvc push` documentation](https://dvc.org/doc/command-reference/push)
- [`dvc pull` documentation](https://dvc.org/doc/command-reference/pull)
- [Pipelines and Stages](https://dvc.org/doc/user-guide/pipelines)
- [Metrics, Plots, and Params](https://dvc.org/doc/user-guide/experiment-management)

Related Deckard docs:

- [Developer DVC Spec](/developers/optimization/dvc)
- [Optimization Contract](/developers/optimization/optimization)
- [Hydra Contract](/developers/optimization/hydra)
- [Pruning Contract](/developers/optimization/pruning)

## Deps

DVC dependencies (`deps`) declare source inputs and code paths that invalidate a stage when changed.

Example from `docs/notebooks/dvc.yaml`:

```yaml
notebook_dvc:
    deps:
        - dvc.ipynb
        - ../../deckard/experiment/
        - ../../deckard/layers/
        - ../../deckard/file.py
        - ../../deckard/utils.py
        - ../../examples/sklearn/config/default.yaml
        - ../../examples/sklearn/config/files/default.yaml
        - ../../examples/sklearn/config/attack/hsj.yaml
        - ../../examples/sklearn/config/defense/class-labels.yaml
        - ../../examples/sklearn/config/plot/
```

## Outs

DVC outputs (`outs`) declare materialized artifacts tracked by a stage.

Concrete `outs` example from `docs/notebooks/dvc.yaml`:

```yaml
notebook_optuna:
    outs:
        - ./build/notebook_artifacts/optuna/single_study.pkl
        - ./build/notebook_artifacts/optuna/multi_study.pkl
        - ./build/notebook_artifacts/optuna/single_best_params.json
        - ./build/notebook_artifacts/optuna/multi_best_params.json
```

## Metrics

DVC metrics (`metrics`) are structured, diff-friendly values used for evaluation and CI comparison.

Concrete metrics example from `docs/notebooks/dvc.yaml`:

```yaml
notebook_dvclive:
    metrics:
        - ./build/dvclive/dvclive/summary.json
```

## Plots

DVC plots (`plots`) are visualization-oriented artifacts for trend inspection and report rendering.

Concrete plots example from `docs/notebooks/dvc.yaml`:

```yaml
notebook_dvclive:
    plots:
        - ./build/dvclive/dvclive_feature_spec.vl.json
```

## Params

DVC params (`params`) capture configuration inputs that define run identity and reproducibility.

In this repository's notebook pipeline file (`docs/notebooks/dvc.yaml`), stages currently rely heavily on config files in `deps` rather than explicit top-level `params` blocks.

If you want explicit params tracking there, add a `params:` block to the stage and point it at specific config keys/files.

## dvc repro

Use this command to (re)run pipeline stages whose dependencies or params changed.

Run only one stage:

```bash
dvc repro notebook_dvc
```

Force-run notebook-prefixed stages (useful after notebook refactors):

```bash
dvc repro --force notebook_*
```

What it does:

- Recomputes stage state from declared dependencies.
- Executes stage command when inputs changed (or when forced).
- Refreshes tracked artifacts for stages that declare `outs`/`metrics`/`plots`.

## dvc push

Use this command to upload cache objects from local cache to configured remote storage.

```bash
dvc push
```

Typical sequence after reproducing artifact-producing stages:

```bash
dvc repro notebook_dvclive notebook_optuna && dvc push
```

What it does:

- Transfers cache objects for tracked stage artifacts to remote.
- Enables collaborators/CI to retrieve identical artifacts via `dvc pull`.

## dvc pull

Use this command to download required cache objects from remote into local cache and workspace.

```bash
dvc pull
```

Use after updating Git metadata:

```bash
git pull && dvc pull
```

What it does:

- Restores tracked outputs for stages that materialize artifacts.
- Aligns local workspace artifacts with remote-backed DVC state.
- Keeps local runs consistent with collaborator and CI outputs.
