# Optimization

This page explains how deckard optimization is configured and executed across
single-run and multi-run workflows.

Optimization orchestration integrates:

- [Hydra](https://hydra.cc) runtime composition and sweeping

- [Optuna](https://optuna.org) studies, trials, and samplers

- {class}`deckard.layers.optimize.DefaultOptimizerCallback` lifecycle hooks

For layer API reference, see [Layers](/api/layers/index).

## Source Configuration

The canonical baseline configuration is:

- [examples/sklearn/config/default.yaml](../../examples/sklearn/config/default.yaml)

## Minimal Callback Example

```yaml
callbacks:
  deckard_optuna:
    _target_: deckard.layers.optimize.DefaultOptimizerCallback
    directions: ${directions}
    optimizers: ${optimizers}
```

## Single-run

In single-run mode, Hydra executes one composed configuration and writes
outputs under `hydra.run.dir`.

```yaml
hydra:
  run:
    dir: outputs/logs/${experiment_name}
```

`hydra.run.dir` is the resolved output folder for one execution, including
callback-resolved files such as params and score payloads.

## Multi-run

In multi-run mode (`--multirun`), Hydra creates a sweep root at
`hydra.sweep.dir`, then nests each trial under `hydra.sweep.subdir`.

```yaml
directions: [maximize, maximize, maximize]
optimizers: [accuracy, evasion_accuracy, attack_generation_time]

hydra:
  sweep:
    dir: outputs/logs/
    subdir: ${hydra.sweeper.study_name}/${hydra.job.num}
  sweeper:
    study_name: ${data_alias}_${model_alias}_${defense_alias}_${attack_alias}
    storage: sqlite:///optuna.db
```

`hydra.sweep.dir` sets the root for the full sweep, while
`hydra.sweep.subdir` defines each trial folder (for example
`${study_name}/0`, `${study_name}/1`, ...).

## Sweep Parameters

Deckard optimization relies on Hydra sweeper parameters plus top-level metric
settings:

- `directions`: objective direction for each metric (`maximize`/`minimize`)

- `optimizers`: metric names aligned with `directions`. These come from the configured scorers and the configured scoring stage/mode.

- `hydra.sweeper.study_name`: Optuna study identifier

- `hydra.sweeper.storage`: Optuna storage backend URI

- `hydra.sweeper.n_trials`: number of trials to run

- `hydra.sweeper.n_jobs`: parallel worker count

- `hydra.sweeper.max_failure_rate`: tolerated failure ratio before aborting

```yaml
directions: [maximize, maximize, maximize]
optimizers: [accuracy, evasion_accuracy, attack_generation_time]

hydra:
  sweeper:
    study_name: ${data_alias}_${model_alias}_${defense_alias}_${attack_alias}
    storage: sqlite:///optuna.db
    n_trials: 100
    n_jobs: 1
    max_failure_rate: 1.0
```

## Optuna Sweeper Configuration

Deckard uses Hydra's Optuna sweeper plugin to translate Hydra multirun
overrides into Optuna trials. Typical config includes the sweeper target,
sampler, and optional pruner.

```yaml
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    sampler:
      _target_: optuna.samplers.RandomSampler
      seed: 42
```

References:

- [Hydra Optuna Sweeper docs](https://hydra.cc/docs/plugins/optuna_sweeper/)

- [Optuna samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/)

## Optuna Dashboard

Use Optuna Dashboard to inspect trial history, parameter importance, and best
trials for the same `storage` and `study_name` used in sweeper config.

```bash
optuna-dashboard sqlite:///optuna.db
```

After launching, open the local URL printed by the command and select the
study created by `hydra.sweeper.study_name`.

References:

- [Optuna Dashboard](https://optuna-dashboard.readthedocs.io/en/latest/)

- [optuna-dashboard docs](https://optuna-dashboard.readthedocs.io/en/latest/)

## Easy Parallelization with Optuna and RDB

By default, the optuna database for optimization uses sqlite version 3. 
While this is a handy way to store data for serial experiments, this is likely to lead to lock and race conditions when conducting experiments in parallel. 
You will have to configure another option (by setting up a database and configuring the URL in the config).
For distributed experiments with many parallel workers, see the [Optuna Parallelization Docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-process-optimization)
