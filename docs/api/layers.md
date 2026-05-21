# Layers

The {mod}`deckard.layers` package exposes CLI layer parser/main pairs and the
registry used by the top-level CLI router.

```{eval-rst}
.. automodule:: deckard.layers
   :members:
   :show-inheritance:
```


## Overview

Layers are thin orchestration entrypoints for higher-level tasks, such as:

- optimization runs
- result compilation
- plotting
- survival analysis
- progress monitoring
- Pareto-front trial selection

Each layer is registered in :data:`deckard.layers.layer_dict` as a
``[parser, main]`` pair consumed by the top-level CLI.

## Optimization 

The optimize layer is implemented in {mod}`deckard.layers.optimize` and
coordinates optimization workflows.

It uses:

- [Optuna studies](https://optuna.readthedocs.io/en/stable/reference/study.html)
- [paretoset](https://paretoset.readthedocs.io/en/latest/) for efficient
  Pareto front computation

Hydra-driven optimization workflows can feed this layer by persisting Optuna
study outputs and objective columns referenced by `optimizers` and `directions`.

The full optimization walkthrough, including single-run/multi-run directory
behavior, sweep parameters, sweeper configuration, and dashboard usage, is in
[Overview: Optimization](../overview/optimize).

```{eval-rst}
.. automodule:: deckard.layers.pareto
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.layers.plot
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.layers.survival
   :members:
   :show-inheritance:
```

## Examples

```{seealso}

    Callback wiring used by optimization is implemented in
    {class}`deckard.layers.optimize.OptunaStudyCallback`.

    The canonical configuration lives in
    [examples/sklearn/config/default.yaml](../../examples/sklearn/config/default.yaml).

   For an end-to-end optimization guide, see
   [Overview: Optimization](../overview/optimize).
```

## Minimal YAML Example

```yaml
callbacks:
   deckard_optuna:
      _target_: deckard.layers.optimize.OptunaStudyCallback
      directions: ${directions}
      optimizers: ${optimizers}
```

## Internals

Layer functions are intentionally small wrappers that parse runtime arguments,
delegate to domain modules, and normalize outputs for CLI and automation.

## Troubleshooting

- Ensure the requested subcommand exists in :data:`deckard.layers.layer_dict`.
- Check config compatibility with the selected layer.
- Verify optional dependencies for survival/plotting extensions are installed.

### See also

* {doc}`experiment`
* {doc}`plot`
* {doc}`lifelines`
* {doc}`file`
* {doc}`utils`
