# Plot

## Introduction

This page is the canonical home for plotting behavior and API details.
It documents backend dispatch, data/experiment preparation boundaries,
and output persistence conventions.

## Plotting APIs

The plotting package exposes two public entry points:

- {class}`~deckard.plot.PlotConfig` chooses between the Seaborn and Yellowbrick backends.
- {class}`~deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig` and
  {class}`~deckard.plugins.yellowbrick.plot.YellowbrickConfigList` behave like
  experiment configs and prepare experiment outputs at most once before
  rendering plots.

Backend references:

- [Matplotlib](https://matplotlib.org/stable/)
- [Seaborn](https://seaborn.pydata.org)
- [Yellowbrick](https://www.scikit-yb.org)
- [lifelines plotting](https://lifelines.readthedocs.io/en/latest/lifelines.plotting.html)

```{eval-rst}
.. automodule:: deckard.plot
   :members:
   :undoc-members:
   :show-inheritance:
```

## Survival Plot Extension

Survival plotting configs are provided in a dedicated optional module.
See also: {doc}`lifelines`.

```{eval-rst}
.. automodule:: deckard.plugins.lifelines.plot
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.plugins.yellowbrick.plot
   :members:
   :show-inheritance:
```

## Overview

Plot configs separate plotting intent from execution details. They support:

- Seaborn-driven exploratory and report plots
- Yellowbrick visualizers for model diagnostics
- shared style and rc parameter configuration
- optional experiment-aware setup for plotting from prior outputs

Hydra users can compose plot backends and parameters through config groups and
overrides; see [Hydra](https://hydra.cc) and {doc}`experiment` for runtime
composition context.

## Examples

```{seealso}

   Notebook-based plotting examples for Yellowbrick and Seaborn are documented
   in:

   - {doc}`notebooks/yellowbrick.ipynb </notebooks/yellowbrick>`
   - {doc}`notebooks/seaborn.ipynb </notebooks/seaborn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## Minimal YAML Example

```yaml
plot:
   _target_: deckard.plot.base.PlotConfig
   backend: seaborn
   data_file: build/sklearn/seaborn_results.csv
   plot_type: scatter
   x: benign_accuracy
   y: evasion_accuracy
```

## Internals

The plotting module routes to backend-specific config objects and ensures
output files are written consistently. Yellowbrick plotting can hydrate
experiment context lazily before rendering to avoid repeated setup.

## Canonical backend ownership

- Seaborn plot configs behave like DataConfig extensions and can source data
  from in-memory frames, `data_file`, `data_config`, or Optuna storage.
- Yellowbrick plot configs behave like ExperimentConfig extensions and keep
  experiment preparation logic in Yellowbrick runtime modules.

## Optuna-backed Seaborn recipes

### Seaborn from Optuna storage

```yaml
plot:
   _target_: deckard.plugins.seaborn.plot.SeabornPlotConfig
   plot_type: scatter
   x: number
   y: value
   optuna_storage: sqlite:///build/optuna.db
   optuna_study_name: tuned_search
   optuna_query:
      trial_states:
         - COMPLETE
      sort_by: value
      ascending: false
      limit: 200
```

### Seaborn from DataConfig runtime payload

```yaml
plot:
   _target_: deckard.plugins.seaborn.plot.SeabornPlotConfig
   plot_type: line
   x: number
   y: value
   data_config:
      _target_: deckard.data.base.DataConfig
      dataset_name: optuna
      target: value
      data_params:
         optuna_storage: sqlite:///build/optuna.db
         study_name: baseline_search
         columns: [number, value]
```

## Troubleshooting

- Confirm plotting dependencies are installed for the selected backend.
- For Yellowbrick presets, ensure the selected `plot=<name>` exists under
  `examples/sklearn/config/plot <../examples/sklearn/config/plot>`\_.
- For Seaborn, ensure `data_file` exists and that `x`/`y`/`hue` columns
  are present in that dataset.
- Verify input score/data files exist and are in expected schema.
- Use explicit output paths to avoid confusion in multirun directories.

### See also

- {doc}`experiment` — experiment orchestration and result generation
- {doc}`score` — scoring framework that produces plotting data
- {doc}`seaborn` — statistical visualization with Seaborn
- {doc}`yellowbrick` — model interpretability visualizations
- {doc}`lifelines` — survival model and plotting integration
- {doc}`layers` — plot, survival, and pareto post-processing workflows
