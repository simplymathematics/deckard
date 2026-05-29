# Layers

The {mod}`deckard.layers` package exposes CLI layer parser/main pairs and the
registry used by the top-level CLI router.

## Internals

Layer functions are intentionally small wrappers that parse runtime arguments,
delegate to domain modules, normalize for CLI and automation, and add `hydra` CLI configuration.


## Overview

Layers are thin orchestration entrypoints for higher-level tasks, such as:

- optimization runs
- result compilation
- plotting
- survival analysis
- progress monitoring
- Pareto-front trial selection

Each layer is registered in :data:`deckard.layers.layer_dict` as a
`[parser, main]` pair consumed by the top-level CLI.

## Optimization

The optimize layer is implemented in {mod}`deckard.layers.optimize` and
coordinates optimization workflows.

### Uses

- [Optuna studies](https://optuna.readthedocs.io/en/stable/reference/study.html)
- [Hydra callbacks and config composition](https://hydra.cc/docs/advanced/callbacks/)
- [OmegaConf runtime config resolution](https://omegaconf.readthedocs.io/)


Hydra-driven optimization workflows can feed this layer by persisting Optuna
study outputs and objective columns referenced by `optimizers` and `directions`.

### Full Walkthrough

The full optimization walkthrough, including single-run/multi-run directory
behavior, sweep parameters, sweeper configuration, and dashboard usage, is in
[Overview: Optimization](../../overview/optimize).

<!-- TODO: Add a complete optimize workflow walkthrough using a real multirun directory tree. -->
<!-- TODO: Add troubleshooting notes for objective-name mismatches and pruned-trial reporting. -->



```{seealso}

    Callback wiring used by optimization is implemented in
   {class}`deckard.layers.optimize.DefaultOptimizerCallback`.

    The canonical configuration lives in
    [examples/sklearn/config/default.yaml](../../examples/sklearn/config/default.yaml).

   For an end-to-end optimization guide, see
   [Overview: Optimization](../../overview/optimize).
```

### Minimal YAML Example

```yaml
callbacks:
   deckard_optuna:
      _target_: deckard.layers.optimize.DefaultOptimizerCallback
      directions: ${directions}
      optimizers: ${optimizers}
```

### CLI Example

```bash
deckard optimize \
   --config-dir examples/sklearn/config \
   --config-name default \
   hydra.mode=MULTIRUN \
   optimizers='[accuracy,evasion_accuracy]' \
   directions='[maximize,maximize]'
```

## Compile Results

The compile-results layer is implemented in {mod}`deckard.layers.compile_results`
and compiles Optuna study runs into one analysis-ready table.

### Uses

- [Optuna study storage](https://optuna.readthedocs.io/en/stable/reference/study.html)
- [pandas DataFrame tabular export](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)
- [PyYAML schema loading](https://pyyaml.org/wiki/PyYAMLDocumentation)

This layer is typically used after multirun optimization to aggregate trial
metrics and study metadata into a single file for plotting and reporting.

### Full Walkthrough

For end-to-end workflow context, including DVC and optimization integration, see
[Developer Optimization: DVC](../../developers/optimization/dvc).

<!-- TODO: Add a schema-driven study-name parsing walkthrough with before/after output columns. -->
<!-- TODO: Document recommended output formats (csv/parquet) per downstream layer. -->

```{seealso}

   Table compilation is implemented in
   {func}`deckard.layers.compile_results.compile_results_main`.

   The compiled output is commonly consumed by
   {doc}`../plot/index` and {doc}`pareto`.

   For broader optimization workflow details, see
   [Overview: Optimization](../../overview/optimize).
```

### Minimal YAML Example

```yaml
# config/my_schema.yaml
# This example yaml file is use to add study-name metadata as key-value pair metadata to each trial run
schema:
   sep: "_"
   model: 0 
   dataset: 1
```

### CLI Example

```bash
deckard compile_results \
   --output-file build/compiled_results.parquet \
   --optuna-db sqlite:///build/optuna.db
   --schema config/my_schema.yaml
```

### API Reference

```{eval-rst}
.. automodule:: deckard.layers.compile_results
   :members:
   :show-inheritance:
```

## Progress Bar

The progress-bar layer is implemented in {mod}`deckard.layers.progress_bar` and
tracks stage-level Optuna progress across studies and trials.

### Uses

- [tqdm progress bars](https://tqdm.github.io/)
- [Optuna storage summaries](https://optuna.readthedocs.io/en/stable/reference/study.html)
- [OmegaConf config loading](https://omegaconf.readthedocs.io/)
- [PyYAML stage/config parsing](https://pyyaml.org/wiki/PyYAMLDocumentation)

This layer polls configured storages and reports completion against expected
trial counts inferred from DVC stage definitions.

### Full Walkthrough

For DVC-stage and sweeper-storage setup details, see
[Developer Optimization: DVC](../../developers/optimization/dvc).

<!-- TODO: Add an end-to-end monitoring walkthrough across multiple storages. -->
<!-- TODO: Add notes on GridSampler trial-count inference and fallback behavior. -->

```{seealso}

   Progress tracking is implemented in
   {func}`deckard.layers.progress_bar.progress_bar_main`.

   Stage-count inference is derived from DVC stage metadata and Hydra sweeper
   storage settings.

   For optimization orchestration context, see
   [Overview: Optimization](../../overview/optimize).
```

### Minimal YAML Example

```yaml
# config/default.yaml (minimum required by progress_bar)
hydra:
   sweeper:
      # Required: storage is used to query study/trial completion.
      storage: sqlite:///optuna.db
      # Optional in code (defaults to 100), but set explicitly in default.yaml.
      n_trials: 100

# dvc.yaml (minimum required when stages are inferred)
stages:
   optimize:
      # Required for auto stage detection in progress_bar_main.
      cmd: python -m deckard optimize --config-name default --multirun

```

### CLI Example

```bash
deckard progress_bar \
   --hydra-cfg-file config/default.yaml \
   --dvc-file dvc.yaml \
   --stages optimize \
   --poll-interval 5
```

### API Reference

```{eval-rst}
.. automodule:: deckard.layers.progress_bar
   :members:
   :show-inheritance:
```

## Pareto

The pareto layer is implemented in {mod}`deckard.layers.pareto` and selects
best trials either by top-k single-objective ranking or multi-objective
Pareto-front filtering.

### Uses

- [Optuna trials dataframe](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)
- [paretoset multi-objective filtering](https://paretoset.readthedocs.io/en/latest/)
- [pandas numeric coercion and filtering](https://pandas.pydata.org/docs/)

This layer is typically run after result compilation when choosing candidate
trials for promotion, reporting, or follow-up experiments.

### Full Walkthrough

For optimization objective setup and selection workflows, see
[Overview: Optimization](../../overview/optimize).

<!-- TODO: Add a walkthrough for single-objective top-k vs multi-objective Pareto selection. -->
<!-- TODO: Add examples for objective-column mapping from metric names vs values_N columns. -->

```{seealso}

   Pareto and top-k selection is implemented in
   {func}`deckard.layers.pareto.pareto_main`.

   This layer consumes Optuna study metrics and produces filtered trial tables
   for downstream {doc}`../plot/index` workflows.

   For objective and direction conventions, see
   {doc}`../../developers/optimization/optimization`.
```

### Minimal YAML Example

```yaml
pareto:
   output_file: build/pareto.csv
   optuna_db: sqlite:///build/optuna.db
   study_name: baseline_search
   optimizers: accuracy,evasion_accuracy
   directions: maximize,maximize
   top_k: 5
```

### CLI Example

```bash
deckard pareto \
   --output-file build/pareto.csv \
   --optuna-db sqlite:///build/optuna.db \
   --study-name baseline_search \
   --optimizers accuracy,evasion_accuracy \
   --directions maximize,maximize \
   --top-k 5
```

### API Reference

```{eval-rst}
.. automodule:: deckard.layers.pareto
   :members:
   :show-inheritance:
```

## Survival Analysis

The survival layer is implemented in {mod}`deckard.layers.survival` and routes
between survival experiment execution and survival plot-only mode.

### Uses

- [Lifelines survival modeling](https://lifelines.readthedocs.io)
- [Hydra/OmegaConf config composition](https://hydra.cc)
- [pandas tabular survival data handling](https://pandas.pydata.org/docs/)

This layer validates and normalizes survival model specs, then runs either
full {class}`deckard.plugins.lifelines.experiment.SurvivalExperimentConfig`
execution or plot rendering based on config shape.

### Full Walkthrough

For survival architecture and scoring context, see
[Overview: Lifelines Extension](../../overview/extensions/lifelines).

<!-- TODO: Add full examples for experiment mode vs plot-only mode selection logic. -->
<!-- TODO: Add model-alias normalization examples (cox/weibull/log-normal). -->

```{seealso}

   Survival layer routing is implemented in
   {func}`deckard.layers.survival.survival_main`.

   Core experiment behavior is defined in
   {doc}`../plugins/lifelines` and {doc}`../experiment/index`.

   Developer-level runtime details are in
   {doc}`../../developers/experiment/experiment`.
```

### Minimal YAML Example

```yaml
survival:
   data:
      _target_: deckard.data.base.DataConfig
      dataset_name: lifelines_rossi
      target: arrest
   model: cox
   target: arrest
   duration_col: week
```

### CLI Example

```bash
deckard survival \
   --config-dir examples/sklearn/config \
   --config-name lifelines \
```

### API Reference

```{eval-rst}
.. automodule:: deckard.layers.survival
   :members:
   :show-inheritance:
```

## Plotting

The plotting layer is implemented in {mod}`deckard.layers.plot` and dispatches
to either Seaborn or Yellowbrick backends based on provided inputs.

### Uses

- [Seaborn statistical plotting](https://seaborn.pydata.org)
- [Yellowbrick model diagnostics](https://www.scikit-yb.org)
- [Hydra runtime argument parsing](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [OmegaConf structured config resolution](https://omegaconf.readthedocs.io/)

This layer supports file-backed plotting (Seaborn) and experiment-backed
plotting (Yellowbrick) with backend auto-selection.

### Full Walkthrough

For plotting architecture and backend behavior, see
[Developer Experiment: Plot](../../developers/experiment/plot).

<!-- TODO: Add backend auto-selection decision table (auto/yellowbrick/seaborn). -->
<!-- TODO: Add examples for experiment-backed yellowbrick vs file-backed seaborn runs. -->

```{seealso}

   Plot layer dispatch is implemented in
   {func}`deckard.layers.plot.plot_main`.

   Backend-specific API docs are in
   {doc}`../plot/index`, {doc}`../plugins/seaborn`, and
   {doc}`../plugins/yellowbrick`.

   For extension-level overview, see
   [Overview: Extensions](../../overview/extensions/index).
```

### Minimal YAML Example

```yaml
plot:
   backend: seaborn
   data_file: build/compiled_results.parquet
   plot_type: scatter
   x: accuracy
   y: evasion_accuracy
   plot_file: build/plots/accuracy_vs_evasion.png
```

### CLI Example

```bash
deckard plot \
   plot.backend=seaborn \
   plot.data_file=build/compiled_results.parquet \
   plot.plot_type=scatter \
   plot.x=accuracy \
   plot.y=evasion_accuracy \
   plot.plot_file=build/plots/accuracy_vs_evasion.png
```

### API Reference

```{eval-rst}
.. automodule:: deckard.layers.plot
   :members:
   :show-inheritance:
```




## Troubleshooting

- Ensure the requested subcommand exists in :data:`deckard.layers.layer_dict`.
- Check config compatibility with the selected layer.
- Verify optional dependencies for survival/plotting extensions are installed.

### See also

- {doc}`../experiment/index`
- {doc}`../plot/index`
- {doc}`../plugins/lifelines`
- {doc}`../file/index`
- {doc}`../utils/index`
