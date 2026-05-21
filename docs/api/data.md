# Data

The {mod}`deckard.data` module defines the {class}`~deckard.data.DataConfig` dataclass,
which provides a unified interface for loading, generating, preprocessing, and
splitting datasets for machine learning experiments.\
It supports both real and synthetic datasets, as well as YAML/Hydra-based configuration.

```{eval-rst}
.. automodule:: deckard.data
   :members:
   :show-inheritance:
```

## Data Sampling

The {mod}`deckard.data.sample` module provides pluggable sampling strategies via
{class}`~deckard.data.sample.BaseSampler`
for robust train/test/validation splits.

```{eval-rst}
.. automodule:: deckard.data.sample
   :members:
   :show-inheritance:
```

## Data Preprocessing Pipelines

The {class}`~deckard.data.DataPipelineConfig` wraps scikit-learn's {class}`~sklearn.pipeline.Pipeline`
to enable configurable feature preprocessing with timing instrumentation.

Common transform components referenced in deckard pipeline configs:

- [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [`sklearn.compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

For torch-native transforms, see {doc}`pytorch` and:

- [`torchvision.transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html)

## Extensions

### Pipeline Extension

deckard exposes a configurable pipeline layer for data preprocessing via
{class}`~deckard.data.DataPipelineConfig`.

### Fairlearn Plugin

The fairlearn plugin adds group-aware sampling and fairness metrics with
`fairlearn` integration.
See also: {doc}`fairlearn`.

```{eval-rst}
.. automodule:: deckard.plugins.fairlearn.data
   :members:
   :show-inheritance:
```

### Torch Framework

The torch framework provides dataset loading and sampling for PyTorch and
torchvision-backed workflows.
See also: {doc}`pytorch`.

```{eval-rst}
.. automodule:: deckard.frameworks.pytorch.data
   :members:
   :show-inheritance:
```

## Lifelines plugin

Survival-specific experiment orchestration is split into a dedicated optional
module.
See also: {doc}`lifelines`.

```{eval-rst}
.. automodule:: deckard.plugins.lifelines.data
   :members:
   :show-inheritance:
```

## Overview

{class}`~deckard.data.DataConfig` can load well-known datasets such as:

- **Adult Income** (via OpenML)
- **Diabetes** and **Digits** (from scikit-learn)
- **Synthetic datasets** via `make_classification` or `make_regression`
- **pd.DataFrame files** that contain a `target` column or

It also supports **reproducible splits** via `train_test_split` with optional stratification,
timing instrumentation, and hashing for config tracking.

### Data scoring mode

{class}`~deckard.data.DataConfig` supports mode-aware dataset scoring via
`score_mode` with
values:

- `train`
- `test`
- `val`
- `pre-sample`

`pre-sample` runs data diagnostics against the full dataset before split
selection (`_X` / `_y`), while split modes run diagnostics on the selected
partition.

## Examples

```{seealso}

   Notebook-based examples for data loading, splitting, fairness/survival data workflows,
   and PyTorch datasets are documented in:

   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/lifelines.ipynb </notebooks/lifelines>`
   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`


```

## Minimal YAML Example

```yaml
data:
   _target_: deckard.data.base.DataConfig
   dataset_name: make_classification
   data_params:
      n_samples: 200
      n_features: 20
   test_size: 0.2
   random_state: 42
```

## Internals

### Timing and logging

The data loading and splitting process is timed, and the duration is stored in
the `_data_load_time` and `_data_sample_time` attributes of the
{class}`~deckard.data.DataConfig` instance.
This can be useful for comparing the run-time efficiency of different datasets
of various methods.
Logging is performed at key steps.

## Troubleshooting

If you encounter issues with dataset loading, ensure that:

- You have an active internet connection for datasets fetched from OpenML, etc.
- The specified .csv/.html/.json file path is correct and the file is accessible.
- Otherwise, use one of the built-in datasets or synthetic data generation options.

### See also

- {doc}`model` — model configuration and training
- {doc}`sample` — pluggable train/test/val samplers
- {doc}`pipeline` — data pipeline config and DataPipelineMixin behavior
- {doc}`experiment` — experiment orchestration
- {doc}`attack` — attack configuration
- {doc}`score` — scoring framework
- {doc}`pytorch` — PyTorch data integration
- {doc}`anjana` — anonymization-aware data
- {doc}`lifelines` — survival analysis data configuration
- {doc}`utils` — utility functions
