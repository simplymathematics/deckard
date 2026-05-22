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

{class}`~deckard.data.DataConfig` is the runtime owner for optional preprocessing
via a `pipeline` attribute that accepts a
{class}`~deckard.data.pipeline.core.DataPipeline` object.

{class}`~deckard.data.DataPipelineConfig` remains available as a legacy alias
to {class}`~deckard.data.DataConfig` for compatibility.

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
{class}`~deckard.data.DataConfig.pipeline`.

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

{class}`~deckard.data.DataConfig` is the runtime owner for dataset loading,
splitting, optional preprocessing, plugin hooks, and data scoring.

At a high level, data execution follows this order:

1. Resolve and load `dataset_name` into `_X` and `_y`.
1. Run sampling to produce `X_train`, `X_test`, and optional `X_val`.
1. Optionally run pipeline transforms.
1. Optionally run data scoring (for split-scoped or pre-sample diagnostics).

The same config model supports:

- built-in sklearn/openml/synthetic datasets,
- optional dataset providers from plugin dependencies,
- local file-backed tabular datasets,
- framework-specific dataset classes such as PyTorch datasets.

## Dataset Catalog

### Core pre-loaded datasets

These names work directly with `deckard.data.base.DataConfig` in `dataset_name`:

- `openml.adult` (canonical) via [sklearn.fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
   and aliases `adult`, `openml_adult`, `sklearn.adult`, `sklearn_adult`
- `diabetes` via [sklearn.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- `digits` via [sklearn.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- `iris` via [sklearn.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- `make_classification` via [sklearn.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
- `make_regression` via [sklearn.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)

For sklearn datasets, all of these forms are accepted:

- bare name (for backward compatibility), for example `make_classification`
- dotted prefix, for example `sklearn.make_classification`
- underscore prefix, for example `sklearn_make_classification`

You can also use `.openml` filenames (for example `adult.openml`) to force the
generic OpenML loader path.

### Optional dependency dataset providers

When optional extras are installed, additional provider namespaces are
auto-discovered at runtime.

- Lifelines datasets (`pip install ".[lifelines]"`):
   loaded from [lifelines.datasets](https://lifelines.readthedocs.io/en/latest/lifelines.datasets.html)
   `load_*` functions.
- Yellowbrick datasets (`pip install ".[yellowbrick]"`):
   loaded from [yellowbrick.datasets](https://www.scikit-yb.org/en/latest/api/datasets/index.html)
   `load_*` functions.
- ART datasets (`pip install ".[art]"`):
   loaded from `art.utils` `load_*` functions and exposed as `art.<name>`.
- Hugging Face datasets (`pip install ".[datasets]"` and `huggingface_hub`):
   discovered from `huggingface_hub.list_datasets(sort="downloads", limit=100)`
   and exposed as `huggingface.<dataset_id>`.

For each discovered lifelines or yellowbrick dataset `name`, the following
forms are supported:

- `lifelines.name` and `lifelines_name`
- `yellowbrick.name` and `yellowbrick_name`

Bare provider names like `rossi` or `occupancy` are intentionally not
registered in the unified runtime loader registry.

### PyTorch and preloaded framework datasets

{class}`~deckard.frameworks.pytorch.data.PytorchDataConfig` supports fully
qualified class paths and shorthand aliases.

- Canonical discovery names: `torchvision.<DatasetClass>`
   (for example `torchvision.MNIST`, `torchvision.CIFAR10`)
- Compatible aliases: `torchvision_<DatasetClass>` and
   `torchvision.datasets.<DatasetClass>`
- Local built-in fairness dataset declarations:
   `fairlearn.TinyFairness` and
   `fairlearn.SyntheticImageSensitiveDataset`
   (with compatibility aliases including the fully-qualified deckard class path)

### Data scoring mode

{class}`~deckard.data.DataConfig` supports mode-aware dataset scoring via
`score_mode` with values:

- `train`
- `test`
- `val`
- `all`

`score_mode` controls split scope only. Hook stage lifecycle is configured
separately using scorer stage metadata (for example `pre-sample`,
`post-sample`, `post-pipeline`).

Mode details for `all`:

- `all`: runs data scorers on concatenated train and test splits
  (`X_train + X_test`, `y_train + y_test`).

If you need full-dataset diagnostics before splitting, configure the scorer
stage to `pre-sample` while using an explicit split scope mode.

## YAML Recipes

### 1) Synthetic classification (core)

```yaml
data:
   _target_: deckard.data.base.DataConfig
   dataset_name: make_classification
   classifier: true
   data_params:
      n_samples: 500
      n_features: 20
      n_informative: 10
      n_redundant: 5
      random_state: 42
   test_size: 0.2
   random_state: 42
```

### 2) sklearn built-in tabular dataset

```yaml
data:
   _target_: deckard.data.base.DataConfig
   dataset_name: diabetes
   classifier: false
   test_size: 0.2
   random_state: 42
```

### 3) Generic OpenML dataset

```yaml
data:
   _target_: deckard.data.base.DataConfig
   dataset_name: adult.openml
   classifier: true
   data_params:
      version: 2
   test_size: 0.2
```

### 4) File-backed dataset

```yaml
data:
   _target_: deckard.data.base.DataConfig
   dataset_name: ./data/train.csv
   target: label
   classifier: true
   drop:
      - id
   test_size: 0.2
   random_state: 7
```

### 5) Lifelines provider dataset (optional)

```yaml
data:
   _target_: deckard.plugins.lifelines.data.LifelinesDataConfig
   dataset_name: lifelines_rossi
   mode: native
   duration_col: week
   event_col: arrest
   classifier: false
   test_size: 0.2
```

### 6) Yellowbrick provider dataset (optional)

```yaml
data:
   _target_: deckard.data.base.DataConfig
   dataset_name: yellowbrick.concrete
   classifier: false
   test_size: 0.2
```

### 7) PyTorch torchvision dataset

```yaml
data:
   _target_: deckard.frameworks.pytorch.data.PytorchDataConfig
   dataset_name: torchvision.datasets.MNIST
   classifier: true
   data_params:
      train: true
      download: true
   train_size: 0.7
   test_size: 0.2
   random_state: 42
```

### 8) Fairness-aware PyTorch built-in dataset

```yaml
data:
   _target_: deckard.frameworks.pytorch.fairness_data.FairlearnPytorchDataConfig
   dataset_name: deckard.frameworks.pytorch.fairness_data.TinyFairness
   classifier: true
   sensitive_columns:
      - _sensitive
   train_size: 0.7
   test_size: 0.2
```

Sensitive feature parsing in fairness-aware torch configs follows two paths:

- Tuple path: if your dataset returns `(x, y, sensitive)` from `__getitem__`,
  the third tuple element is collected as sensitive metadata.
- Attribute path: if samples return only `(x, y)`, deckard falls back to a
  dataset-level `_sensitive` attribute (length must match the dataset size).

### 9) Custom dataset class from an installed package

```yaml
data:
   _target_: deckard.frameworks.pytorch.data.PytorchDataConfig
   dataset_name: my_package.MyDataset
   classifier: true
   data_params:
      split: train
      root: ./data
   train_size: None
   test_size: None
   random_state: 42
```

### 10) Custom dataset class from a local Python file

```yaml
data:
   _target_: deckard.frameworks.pytorch.data.PytorchDataConfig
   dataset_name: my_file.py:MyDataset
   classifier: true
   data_params:
      root: ./data
      download: false
   train_size: 0.7
   test_size: 0.2
   random_state: 42
```

### Discover optional dataset names at runtime

```python
from deckard.data.base import _lifelines_dataset_loaders, _yellowbrick_dataset_loaders

print("lifelines:", sorted(_lifelines_dataset_loaders().keys()))
print("yellowbrick:", sorted(_yellowbrick_dataset_loaders().keys()))
```

## Examples

```{seealso}

   Notebook-based examples for data loading, splitting, fairness/survival data workflows,
   and PyTorch datasets are documented in:

   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/lifelines.ipynb </notebooks/lifelines>`
   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`


```

## Internals

### Timing and logging

The data runtime records canonical timing keys in `times`, including
`data_load_time`, `data_sample_time`, `data_pipeline_time`, and
`data_score_time`.

The same values are mirrored onto matching top-level attributes on
{class}`~deckard.data.DataConfig` for compatibility.
The presence (or absence) of these timing values controls the execution of relevant steps.
These timings can be useful for comparing the run-time efficiency of different datasets
of various methods.
Logging is performed at key steps.

## Troubleshooting

If you encounter issues with dataset loading, ensure that:

- You have an active internet connection for datasets fetched from OpenML, etc.
- The selected optional dataset provider is installed (`lifelines`, `yellowbrick`,
  or `torch` extras when applicable).
- The selected file path and `target` column are correct for file-backed data.
- Otherwise, use one of the built-in dataset names from the catalog above.

### See also

- {doc}`model` — model configuration and training
- {doc}`sample` — pluggable train/test/val samplers
- {doc}`pipeline` — runtime data pipeline object and compatibility config aliases
- {doc}`experiment` — experiment orchestration
- {doc}`attack` — attack configuration
- {doc}`score` — scoring framework
- {doc}`pytorch` — PyTorch data integration
- {doc}`anjana` — anonymization-aware data
- {doc}`lifelines` — survival analysis data configuration
- {doc}`utils` — utility functions
