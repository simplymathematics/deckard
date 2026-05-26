# Data Pipeline

## Introduction

This page is the canonical home for pipeline runtime behavior and API details.
It documents DataPipeline stage execution, parent-config ownership, extension
boundaries, and API surface.

## Overview

Deckard data preprocessing is executed by the runtime
{class}`deckard.data.pipeline.base.DataPipeline` object.

{class}`deckard.data.base.DataConfig` owns pipeline orchestration and accepts an
optional `pipeline` runtime object.

{class}`deckard.data.base.DataConfig` remains as a legacy alias to
{class}`deckard.data.base.DataConfig`.

## Parent Config and Mixin Map

- {class}`deckard.data.base.DataConfig` is the canonical runtime owner.
- {class}`deckard.data.pipeline.base.DataPipeline` executes stage order:
  `fit_pre_sample`, `fit_X`, `fit_y`, `fit_Xy`.
- {class}`deckard.data.base.DataConfig` is a compatibility alias.
- {class}`deckard.data.pipeline.base.DataConfig` is the default
  no-op pipeline variant.
- {class}`deckard.data.pipeline.base.DataConfig` is a fairness
  family marker variant.
- {class}`deckard.data.pipeline.base.DataConfig` is an anonymization
  family marker variant.
- Optional torch variant: {class}`deckard.frameworks.pytorch.data.PytorchDataConfig` (available when torch extras
  are installed) in {mod}`deckard.frameworks.pytorch.data`.

## External References

- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html)

## API Reference

```{eval-rst}
.. automodule:: deckard.data.pipeline
   :members:
  :exclude-members: DataConfig
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.data.pipeline.base
   :members:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
data:
  _target_: deckard.data.base.DataConfig
  dataset_name: make_classification
  pipeline:
    scale:
      name: sklearn.preprocessing.StandardScaler
```

## See also

- {doc}`data`
- {doc}`sample`
- {doc}`fairlearn`
- {doc}`pytorch`
- {doc}`../developers/data`
- {doc}`../developers/migration`
