# Data Pipeline

## Overview

Deckard pipeline configuration for data preprocessing is owned by
{class}`deckard.data.base.DataPipelineConfig`.

`DataPipelineConfig` composes parent data runtime behavior from
{class}`deckard.data.base.DataConfig` with mixin behavior from
{class}`deckard.data._mixins.DataPipelineMixin`.

## Parent Config and Mixin Map

- {class}`deckard.data.base.DataPipelineConfig` inherits
  {class}`deckard.data._mixins.DataPipelineMixin` and
  {class}`deckard.data.base.DataConfig`.
- {class}`deckard.data.pipeline.core.DefaultDataPipelineConfig` is the default
  no-op pipeline variant.
- {class}`deckard.data.pipeline.core.FairlearnDataPipelineConfig` is a fairness
  family marker variant.
- {class}`deckard.data.pipeline.core.AnjanaDataPipelineConfig` is an anonymization
  family marker variant.
- Optional torch variant: `PytorchDataPipelineConfig` (available when torch extras
  are installed) in {mod}`deckard.frameworks.pytorch.data`.

## External References

- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html)

## API Reference

```{eval-rst}
.. automodule:: deckard.data.pipeline
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.data.pipeline.core
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.data._mixins
   :members:
   :show-inheritance:
```

## See also

- {doc}`data`
- {doc}`sample`
- {doc}`fairlearn`
- {doc}`pytorch`
