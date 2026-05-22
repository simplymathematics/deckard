# Data Pipeline

## Overview

Deckard data preprocessing is executed by the runtime
{class}`deckard.data.pipeline.core.DataPipeline` object.

{class}`deckard.data.base.DataConfig` owns pipeline orchestration and accepts an
optional `pipeline` runtime object.

{class}`deckard.data.base.DataPipelineConfig` remains as a legacy alias to
{class}`deckard.data.base.DataConfig`.

## Parent Config and Mixin Map

- {class}`deckard.data.base.DataConfig` is the canonical runtime owner.
- {class}`deckard.data.pipeline.core.DataPipeline` executes stage order:
  `fit_pre_sample`, `fit_X`, `fit_y`, `fit_Xy`.
- {class}`deckard.data.base.DataPipelineConfig` is a compatibility alias.
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

## Minimal YAML Example

```yaml
data:
  _target_: deckard.data.base.DataPipelineConfig
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
