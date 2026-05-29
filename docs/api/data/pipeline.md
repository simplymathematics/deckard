# Data Pipeline

## Contract References

- Canonical API contract page: {doc}`/api/data/pipeline`
- Developer authoring contract: {doc}`/developers/data/pipelines`
- Shared config/mixin/plugin contracts: {doc}`/developers/design/configs`, {doc}`/developers/extensions/mixins`, {doc}`/developers/extensions/plugins`

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
  {meth}`~deckard.data.pipeline.base.DataPipeline.fit_pre_sample`,
  {meth}`~deckard.data.pipeline.base.DataPipeline.fit_X`,
  {meth}`~deckard.data.pipeline.base.DataPipeline.fit_y`,
  {meth}`~deckard.data.pipeline.base.DataPipeline.fit_Xy`.
- {mod}`deckard.data.pipeline` re-exports {class}`deckard.data.base.DataConfig`
  and {class}`deckard.data.pipeline.base.DataPipeline` as the public pipeline
  package entrypoints.
- Optional torch variant: {class}`deckard.frameworks.pytorch.data.PytorchDataConfig` (available when torch extras
  are installed) in {mod}`deckard.frameworks.pytorch.data`.

## External References

- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html)

## API Reference

```{eval-rst}
.. automodule:: deckard.data.pipeline.base
   :members:
   :exclude-members: DataConfig, PytorchDataConfig
   :no-index:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
data:
  _target_: deckard.data.base.DataConfig
  name: make_classification
  pipeline:
    scale:
      name: sklearn.preprocessing.StandardScaler
```

## See also

- {doc}`/api/data/index`
- {doc}`/api/data/sample`
- {doc}`/api/plugins/fairlearn`
- {doc}`/api/pytorch/index`
- {doc}`/developers/data/data`
- {doc}`/developers/data/pipelines`
- {doc}`/developers/contributor/migration`
