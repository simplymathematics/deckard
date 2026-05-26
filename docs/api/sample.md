# Sampling

## Overview

Sampling in Deckard is owned by {class}`deckard.data.DataConfig` through its
`sampler` field and concrete sampler callables from {mod}`deckard.data.sample`.

Samplers return `(train_idx, test_idx, val_idx)` and are applied during data
runtime orchestration.

## Parent Config and Runtime Behavior

- Parent config: {class}`deckard.data.DataConfig`
- Runtime integration point: {meth}`deckard.data.DataConfig._resolve_sample` and public lifecycle methods
  {meth}`deckard.data.DataConfig.sample` / {meth}`deckard.data.DataConfig.sample_data`
- Hydra sampler defaults are registered in {func}`deckard.data.sample.register_sampler_configs` under
  the `sample` config group.

Available sampler configs:

- {class}`deckard.data.sample.SplitSampler`
- {class}`deckard.data.sample.KFoldSampler`
- {class}`deckard.data.sample.ShuffleSampler`

## External References

- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)

## API Reference

```{eval-rst}
.. automodule:: deckard.data.sample
   :members:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
sample:
  name: deckard.data.sample.KFoldSampler
  n_splits: 5
  shuffle: true
```

## See also

- {doc}`data`
- {doc}`pipeline`
- {doc}`experiment`
