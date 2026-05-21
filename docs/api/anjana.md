# Anjana Integration

## Overview

The Anjana integration provides anonymization-aware data preparation and scoring
for privacy-preserving machine-learning workflows.

Related Deckard docs:

- {doc}`data` for data pipeline composition
- {doc}`model` for training/evaluation stages after anonymization
- {doc}`score` for utility/privacy scoring hooks
- {doc}`/overview/extensions` for extension mapping

External references:

- [Anjana package on PyPI](https://pypi.org/project/anjana/)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for transform composition patterns used with anonymization steps

## Notebook Example

```{seealso}

   End-to-end examples are documented in {doc}`/notebooks/anjana`.

```
## API Reference

```{eval-rst}
.. automodule:: deckard.plugins.anjana
   :members:
   :show-inheritance:
```



## Typical Workflow

1. Configure an Anjana-aware data pipeline.
2. Train a compatible model configuration.
3. Score utility and privacy-sensitive outcomes.
