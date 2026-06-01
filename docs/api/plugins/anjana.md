# Anjana Integration

## Overview

The Anjana integration provides anonymization-aware data preparation and scoring
for privacy-preserving machine-learning workflows.

## Parent Core Modules and Behavior Deltas

Parent core pages:

- {doc}`/api/data/index`
- {doc}`/api/model/index`
- {doc}`/api/score/index`

Behavior deltas in this integration:

- anonymization-aware data policy hooks before and after pipeline stages,
- privacy/utility score tails merged with canonical score payloads,
- optional dependency layer that does not change core orchestration ownership.

Related Deckard docs:

- {doc}`/api/data/index` for data pipeline composition
- {doc}`/api/model/index` for training/evaluation stages after anonymization
- {doc}`/api/score/index` for utility/privacy scoring hooks
- {doc}`/overview/extensions/index` for extension mapping

External references:

- [Anjana package on PyPI](https://pypi.org/project/anjana/)
- [scikit-learn
  Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  for transform composition patterns used with anonymization steps

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

## Canon Runtime Contract

Anjana data behavior is implemented as policy hooks on top of canonical data
runtime orchestration:

- pipeline policy hook at `before_sample` for pre-sample anonymization
- score tail policy hook at `after_score_post_pipeline`
- score scope remains split-scoped (`train|test|val|all`)
- persistence remains files-only via `files={...}`
- the public privacy mixin is {class}`~deckard.plugins.anjana.data.PrivacyBehaviorMixin`

Anjana tail metrics are emitted as flat, collision-safe score entries so they
compose with other plugin score tails, including Fairlearn-last merges.

## See also

- {doc}`/api/model/defend` — defense runtime integration for ART and plugin-backed workflows
- {doc}`/api/score/index` — scoring runtime and privacy scorer configuration
- {doc}`/api/data/index`
- {doc}`/api/data/pipeline`
- {doc}`/api/plugins/fairlearn`
- {doc}`/developers/data/data`
- {doc}`/developers/contributor/migration`
