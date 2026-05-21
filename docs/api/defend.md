# Defense Runtime

## Overview

Deckard defense orchestration is centered on
{class}`deckard.model.defend.DefensePipelineConfig` and
{class}`deckard.model.defend.DefenseConfig`.

This layer composes model wrappers, ART defenses, and plugin hooks with
runtime mixin dispatch.

## Parent Config and Mixin Map

- {class}`deckard.model.defend.DefensePipelineConfig` uses
  {class}`deckard.model.defend._DefensePipelineConfigBehaviorMixin` to normalize
  defense chains, enforce ordering, and run before/after hooks.
- {class}`deckard.model.defend.DefenseConfig` uses
  {class}`deckard.model.defend._ARTDefenseBehaviorMixin` to resolve ART wrapper
  classes, parse defense families, and dispatch subtype handlers.
- {class}`deckard.model.preprocessor.PreprocessorDefenseConfig` inherits
  {class}`deckard.model.preprocessor._PreprocessorDefenseMixin` and
  {class}`deckard.model.defend.DefensePipelineConfig`.
- {class}`deckard.model.postprocessor.PostprocessorDefenseConfig` inherits
  {class}`deckard.model.postprocessor._PostprocessorDefenseMixin` and
  {class}`deckard.model.defend.DefensePipelineConfig`.
- {class}`deckard.model.trainer.TrainerDefenseConfig` inherits
  {class}`deckard.model.trainer._TrainerDefenseMixin` and
  {class}`deckard.model.defend.DefensePipelineConfig`.
- {class}`deckard.model.transformer.TransformerDefenseConfig` inherits
  {class}`deckard.model.transformer._TransformerDefenseMixin` and
  {class}`deckard.model.defend.DefensePipelineConfig`.
- {class}`deckard.model.detector.DetectorDefenseConfig` inherits
  {class}`deckard.model.detector._DetectorDefenseMixin` and
  {class}`deckard.model.defend.DefensePipelineConfig`.
- {class}`deckard.model.regularizer.RegularizerDefenseConfig` inherits
  {class}`deckard.model.regularizer._RegularizerDefenseMixin` and
  {class}`deckard.model.defend.DefensePipelineConfig`.

## Defense Families

- `preprocessor`
- `postprocessor`
- `trainer`
- `transformer`
- `detector`
- `regularizer`

External ART references:

- [ART preprocessor defenses](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html)
- [ART postprocessor defenses](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/postprocessor.html)
- [ART trainer defenses](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html)

## API Reference

```{eval-rst}
.. automodule:: deckard.model.defend
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.preprocessor
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.postprocessor
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.trainer
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.transformer
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.detector
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.regularizer
   :members:
   :show-inheritance:
```

## See also

- {doc}`model`
- {doc}`attack`
- {doc}`score`
- {doc}`train`
