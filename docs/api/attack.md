# Attack

## Overview

The attack module orchestrates adversarial example generation across supported
backends and attack families.

It provides:

- attack configuration and instantiation
- attack execution over model/data outputs
- artifact persistence for attacked samples and labels
- attack-aware scoring hooks used by experiment orchestration

## Attack Families

Deckard runtime attack dispatch is centered on
{class}`deckard.attack.base.AttackConfig`, which resolves a family/subtype and
dispatches to mixin handlers via plugins.

Supported attack families:

- `evasion`
- `poisoning`
- `extraction`
- `inference`

Supported inference subtypes:

- `membership_inference`
- `attribute_inference`
- `model_inversion`
- `reconstruction`

Scoring keys include:

- `blackbox_membership_inference`
- `blackbox_evasion`
- `whitebox_evasion`
- `blackbox_attribute_inference`
- `whitebox_attribute_inference`

## Common Attack and Defense Components

Attack configuration in this module is typically composed with
{doc}`model` defense pipelines and {doc}`score` attack scorer profiles.

Common ART evasion attack classes used with deckard attack configs:

- [`FastGradientMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm)
- [`ProjectedGradientDescent`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#projected-gradient-descent-pgd)
- [`BasicIterativeMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#basic-iterative-method-bim)

Common ART defenses that are paired with attack runs in deckard:

- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`SpatialSmoothing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#spatial-smoothing)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

Related Deckard docs:

- {doc}`model` for configuring defense pipelines
- {doc}`defend` for defense pipeline and defense mixin dispatch
- {doc}`score` for attack-specific scorer profiles
- {doc}`pytorch` for ART estimator integration in torch workflows
- {doc}`/overview/extensions` for cross-framework extensions map

## Examples

```{seealso}

   Notebook-based attack workflows are documented in:

   - {doc}`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## Minimal YAML Example

```yaml
attack:
   _target_: deckard.attack.base.AttackConfig
   attack_type: art.attacks.evasion.FastGradientMethod
   attack_params:
      eps: 0.1
   attack_size: 100
```

## API Reference

```{eval-rst}
.. automodule:: deckard.attack.base
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.evasion
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.poisoning
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.extraction
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.inference
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.reconstruction
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.frameworks.pytorch.attack
   :members:
   :show-inheritance:
```

## Troubleshooting

- Ensure the selected attack backend matches the active model backend.
- Confirm attack parameters are valid for the chosen ART/Fairlearn attack type.
- Verify the attack receives compatible input shapes and labels.

### See also

- {doc}`experiment` — experiment orchestration
- {doc}`model` — model configuration and execution
- {doc}`defend` — defense pipeline configuration and mixin behavior
- {doc}`data` — data loading and split handling
- {doc}`sample` — sampling/split strategy definitions
- {doc}`score` — attack scoring profiles
