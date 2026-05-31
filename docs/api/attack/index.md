# Attack

## Basic flow state

`resolve mode/stage -> generate -> predict -> score -> persist`.

## Purpose

Define user-facing attack runtime owner behavior, including stage versus mode
semantics, scoring outputs, persistence paths, and boundaries for framework
adapters and plugin integrations.

## Capabilities

- Normalize attack mode and stage tokens for consistent execution.
- Generate attacked samples across supported attack families.
- Produce attack prediction payloads for downstream evaluation.
- Emit attack-scoped metrics and persist canonical attack artifacts.
- Consume model outputs from {doc}`/api/model/index` and emit metrics used by {doc}`/api/score/index`.

Implementation-level runtime contracts are documented in {doc}`/developers/attack/attack`.

## Outputs

- Attack artifacts (`attack_file`, `attack_predictions_file`, `score_file`).
- Attacked labels/predictions and score-ready payloads.
- Attack timing fields (`attack_generation_time`, `attack_prediction_time`, `attack_score_time`).
- Attack score dictionaries merged with runtime timing metadata.

## Introduction

This page is the canonical home for attack module behavior and API details.
It documents attack-family dispatch, mode/stage semantics, persistence
contracts, and framework extensions.

## Overview

The attack module orchestrates adversarial example generation across supported
backends and attack families.

It provides:

- attack configuration and instantiation
- attack execution over model/data outputs
- artifact persistence for attacked samples and labels
- attack-aware scoring hooks used by experiment orchestration

Canonical runtime contract:

- files: attack artifacts persist through files-only paths (`attack_file`, `attack_predictions_file`, `score_file`)
- times: `attack_generation_time`, `attack_prediction_time`, `attack_score_time`
- scores: attack metrics merged with runtime timing metadata
- stage: canonical stage tokens (`pre-attack`, `post-attack`) with compatibility aliases normalized at runtime
- mode: split scope (auto, train, test, val) distinct from stage/hook lifecycle

## Attack Families

Deckard runtime attack dispatch is centered on
{class}`deckard.attack.base.AttackConfig`, which resolves a family/subtype and
dispatches to direct `*AttackConfig` runtime handlers

Attack execution ordering is explicit in runtime metadata and defaults to
`post-defense`, so downstream scoring and detector layers can reason about
defense/attack sequencing consistently.

Supported attack families:

- [`evasion`](#evasion-attacks)
- [`poisoning`](#poisoning-attacks)
- [`extraction`](#extraction-attacks)
- [`inference`](#inference-attacks)

Supported inference subtypes:

- [`membership_inference`](#membership-inference-attacks)
- [`attribute_inference`](#attribute-inference-attacks)
- [`model_inversion`](#model-inversion-attacks)
- [`reconstruction`](#reconstruction-attacks)

(evasion-attacks)=
### Evasion attacks

Evasion attacks measure post-attack model robustness by perturbing evaluation
inputs while preserving their semantic class intent. In deckard outputs, these
typically map to evasion score contexts such as `blackbox_evasion` and
`whitebox_evasion`.

Common implementations include gradient- and decision-based methods such as
`FastGradientMethod`, `ProjectedGradientDescent`, `HopSkipJump`, and
`BoundaryAttack`.

(poisoning-attacks)=
### Poisoning attacks

Poisoning attacks measure training-time robustness by injecting crafted samples
or labels into the training process and evaluating downstream degradation (for
example, accuracy drop or calibration drift) after retraining.

In notebook and config workflows this commonly includes `PoisoningAttackSVM`.

(extraction-attacks)=
### Extraction attacks

Extraction attacks measure model theft risk by querying a target model and
training a surrogate to replicate the target's behavior. Evaluation focuses on
fidelity between target and surrogate predictions, plus downstream task metrics
on held-out data.

In notebook and config workflows this commonly includes `CopycatCNN`.

(inference-attacks)=
### Inference attacks

Inference attacks measure privacy leakage from model behavior or outputs rather
than only predictive robustness.

(membership-inference-attacks)=
#### Membership inference attacks

Membership inference measures whether an attacker can determine if a sample was
part of the target model's training set. Deckard emits membership-scoped
metrics under the `membership_inference_*` namespace.

Common implementation: `MembershipInferenceBlackBox`.

(attribute-inference-attacks)=
#### Attribute inference attacks

Attribute inference measures whether sensitive or hidden attributes can be
recovered from model predictions and observed features. Deckard emits these as
`inferred_<attribute>_*` metrics.

Common implementation: `AttributeInferenceBlackBox`.

(model-inversion-attacks)=
#### Model inversion attacks

Model inversion measures how much feature-level information can be reconstructed
about target classes or individuals from model outputs or gradients.

Common implementation: `MIFace`.

(reconstruction-attacks)=
#### Reconstruction attacks

Reconstruction attacks measure the ability to recover representative training
inputs or latent data structure from model behavior. These are tracked as
reconstruction-oriented attack outputs in deckard workflows.

Scoring keys include:

- `blackbox_membership_inference`
- `blackbox_evasion`
- `whitebox_evasion`
- `blackbox_<attribute>_inference`
- `whitebox_<attribute>_inference`

## Common Attack and Defense Components

Attack configuration in this module is typically composed with
{doc}`/api/model/index` defense pipelines and {doc}`/api/score/index` attack scorer profiles.

Common ART evasion attack classes used with deckard attack configs:

- [`FastGradientMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm)
- [`HopSkipJump`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#hopskipjump)
- [`BoundaryAttack`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#boundary-attack-decision-based-attack)
- [`ProjectedGradientDescent`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#projected-gradient-descent-pgd)
- [`BasicIterativeMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#basic-iterative-method-bim)

Common ART poisoning attack classes used with deckard attack configs:

- [`PoisoningAttackSVM`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/poisoning.html#poisoning-attack-svm)

Common ART extraction attack classes used with deckard attack configs:

- [`CopycatCNN`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/extraction.html#copycat-cnn)

Common ART inference attack classes used with deckard attack configs:

- [`MembershipInferenceBlackBox`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-black-box)
- [`AttributeInferenceBlackBox`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-black-box)
- [`MIFace`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/model_inversion.html#miface-model-inversion-attack)

Common ART defenses that are paired with attack runs in deckard:

- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`SpatialSmoothing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#spatial-smoothing)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

Related Deckard docs:

- {doc}`/api/model/index` for configuring defense pipelines
- {doc}`/api/model/defend` for defense pipeline and defense mixin dispatch
- {doc}`/api/score/index` for attack-specific scorer profiles
- {doc}`/api/pytorch/index` for ART estimator integration in torch workflows
- {doc}`/api/plugins/textattack` and {doc}`/api/plugins/openattack` for text attack plugin runtime integrations
- {doc}`/api/plugins/fairlearn`, {doc}`/api/plugins/lifelines`, and {doc}`/api/plugins/anjana` for additional plugin-specific attack/scoring integrations

## Integrations

- Framework integration: {doc}`/api/pytorch/index`
- Plugin integrations: {doc}`/api/plugins/index`

## Runtime API Surface

Stable runtime entrypoints on {class}`deckard.attack.base.AttackConfig`:

- `run(data, model, files=...)` executes attack orchestration.
- `load(attack_file=..., attack_predictions_file=...)` loads cached artifacts.
- `score(attack_kind=..., y_true=..., y_pred=...)` forwards to scorer runtime.
- `resolve_runtime_attack_config(attack_family, attack_sub_family)` returns
   runtime `*AttackConfig` handlers.
- `resolve_runtime_attack_handler(attack_family, attack_sub_family)` resolves
   the callable runtime attack handler.

### Python API Examples

```python
from deckard.attack import AttackConfig

# Built-in ART attack path.
attack = AttackConfig(name="art.attacks.evasion.FastGradientMethod")
scores = attack.run(data=data_cfg, model=model_cfg, files=files_cfg.as_dict())

# TextAttack plugin path through canonical attack name.
textattack = AttackConfig(
   name="textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019",
)
textattack_handler = textattack.resolve_runtime_attack_handler("evasion", "")

# OpenAttack plugin path through canonical attack name.
openattack = AttackConfig(
   name="OpenAttack.attackers.PWWSAttacker",
)
openattack_handler = openattack.resolve_runtime_attack_handler("evasion", "")
```

### CLI Example

```bash
python -m deckard +attack.name=art.attacks.evasion.FastGradientMethod
```

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
   name: art.attacks.evasion.FastGradientMethod
   attack_params:
      eps: 0.1
   attack_size: 100
```

## Plugin YAML Examples

```yaml
attack:
   name: textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021
   attack_params:
      split: test
      fail_on_error: false
```

```yaml
attack:
   name: OpenAttack.attackers.PWWSAttacker
   attack_params:
      split: test
      fail_on_error: false
```

## Transformers-Oriented Example

```yaml
model:
   _target_: deckard.frameworks.transformers.model.HuggingFacePytorchModelConfig

attack:
   name: textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019
   attack_params:
      split: test
```

## Troubleshooting

- Ensure the selected attack backend matches the active model backend.
- Confirm attack parameters are valid for the chosen ART/Fairlearn attack type.
- Verify the attack receives compatible input shapes and labels.

### See also

- {doc}`/api/experiment/index` — experiment orchestration
- {doc}`/api/model/index` — model configuration and execution
- {doc}`/api/model/defend` — defense pipeline configuration and mixin behavior
- {doc}`/api/data/index` — data loading and split handling
- {doc}`/api/data/sample` — sampling/split strategy definitions
- {doc}`/api/score/index` — attack scoring profiles

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

Framework-specific attack adapters are documented in {doc}`/api/pytorch/index`.
