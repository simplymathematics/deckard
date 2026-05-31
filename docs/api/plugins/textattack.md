# TextAttack Integration

The TextAttack integration provides optional text-oriented adversarial attack
execution for Deckard runtime workflows.

## Parent Core Modules and Behavior Deltas

Parent core pages:

- {doc}`/api/attack/index`
- {doc}`/api/model/index`
- {doc}`/api/score/index`

Behavior deltas in this integration:

- canonical plugin-name routing to a dedicated runtime config handler,
- explicit evasion-family routing for `textattack.*` attack names,
- transformer/text adapter resolution for model-tokenizer execution,
- text-attack result normalization into deckard attack score payloads.

## Overview

TextAttack runtime behavior is implemented through
{class}`deckard.plugins.textattack.attack.TextAttackConfig`, which is resolved
from canonical plugin attack names such as
`textattack.attack_recipes.*`.

When optional TextAttack dependencies are installed, this runtime path executes
through the standard {class}`deckard.attack.base.AttackConfig` orchestration
surface.

Current family scope:

- TextAttack names map to the `evasion` family.
- [inference](/api/attack/index#inference-attacks),
  [poisoning](/api/attack/index#poisoning-attacks),
  [extraction](/api/attack/index#extraction-attacks), and
  [reconstruction](/api/attack/index#reconstruction-attacks) families are not
  currently provided by TextAttack runtime handlers.

External references:

- [TextAttack documentation](https://textattack.readthedocs.io/)
- [TextAttack attack recipes](https://textattack.readthedocs.io/en/latest/1start/attacks4Components.html)

## Typical Workflow

1. Configure an attack with a canonical TextAttack recipe name.
2. Resolve runtime model/tokenizer context from the active model configuration.
3. Execute recipe attacks and merge normalized outputs into attack scoring payloads.

## Minimal YAML Example

```yaml
attack:
   name: textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021
   attack_params:
      split: test
      fail_on_error: false
      transformation_cache_size: 32768
      constraint_cache_size: 32768
```

## Troubleshooting

- Ensure the optional [textattack](/api/plugins/textattack) dependency is installed.
- Ensure the recipe token in `attack.name` maps to an available TextAttack recipe.
- Ensure model/tokenizer inputs are text-compatible for the configured split.

## See also

- {doc}`/api/plugins/index`
- {doc}`/api/attack/index`
- {doc}`/overview/extensions/index`
- {doc}`/notebooks/huggingface`

## API Reference

```{eval-rst}
.. automodule:: deckard.plugins.textattack.attack
   :members:
   :show-inheritance:
```
