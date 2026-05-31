# OpenAttack Integration

The OpenAttack integration provides optional text-oriented adversarial attack
execution for Deckard runtime workflows.

## Parent Core Modules and Behavior Deltas

Parent core pages:

- {doc}`../attack/index`
- {doc}`../model/index`
- {doc}`../score/index`

Behavior deltas in this integration:

- canonical plugin-name routing to a dedicated runtime config handler,
- explicit evasion-family routing for `OpenAttack.*` attack names,
- OpenAttack classifier adapter binding over deckard runtime models,
- attack result normalization into deckard score payloads.

## Overview

OpenAttack runtime behavior is implemented through
{class}`deckard.plugins.openattack.attack.OpenAttackConfig`, which is resolved
from canonical plugin attacker names such as
`OpenAttack.attackers.*`.

When optional OpenAttack dependencies are installed, this runtime path executes
through the standard {class}`deckard.attack.base.AttackConfig` orchestration
surface.

Current family scope:

- OpenAttack names map to the `evasion` family.
- `inference`, `poisoning`, `extraction`, and `reconstruction` families are not
   currently provided by OpenAttack runtime handlers.

External references:

- [OpenAttack documentation](https://openattack.readthedocs.io/)
- [OpenAttack attackers](https://openattack.readthedocs.io/en/latest/apis/attackers.html)

## Typical Workflow

1. Configure an attack with a canonical OpenAttack attacker name.
2. Resolve runtime model/tokenizer context from the active model configuration.
3. Execute attacker evaluation and merge normalized outputs into attack scoring payloads.

## Minimal YAML Example

```yaml
attack:
   name: OpenAttack.attackers.PWWSAttacker
   attack_params:
      split: test
      fail_on_error: false
```

## Troubleshooting

- Ensure the optional `OpenAttack` dependency is installed.
- Ensure the attacker token in `attack.name` maps to an available OpenAttack attacker.
- Ensure model/tokenizer inputs are text-compatible for the configured split.

## See also

- {doc}`/api/plugins/index`
- {doc}`/api/attack/index`
- {doc}`/overview/extensions/index`
- {doc}`/notebooks/huggingface`

## API Reference

```{eval-rst}
.. automodule:: deckard.plugins.openattack.attack
   :members:
   :show-inheritance:
```
