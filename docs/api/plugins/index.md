# API Plugin Integrations

Plugin integration APIs are grouped here.

Core runtime behavior remains owned by
{class}`deckard.attack.base.AttackConfig`. Plugin integrations are optional
extensions that attach through explicit plugin hooks or canonical plugin attack
paths.

For attack runtime plugins, use canonical names that resolve to dedicated
runtime config handlers:

- `textattack.attack_recipes.*` -> {class}`deckard.plugins.textattack.attack.TextAttackConfig`
- `OpenAttack.attackers.*` -> {class}`deckard.plugins.openattack.attack.OpenAttackConfig`

With optional plugin libraries installed, canonical plugin names resolve
through these runtime handlers without extra base-runtime override wiring.

See {doc}`/api/attack/index` for the flattened AttackConfig runtime model and
built-in versus plugin dispatch boundaries.
See {doc}`/overview/extensions/index` for a narrative extensions map.

- {doc}`/api/plugins/anjana`
- {doc}`/api/plugins/fairlearn`
- {doc}`/api/plugins/textattack`
- {doc}`/api/plugins/openattack`
- {doc}`/api/plugins/lifelines`
- {doc}`/api/plugins/seaborn`
- {doc}`/api/plugins/yellowbrick`

```{toctree}
:hidden:
:maxdepth: 1

anjana
fairlearn
textattack
openattack
lifelines
seaborn
yellowbrick
```
