# Config Class-Doc Contract

This page is the canonical class-doc contract for public `*Config` classes.
Behavior belongs in each owning API page, for example {doc}`data` or {doc}`model`.

## Purpose

Config classes define Deckard's declarative runtime surface. They collect stable
parameters, persistence paths, and composition points so orchestration layers can
instantiate and run experiments without hidden side effects.

## Capabilities

- Bind framework/runtime implementations behind serializable field values.
- Normalize defaults so runtime behavior is reproducible.
- Expose extension hooks via typed fields instead of ad hoc kwargs.
- Enable Hydra/YAML composition for experiment reuse.

## Required Docstring Shape

Public config classes must use MyST-native Google-style docstrings with:

- `Attributes:`
- `Args:` when constructor/runtime parameters are accepted
- `Returns:` when a public method returns non-`None`
- `Raises:` when errors are part of the contract
- `Note:` when execution/persistence semantics are relevant

Use the canonical project rule in {doc}`../developers/docstrings`.

## Attribute Documentation Rules

- Document class-scoped fields that define runtime behavior.
- Prefer stable semantic names over implementation-only details.
- Keep field descriptions short and execution-focused.
- Do not duplicate hook/plugin orchestration semantics here; link to behavior docs.

## Naming and Type Contract

- Name pattern: `<Framework><Type>Config`
- Config classes inherit {class}`deckard.utils.BaseConfig`
- Cross-reference runtime owners using MyST roles, for example {class}`deckard.data.base.DataConfig`.

## See also

- {doc}`../developers/mixins`
- {doc}`../developers/plugins`
- {doc}`../developers/configs`
