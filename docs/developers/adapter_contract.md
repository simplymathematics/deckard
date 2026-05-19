# Adapter Contract

Adapter mixins in `deckard/frameworks/adapters.py` are the only permitted cross-boundary translation layer between core, framework, and plugin configs.

## Rules
- Adapters must read/write only public (non-underscore-prefixed) attributes of the target config object
- Any state needed by the adapter must be exposed through typed public fields or properties
- Private attributes (`_private`) may only be used as local variables inside the adapter method body
- No adapter may access or mutate a target's private attribute

## Enforcement
- Linting and test assertions must verify that no adapter method accesses private attributes on the target config
- All exchanged values must be inspectable via the public API

---

**Related:** [Core/Framework/Plugin Boundaries](core_framework_plugin_boundaries.md) | [Mixin and Plugin Rules](mixin_plugin_rules.md)
