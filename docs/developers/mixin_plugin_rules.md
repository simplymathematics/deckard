# Mixin and Plugin Rules

## Mixins
- Must be dataclasses
- Encapsulate reusable behavior, parameters, and logic
- Expose at least one public method
- No top-level orchestration (`__call__`)
- Must use MyST-native Google-style docstrings

## Plugins
- Compose one or more mixins
- Define deterministic execution order
- Implement runtime execution hooks
- Expose a public `__call__` method
- Orchestrate mixin execution deterministically
- No hidden runtime mutation

## Forbidden Patterns
- Undocumented mixins
- Private-only APIs
- Implicit execution contracts
- Untyped runtime payloads
- Hidden orchestration logic
- Side-effect-only mixins

---

**Related:** [Naming Conventions](naming_conventions.md) | [Docstring Standard](docstring_standard.md)
