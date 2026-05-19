# Config Declaration Architecture

This document defines the refactor design for Hydra config declaration discovery and registration.

## Canonical Sources

Authoritative config declarations live in:

- `examples/sklearn/configs/`
- `examples/pytorch/configs/`
- External roots provided via `DECKARD_CONFIG_DIRS` (optional)

The `deckard/` Python package should provide runtime registration logic, not authoritative declaration content.

## Runtime Registration Model

`deckard/declarations.py` is the runtime registration entrypoint.

Expected responsibilities:

- discover config roots (built-in + `DECKARD_CONFIG_DIRS`)
- iterate YAML declaration files under those roots
- parse declarations into Hydra-compatible registration metadata
- register declarations dynamically with `safe_store()`
- gate framework/plugin registration on installed optional dependencies

## Hydra Organization

Use canonical Hydra groups with this structure:

- `<domain>/<type>/<implementation>`

Examples:

- `model/classifier/sklearn-random-forest`
- `attack/evasion/fgsm`

YAML declaration file names should use `snake-case.yaml`.

## Migration Rules

- Prefer YAML declarations over Python hardcoded `ConfigStore` calls.
- Remove duplicated declarations in module-local declaration files once YAML equivalents exist.
- Keep registration deterministic and side-effect-safe for optional dependencies.
- Keep declaration loading extensible for downstream users via `DECKARD_CONFIG_DIRS`.

## Test Strategy

- Compose-first tests: verify Hydra composition for canonical declarations.
- Unit tests: consume canonical composed configs instead of fixture-local declaration copies.
- Experiment tests: validate end-to-end execution from composed canonical config stacks.

---

**Related:** [Refactor Plan](refactor_plan.md) | [Naming Conventions](naming_conventions.md) | [Mixin and Plugin Rules](mixin_plugin_rules.md)
