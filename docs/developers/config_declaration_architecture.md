# Config Declaration Architecture

This document defines the refactor design for [Hydra](https://hydra.cc) config
declaration discovery and registration.

## Canonical Sources

Authoritative config declarations live in:

- `examples/sklearn/configs/`
- `examples/pytorch/configs/`
- External roots provided via `DECKARD_CONFIG_DIRS` (optional)

Plugin declaration families that commonly register through these roots include:

- [Fairlearn plugin API](../api/fairlearn)
- [Lifelines plugin API](../api/lifelines)
- [Seaborn plugin API](../api/seaborn)
- [Yellowbrick plugin API](../api/yellowbrick)
- [Anjana plugin API](../api/anjana)

The `deckard/` Python package should provide runtime registration logic, not
authoritative declaration content.

## Runtime Registration Model

`deckard/declarations.py` is the runtime registration entrypoint.

Expected responsibilities:

- discover config roots (built-in + `DECKARD_CONFIG_DIRS`)
- iterate YAML declaration files under those roots
- parse declarations into [Hydra](https://hydra.cc)-compatible registration metadata
- register declarations dynamically with `safe_store()`
- gate framework/plugin registration on installed optional dependencies

## Hydra Organization

Use canonical [Hydra](https://hydra.cc) config groups with this structure:

- `<group>@<sub-group>/<plugin>_<alias-name>`

Examples:

- `model@defense=fairlearn_exponeniated-gradient.yaml`
- `attack/evasion_fgsm`

YAML declaration file names should use `modified_snake-case.yaml`.
Separate logical groups with `_`.
Use `-` in multi-word aliases.

- Prefer YAML declarations over Python hardcoded `ConfigStore` calls.
- Remove duplicated declarations in module-local declaration files once YAML
  equivalents exist.
- Keep registration deterministic and side-effect-safe for optional dependencies.
- Keep declaration loading extensible for downstream users via `DECKARD_CONFIG_DIRS`.

## Test Strategy

- Compose-first tests: verify [Hydra](https://hydra.cc) composition for
  canonical declarations.
  Use:

```bash
deckard optimize --cfg job
```

- Unit tests: consume canonical composed configs instead of fixture-local
declaration copies.

- Experiment tests: validate end-to-end execution from composed canonical config
  stacks.

______________________________________________________________________

**Related:** [Refactor Plan](refactor_plan) | [Naming
Conventions](naming_conventions) | [Mixin and Plugin Rules](mixin_plugin_rules)
